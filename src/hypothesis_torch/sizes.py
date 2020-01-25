from typing import Any, Sequence, Union


import torch

from hypothesis import strategies as st
from hypothesis.strategies._internal.core import cacheable

from hypothesis.errors import InvalidArgument


def is_valid_dim(x: Any) -> bool:
    """determine if the argument will be valid dim when included in torch.Size.
    """
    return isinstance(x, int) and x > 0


class DimStrategy(st.SearchStrategy[int]):

    """A strategy for representing variable dimensions within tensor shapes.
    """

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size
        self._inner_strat = st.integers(self.min_size, self.max_size)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(min_size, max_size={self.max_size})'

    def do_draw(self, data) -> int:
        return data.draw(self._inner_strat)


class SizeStrategy(st.SearchStrategy[torch.Size]):

    def __init__(self, elements: st.SearchStrategy):
        self.element_strategy = elements

    def __repr__(self):
        return 'SizeStrategy({", ".join([repr(e) for e in self.element_strategies]))})'

    def do_draw(self, data) -> torch.Size:
        return torch.Size(data.draw(self.element_strategy))

    def do_validate(self):
        return self.element_strategy.validate()

    def calc_has_reusable_values(self, recur):
        return self.element_strategy.calc_has_reusable_values(recur)

    def calc_is_empty(self, recur):
        return self.element_strategy.calc_is_empty(recur)


@cacheable
def dims(min_size: int = 1, max_size: int = 1000) -> DimStrategy:
    """Strategy for generating variable sized dims

    Example:
        >>> import random
        >>> random.seed(0)
        >>> from hypothesis_torch.sizes import dims
        >>>
        >>> strat = dims(2, 10)
        >>> strat.example()
        9
    """
    if not is_valid_dim(min_size):
        raise InvalidArgument('min_size must be an integer greater than 0')
    if not is_valid_dim(max_size):
        raise InvalidArgument('max_size must be an integer greater than 0')
    if min_size > max_size:
        raise InvalidArgument('min_size must be < max_size')
    return DimStrategy(min_size, max_size)


def sizes(*dims: Sequence[Union[int, DimStrategy]]) -> SizeStrategy:
    """A strategy for creating torch.Size objects

    Example:
        >>> import random
        >>> random.seed(0)
        >>>
        >>> import torch
        >>> from hypothesis_torch.sizes import sizes, dims
        >>>
        >>> size_strat = sizes(1, 2)
        >>> size_strat.example()
        torch.Size([1, 2])

        more usefully...
        >>> from hypothesis_torch.sizes import dims
        >>>
        >>> variable_size_strat = sizes(dims(), 5, dims(max_size=10))
        >>> variable_size_strat.example()
        torch.Size([516, 5, 9])

        or use with no arguments to get any old sizes with variable dims...
        >>> variable_dim_strat = sizes()
        >>> variable_dim_strat.example()
        torch.Size([46, 415, 809, 788])
    """
    if not dims:
        return SizeStrategy(st.lists(st.integers(1, 1000), min_size=1, max_size=10))
    else:
        dim_strats = []
        for d in dims:
            if not isinstance(d, DimStrategy):
                if is_valid_dim(d):
                    dim_strats.append(st.just(d))
                else:
                    raise InvalidArgument('dims must be either positive integers or DimStrategy')
            else:
                dim_strats.append(d)
    return SizeStrategy(st.tuples(*dim_strats))
