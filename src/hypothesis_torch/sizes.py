from typing import Any

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
