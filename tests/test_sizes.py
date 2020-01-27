from typing import Any

import torch

import pytest

from hypothesis import given
from hypothesis import strategies as st

from hypothesis_torch.sizes import is_valid_dim, dims, sizes, InvalidArgument

from tests.utils import param, mark_params


def drawif(data, value):
    if isinstance(value, st.SearchStrategy):
        return data.draw(value)
    return value


@mark_params
@param(tag='ints < 1', inp=st.integers(max_value=0), expected=False)
@param(tag='floats', inp=st.floats(max_value=0), expected=False)
@param(tag='ints >= 1', inp=st.floats(min_value=1), expected=False)
@given(data=st.data())
def test_is_valid_dim(data, inp: Any, expected: bool):
    drawn = data.draw(inp)
    assert is_valid_dim(drawn) is expected


@mark_params
@param(
    tag='min_size > max_size', min_size=10, max_size=9, expected_msg='min_size must be < max_size'
)
@param(
    tag='min_size < 1',
    min_size=st.integers(max_value=0),
    max_size=1000,
    expected_msg='min_size must be an integer greater than 0',
)
@given(data=st.data())
def test_dim_strategy_invalid_init(data, min_size, max_size, expected_msg):
    min_size = drawif(data, min_size)
    max_size = drawif(data, max_size)
    print(min_size, max_size)
    with pytest.raises(InvalidArgument, match=expected_msg):
        dims(min_size, max_size)


@given(dim=dims(10, 100))
def test_dim_strat_valid_init(dim):
    assert isinstance(dim, int)
    assert dim >= 10
    assert dim <= 100


@given(dim=dims())
def test_dim_strat_with_default_init(dim):
    assert isinstance(dim, int)
    assert dim >= 1
    assert dim <= 1000


@given(sizes())
def test_sizes_with_no_args(size):
    assert isinstance(size, torch.Size)
    assert torch.zeros(size).shape == size
