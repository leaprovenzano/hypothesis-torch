from typing import Any
from hypothesis import given
from hypothesis import strategies as st

from hypothesis_torch.sizes import is_valid_dim

from tests.utils import param, mark_params


@mark_params
@param(tag="ints < 1", inp=st.integers(max_value=0), expected=False)
@param(tag="floats", inp=st.floats(max_value=0), expected=False)
@param(tag="ints >= 1", inp=st.floats(min_value=1), expected=False)
@given(data=st.data())
def test_is_valid_dim(data, inp: Any, expected: bool):
    drawn = data.draw(inp)
    assert is_valid_dim(drawn) is expected
