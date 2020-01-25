from typing import Any


def is_valid_dim(x: Any) -> bool:
    """determine if the argument will be valid dim when included in torch.Size.
    """
    return isinstance(x, int) and x > 0
