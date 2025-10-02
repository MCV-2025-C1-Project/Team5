import numpy as np


def validate_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    """Raise if inputs do not share the same shape."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape=} vs {b.shape=}")
