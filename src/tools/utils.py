import numpy as np


def validate_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    """Raise if inputs do not share the same shape."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape=} vs {b.shape=}")
    
def validate_same_dim(A: np.ndarray, B: np.ndarray) -> None:
    """
    Validate that two input arrays have the same descriptor dimension.

    Args:
        A (np.ndarray): First input array, shape (N, D).
        B (np.ndarray): Second input array, shape (M, D).

    Raises:
        ValueError: If arrays do not have the same descriptor dimension (D).
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Both inputs must be 2D arrays, got A.ndim={A.ndim}, B.ndim={B.ndim}")
    
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Descriptor dimensions must match: got {A.shape[1]} and {B.shape[1]}")


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize an input mask to a boolean HxW array using a simple rule.

    Args:
        mask: Input mask. Accepted formats:
            - HxW boolean or integer array (single-channel)
            - HxWxC array (multi-channel image where any non-zero channel
              indicates foreground)

    Returns:
        np.ndarray: Boolean array with shape (H, W) where True indicates
            foreground pixels.

    Raises:
        ValueError: If ``mask`` has unsupported number of dimensions.
    """
    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)

    if mask.dtype == bool and mask.ndim == 2:
        return mask.copy()

    if mask.ndim == 2:
        return (mask != 0)

    if mask.ndim == 3:
        # Any non-zero value in any channel indicates foreground
        return (mask != 0).any(axis=-1)

    raise ValueError(f"Unsupported mask dimensions: {mask.ndim}")