import numpy as np
from src.tools import utils

def compute_hellinger_kernel(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the Hellinger kernel
    between two normalized histograms.

    Args:
        hist1 (np.ndarray): First histogram (should be L1-normalized).
        hist2 (np.ndarray): Second histogram (should be L1-normalized).

    Returns:
        np.float64: Hellinger kernel similarity (1 = identical, 0 = no overlap).
    """
    utils.validate_same_shape(hist1, hist2)
    return np.sum(np.sqrt(hist1 * hist2))
