import numpy as np
from src.tools import utils


def compute_hellinger_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the Hellinger distance between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Hellinger distance between the two histograms
            (0 = identical, higher = more different).
    """
    utils.validate_same_shape(hist1, hist2)
    return np.sqrt(0.5 * np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2))
