import numpy as np

from src.tools import utils


def compute_l1_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the L1 (Manhattan) distance between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: L1 distance between the two histograms.
    """
    utils.validate_same_shape(hist1, hist2)
    return np.sum(np.abs(hist1 - hist2))
