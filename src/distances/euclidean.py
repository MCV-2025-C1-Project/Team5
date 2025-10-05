import numpy as np

from src.tools import utils


def compute_euclidean_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the Euclidean (L2) distance between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Euclidean distance between the two histograms.
    """
    utils.validate_same_shape(hist1, hist2)
    return np.sqrt(np.sum(np.square(hist1 - hist2)))
