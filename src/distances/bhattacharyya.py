import numpy as np
from src.tools import utils


def bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """
    Compute Bhattacharyya distance between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Bhattacharyya distance between two histograms.
    """
    utils.validate_same_shape(hist1, hist2)

    bc = np.sum(np.sqrt(hist1 * hist2))
    return -np.log(bc + 1e-10)
