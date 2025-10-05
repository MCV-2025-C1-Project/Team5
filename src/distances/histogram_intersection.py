import numpy as np

from src.tools import utils


def compute_histogram_intersection_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the histogram intersection between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Histogram intersection value between the two histograms.
    """
    utils.validate_same_shape(hist1, hist2)
    intersection = np.sum(np.minimum(hist1, hist2))
    return 1.0 - intersection
