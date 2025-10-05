import numpy as np
from src.tools import utils


def compute_canberra_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """
    Compute the Canberra distance between two histograms.
    Weighted version of L1 distance, sensitive to small changes near zero.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Canberra distance between the two histograms.
    """
    utils.validate_same_shape(hist1, hist2)

    epsilon = 1e-10
    numerator = np.abs(hist1 - hist2)
    denominator = np.abs(hist1) + np.abs(hist2) + epsilon
    return np.sum(numerator / denominator)
