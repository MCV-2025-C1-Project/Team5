import numpy as np
from src.tools import utils

def canberra_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compute the Canberra distance between two histograms.
    Weighted version of L1 distance, sensitive to small changes near zero.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        float: Canberra distance between the two histograms.
    """

    utils.validate_same_shape(hist1, hist2)

    epsilon = 1e-10
    numerator = np.abs(hist1 - hist2)
    denominator = np.abs(hist1) + np.abs(hist2) + epsilon
    return np.sum(numerator / denominator)
