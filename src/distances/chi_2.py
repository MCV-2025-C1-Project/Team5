import numpy as np
from src.tools import utils


def compute_chi_2_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the Chi-squared distance between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Chi-squared distance between the two histograms.
    """
    utils.validate_same_shape(hist1, hist2)
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    return np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + epsilon))
