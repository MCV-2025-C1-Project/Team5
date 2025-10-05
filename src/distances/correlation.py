import numpy as np
from src.tools import utils


def correlation_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """
    Compute Correlation distance (1 - correlation coefficient).
    Measures linear relationship between histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Correlation distance between the two histograms.
    """
    utils.validate_same_shape(hist1, hist2)

    # Pearson correlation coefficient
    hist1_centered = hist1 - np.mean(hist1)
    hist2_centered = hist2 - np.mean(hist2)

    numerator = np.sum(hist1_centered * hist2_centered)
    denominator = np.sqrt(np.sum(hist1_centered**2)
                          * np.sum(hist2_centered**2))

    if denominator < 1e-10:
        return 1.0  # Completely uncorrelated

    correlation = numerator / denominator
    return 1.0 - correlation
