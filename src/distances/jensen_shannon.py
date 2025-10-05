import numpy as np
from src.tools import utils


def compute_js_divergence(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """
    Compute the Jensen-Shannon divergence between two probability distributions.

    Args:
        hist1 (np.ndarray): First histogram (normalized to sum to 1).
        hist2 (np.ndarray): Second histogram (normalized to sum to 1).

    Returns:
        np.float64: Jensen-Shannon divergence in the range [0, 1].
    """
    utils.validate_same_shape(hist1, hist2)

    epsilon = 1e-10
    hist1_safe = hist1 + epsilon
    hist2_safe = hist2 + epsilon

    m = (hist1_safe + hist2_safe) / 2

    kl1 = np.sum(hist1_safe * np.log2(hist1_safe / m))
    kl2 = np.sum(hist2_safe * np.log2(hist2_safe / m))

    return (kl1 + kl2) / 2
