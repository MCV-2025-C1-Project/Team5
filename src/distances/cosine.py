import numpy as np
from src.tools import utils


def compute_cosine_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the inverse of the Cosine similarity between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Cosine distance (0 = identical, 1 = orthogonal).
    """
    utils.validate_same_shape(hist1, hist2)
    dot_product = np.dot(hist1, hist2)
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)
    epsilon = 1e-10
    cosine_sim = dot_product / (norm1 * norm2 + epsilon)
    return 1.0 - cosine_sim
