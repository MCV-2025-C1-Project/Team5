import numpy as np
from src.tools import utils


def compute_cosine_similarity(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the Cosine similarity between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Cosine similarity (1 = identical, 0 = orthogonal).
    """
    utils.validate_same_shape(hist1, hist2)
    dot = np.dot(hist1, hist2)
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)
    if norm1 == 0 or norm2 == 0:
        return 0.0  # avoid division by zero
    return dot / (norm1 * norm2)