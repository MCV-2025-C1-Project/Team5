import numpy as np

def compute_cosine_similarity(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the Cosine similarity between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Cosine similarity (1 = identical, 0 = orthogonal).
    """
    dot_product = np.dot(hist1, hist2)
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)
    epsilon = 1e-10
    cosine_sim = dot_product / (norm1 * norm2 + epsilon)
    return 1.0 - cosine_sim