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

def compute_cosine_distance_matrix(A, B, batch_size=None):
    """
    Compute pairwise Cosine distances (1 - cosine similarity).

    Args:
        A (np.ndarray): First set of vectors, shape (N, D).
        B (np.ndarray): Second set of vectors, shape (M, D).
        batch_size (int | None): If given, compute in batches.

    Returns:
        np.ndarray: Cosine distances, shape (N, M).
    """
    utils.validate_same_dim(A,B)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    if batch_size is None:
        cosine_sim = np.dot(A_norm, B_norm.T)
    else:
        N, M = A.shape[0], B.shape[0]
        cosine_sim = np.zeros((N, M), dtype=np.float32)
        for i in range(0, N, batch_size):
            Ai = A_norm[i:i + batch_size]
            cosine_sim[i:i + len(Ai)] = np.dot(Ai, B_norm.T)
    return 1.0 - cosine_sim

