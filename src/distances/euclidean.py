import numpy as np

from src.tools import utils

def compute_euclidean_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the Euclidean (L2) distance between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Euclidean distance between the two histograms.
    """
    utils.validate_same_shape(hist1, hist2)
    return np.sqrt(np.sum(np.square(hist1 - hist2)))

def compute_euclidean_distance_matrix(A, B, batch_size=None):
    """
    Compute pairwise Euclidean (L2) distances between two sets of vectors.

    Args:
        A (np.ndarray): First set of vectors, shape (N, D).
        B (np.ndarray): Second set of vectors, shape (M, D).
        batch_size (int | None): If given, compute in batches to save memory.

    Returns:
        np.ndarray: Pairwise Euclidean distances, shape (N, M).
    """
    utils.validate_same_dim(A,B)
    if batch_size is None:
        # fully vectorized
        return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    else:
        N, M = A.shape[0], B.shape[0]
        D = np.zeros((N, M), dtype=np.float32)
        for i in range(0, N, batch_size):
            Ai = A[i:i + batch_size]
            D[i:i + len(Ai)] = np.sqrt(((Ai[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
        return D

