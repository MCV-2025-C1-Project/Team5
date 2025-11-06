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

def compute_chi_2_distance_matrix(A, B, batch_size=None):
    """
    Compute pairwise Chi-squared distances between histograms.

    Args:
        A (np.ndarray): First histogram set, shape (N, D).
        B (np.ndarray): Second histogram set, shape (M, D).
        batch_size (int | None): Batch size to process in chunks.

    Returns:
        np.ndarray: Chi-squared distance matrix, shape (N, M).
    """
    utils.validate_same_dim(A, B)
    epsilon = 1e-10
    if batch_size is None:
        return np.sum(((A[:, None, :] - B[None, :, :]) ** 2) / (A[:, None, :] + B[None, :, :] + epsilon), axis=2)
    else:
        N, M = A.shape[0], B.shape[0]
        D = np.zeros((N, M), dtype=np.float32)
        for i in range(0, N, batch_size):
            Ai = A[i:i + batch_size]
            D[i:i + len(Ai)] = np.sum(((Ai[:, None, :] - B[None, :, :]) ** 2) /
                                      (Ai[:, None, :] + B[None, :, :] + epsilon), axis=2)
        return D

