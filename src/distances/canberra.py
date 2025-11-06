import numpy as np
from src.tools import utils


def compute_canberra_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """
    Compute the Canberra distance between two histograms.
    Weighted version of L1 distance, sensitive to small changes near zero.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Canberra distance between the two histograms.
    """
    utils.validate_same_shape(hist1, hist2)

    epsilon = 1e-10
    numerator = np.abs(hist1 - hist2)
    denominator = np.abs(hist1) + np.abs(hist2) + epsilon
    return np.sum(numerator / denominator)

def compute_canberra_distance_matrix(A: np.ndarray, B: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """
    Compute pairwise Canberra distances between two sets of histograms.

    Args:
        A (np.ndarray): First histogram set, shape (N, D).
        B (np.ndarray): Second histogram set, shape (M, D).
        batch_size (int | None): Number of rows from A to process per batch.

    Returns:
        np.ndarray: Canberra distance matrix, shape (N, M).
    """
    utils.validate_same_dim(A, B)
    epsilon = 1e-10
    N, M = A.shape[0], B.shape[0]
    D = np.zeros((N, M), dtype=np.float32)

    if batch_size is None:
        numerator = np.abs(A[:, None, :] - B[None, :, :])
        denominator = np.abs(A[:, None, :]) + np.abs(B[None, :, :]) + epsilon
        D = np.sum(numerator / denominator, axis=2)
    else:
        for i in range(0, N, batch_size):
            Ai = A[i:i + batch_size]
            numerator = np.abs(Ai[:, None, :] - B[None, :, :])
            denominator = np.abs(Ai[:, None, :]) + np.abs(B[None, :, :]) + epsilon
            D[i:i + len(Ai)] = np.sum(numerator / denominator, axis=2)
    return D
