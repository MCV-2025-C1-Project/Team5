import numpy as np
from src.tools import utils


def bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """
    Compute Bhattacharyya distance between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Bhattacharyya distance between two histograms.
    """
    utils.validate_same_shape(hist1, hist2)

    bc = np.sum(np.sqrt(hist1 * hist2))
    return -np.log(bc + 1e-10)

def compute_bhattacharyya_distance_matrix(A: np.ndarray, B: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """
    Compute pairwise Bhattacharyya distances between two sets of histograms.

    Args:
        A (np.ndarray): First histogram set, shape (N, D).
        B (np.ndarray): Second histogram set, shape (M, D).
        batch_size (int | None): Number of rows from A to process per batch.

    Returns:
        np.ndarray: Bhattacharyya distance matrix, shape (N, M).
    """
    utils.validate_same_dim(A, B)
    N, M = A.shape[0], B.shape[0]
    D = np.zeros((N, M), dtype=np.float32)

    if batch_size is None:
        bc = np.sum(np.sqrt(A[:, None, :] * B[None, :, :]), axis=2)
        D = -np.log(bc + 1e-10)
    else:
        for i in range(0, N, batch_size):
            Ai = A[i:i + batch_size]
            bc = np.sum(np.sqrt(Ai[:, None, :] * B[None, :, :]), axis=2)
            D[i:i + len(Ai)] = -np.log(bc + 1e-10)
    return D
