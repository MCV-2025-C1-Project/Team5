import numpy as np

from src.tools import utils


def compute_histogram_intersection_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the histogram intersection between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Histogram intersection value between the two histograms.
    """
    utils.validate_same_shape(hist1, hist2)
    intersection = np.sum(np.minimum(hist1, hist2))
    return 1.0 - intersection

def compute_histogram_intersection_matrix(A: np.ndarray, B: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """
    Compute histogram intersection distances (1 - intersection).

    Args:
        A (np.ndarray): First histogram set, shape (N, D).
        B (np.ndarray): Second histogram set, shape (M, D).
        batch_size (int | None): Number of rows from A to process per batch.

    Returns:
        np.ndarray: Histogram intersection distances, shape (N, M).
    """
    utils.validate_same_dim(A,B)
    N, M = A.shape[0], B.shape[0]
    D = np.zeros((N, M), dtype=np.float32)

    if batch_size is None:
        D = 1.0 - np.sum(np.minimum(A[:, None, :], B[None, :, :]), axis=2)
    else:
        for i in range(0, N, batch_size):
            Ai = A[i:i + batch_size]
            D[i:i + len(Ai)] = 1.0 - np.sum(np.minimum(Ai[:, None, :], B[None, :, :]), axis=2)
    return D
