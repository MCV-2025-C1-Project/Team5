import numpy as np
from src.tools import utils


def compute_hellinger_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """Compute the Hellinger distance between two histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Hellinger distance between the two histograms
            (0 = identical, higher = more different).
    """
    utils.validate_same_shape(hist1, hist2)
    return np.sqrt(0.5 * np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2))

def compute_hellinger_distance_matrix(A: np.ndarray, B: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """
    Compute pairwise Hellinger distances between histograms.

    Args:
        A (np.ndarray): First histogram set, shape (N, D).
        B (np.ndarray): Second histogram set, shape (M, D).
        batch_size (int | None): Number of rows from A to process per batch.
            If None, process all at once.

    Returns:
        np.ndarray: Hellinger distance matrix, shape (N, M).
    """
    utils.validate_same_dim(A,B)
    A_sqrt, B_sqrt = np.sqrt(A), np.sqrt(B)

    N, M = A.shape[0], B.shape[0]
    D = np.zeros((N, M), dtype=np.float32)

    if batch_size is None:
        D = np.sqrt(0.5 * np.sum((A_sqrt[:, None, :] - B_sqrt[None, :, :]) ** 2, axis=2))
    else:
        for i in range(0, N, batch_size):
            Ai = A_sqrt[i:i + batch_size]
            D[i:i + len(Ai)] = np.sqrt(0.5 * np.sum((Ai[:, None, :] - B_sqrt[None, :, :]) ** 2, axis=2))
    return D

