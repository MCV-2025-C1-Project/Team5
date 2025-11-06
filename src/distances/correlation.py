import numpy as np
from src.tools import utils


def correlation_distance(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """
    Compute Correlation distance (1 - correlation coefficient).
    Measures linear relationship between histograms.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.

    Returns:
        np.float64: Correlation distance between the two histograms.
    """
    utils.validate_same_shape(hist1, hist2)

    # Pearson correlation coefficient
    hist1_centered = hist1 - np.mean(hist1)
    hist2_centered = hist2 - np.mean(hist2)

    numerator = np.sum(hist1_centered * hist2_centered)
    denominator = np.sqrt(np.sum(hist1_centered**2)
                          * np.sum(hist2_centered**2))

    if denominator < 1e-10:
        return 1.0  # Completely uncorrelated

    correlation = numerator / denominator
    return 1.0 - correlation

def compute_correlation_distance_matrix(A: np.ndarray, B: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """
    Compute pairwise correlation distances (1 - Pearson correlation coefficient).

    Args:
        A (np.ndarray): First feature set, shape (N, D).
        B (np.ndarray): Second feature set, shape (M, D).
        batch_size (int | None): Number of rows from A to process per batch.

    Returns:
        np.ndarray: Correlation distance matrix, shape (N, M).
    """
    utils.validate_same_dim(A, B)
    A_centered = A - A.mean(axis=1, keepdims=True)
    B_centered = B - B.mean(axis=1, keepdims=True)
    N, M = A.shape[0], B.shape[0]
    D = np.zeros((N, M), dtype=np.float32)

    if batch_size is None:
        num = np.dot(A_centered, B_centered.T)
        denom = np.outer(np.linalg.norm(A_centered, axis=1), np.linalg.norm(B_centered, axis=1)) + 1e-10
        D = 1.0 - (num / denom)
    else:
        for i in range(0, N, batch_size):
            Ai = A_centered[i:i + batch_size]
            num = np.dot(Ai, B_centered.T)
            denom = np.outer(np.linalg.norm(Ai, axis=1), np.linalg.norm(B_centered, axis=1)) + 1e-10
            D[i:i + len(Ai)] = 1.0 - (num / denom)
    return D
