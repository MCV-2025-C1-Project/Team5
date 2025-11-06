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

def compute_euclidean_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between two sets of descriptors.

    Args:
        A (np.ndarray): First set of descriptors of shape (N, D)
        B (np.ndarray): Second set of descriptors of shape (M, D)

    Returns:
        np.ndarray: Distance matrix of shape (N, M),
                    where element (i, j) is the Euclidean distance
                    between A[i] and B[j].
    """
    dist_matrix = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    return dist_matrix
