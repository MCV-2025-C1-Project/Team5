import numpy as np
from src.tools import utils

def compute_hamming_distance(desc1: np.ndarray, desc2: np.ndarray) -> np.float64:
    """Compute the Hamming distance between two binary descriptors.

    Args:
        desc1 (np.ndarray): First binary descriptor (values 0/1 or bool).
        desc2 (np.ndarray): Second binary descriptor (values 0/1 or bool).

    Returns:
        np.float64: Hamming distance between the two binary descriptors.
    """
    utils.validate_same_shape(desc1, desc2)
    desc1 = desc1.astype(bool)
    desc2 = desc2.astype(bool)
    return np.sum(desc1 != desc2)

def compute_hamming_distance_matrix(A: np.ndarray, B: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """
    Compute pairwise Hamming distances between two sets of binary descriptors.

    Args:
        A (np.ndarray): Binary descriptor matrix, shape (N, D), values in {0,1} or bool.
        B (np.ndarray): Binary descriptor matrix, shape (M, D), values in {0,1} or bool.
        batch_size (int | None): Number of rows from A to process per batch.
            If None, compute all at once (default).

    Returns:
        np.ndarray: Pairwise Hamming distance matrix, shape (N, M).
    """
    utils.validate_same_dim(A, B)

    A = A.astype(bool)
    B = B.astype(bool)
    N, M = A.shape[0], B.shape[0]
    D = np.zeros((N, M), dtype=np.int32)

    if batch_size is None:
        # Fully vectorized (fastest)
        D = np.bitwise_xor(A[:, None, :], B[None, :, :]).sum(axis=2)
    else:
        # Memory-safe version
        for i in range(0, N, batch_size):
            Ai = A[i:i + batch_size]
            D[i:i + len(Ai)] = np.bitwise_xor(Ai[:, None, :], B[None, :, :]).sum(axis=2)
    return D


