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


def compute_hamming_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute pairwise Hamming distances between two sets of binary descriptors.

    Args:
        A (np.ndarray): Array of shape (N, D)
        B (np.ndarray): Array of shape (M, D)

    Returns:
        np.ndarray: Distance matrix of shape (N, M)
    """
    A = A.astype(bool)
    B = B.astype(bool)
    # Broadcasting comparison: XOR → True where bits differ
    return np.bitwise_xor(A[:, None, :], B[None, :, :]).sum(axis=2)

