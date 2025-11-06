import numpy as np
from src.tools import utils


def compute_js_divergence(hist1: np.ndarray, hist2: np.ndarray) -> np.float64:
    """
    Compute the Jensen-Shannon divergence between two probability distributions.

    Args:
        hist1 (np.ndarray): First histogram (normalized to sum to 1).
        hist2 (np.ndarray): Second histogram (normalized to sum to 1).

    Returns:
        np.float64: Jensen-Shannon divergence in the range [0, 1].
    """
    utils.validate_same_shape(hist1, hist2)

    epsilon = 1e-10
    hist1_safe = hist1 + epsilon
    hist2_safe = hist2 + epsilon

    m = (hist1_safe + hist2_safe) / 2

    kl1 = np.sum(hist1_safe * np.log2(hist1_safe / m))
    kl2 = np.sum(hist2_safe * np.log2(hist2_safe / m))

    return (kl1 + kl2) / 2

def compute_js_divergence_matrix(A: np.ndarray, B: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """
    Compute Jensen–Shannon divergence between two sets of probability histograms.

    Args:
        A (np.ndarray): First normalized histogram set, shape (N, D).
        B (np.ndarray): Second normalized histogram set, shape (M, D).
        batch_size (int | None): Number of rows from A to process per batch.

    Returns:
        np.ndarray: Jensen–Shannon divergence matrix, shape (N, M).
    """
    utils.validate_same_dim(A, B)
    epsilon = 1e-10
    A_safe, B_safe = A + epsilon, B + epsilon
    N, M = A.shape[0], B.shape[0]
    D = np.zeros((N, M), dtype=np.float32)

    if batch_size is None:
        Mmix = 0.5 * (A_safe[:, None, :] + B_safe[None, :, :])
        kl1 = np.sum(A_safe[:, None, :] * np.log2(A_safe[:, None, :] / Mmix), axis=2)
        kl2 = np.sum(B_safe[None, :, :] * np.log2(B_safe[None, :, :] / Mmix), axis=2)
        D = 0.5 * (kl1 + kl2)
    else:
        for i in range(0, N, batch_size):
            Ai = A_safe[i:i + batch_size]
            Mmix = 0.5 * (Ai[:, None, :] + B_safe[None, :, :])
            kl1 = np.sum(Ai[:, None, :] * np.log2(Ai[:, None, :] / Mmix), axis=2)
            kl2 = np.sum(B_safe[None, :, :] * np.log2(B_safe[None, :, :] / Mmix), axis=2)
            D[i:i + len(Ai)] = 0.5 * (kl1 + kl2)
    return D

