import numpy as np

def hellinger_kernel(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Hellinger distance"""
    return np.sqrt(0.5 * np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2))
