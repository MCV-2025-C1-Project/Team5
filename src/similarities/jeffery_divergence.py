import numpy as np
from src.tools import utils

def jeffrey_divergence(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Code for Jeffrey divergence (symmetric version of KL divergence).
    Also known as Jensen-Shannon divergence.
    """

    utils.validate_same_shape(hist1, hist2)

    epsilon = 1e-10
    hist1_safe = hist1 + epsilon
    hist2_safe = hist2 + epsilon
    
    # Jeffrey divergence = KL(hist1 || m) + KL(hist2 || m)
    # where m = (hist1 + hist2) / 2
    m = (hist1_safe + hist2_safe) / 2
    
    kl1 = np.sum(hist1_safe * np.log(hist1_safe / m))
    kl2 = np.sum(hist2_safe * np.log(hist2_safe / m))
    
    return (kl1 + kl2) / 2