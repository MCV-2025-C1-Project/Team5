import numpy as np
from src.tools import utils

def bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Code for Bhattacharyya distance"""

    utils.validate_same_shape(hist1, hist2)
    
    bc = np.sum(np.sqrt(hist1 * hist2))
    return -np.log(bc + 1e-10)