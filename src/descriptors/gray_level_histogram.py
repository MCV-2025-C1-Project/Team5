import cv2
import numpy as np


def convert_img_to_gray_scale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Grayscale image.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def compute_histogram(img: np.ndarray, density: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute the histogram of a grayscale image.

    Args:
        img (np.ndarray): Input grayscale image.
        density (bool, optional): If True, normalize the histogram so that the 
            area under the histogram integrates to 1. If False, return raw counts. 
            Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - Histogram values (counts or normalized probabilities).
            - Bin edges corresponding to the histogram.
    """
    hist, bin_edges = np.histogram(img, 256, [0, 256], density=density)
    return hist, bin_edges
