import cv2
import math
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


def compute_histogram(
    img: np.ndarray,
        values_per_bin: int = 1,
        density: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the histogram of a grayscale image with adjustable bin size.

    Args:
        img (np.ndarray): Input grayscale image.
        values_per_bin (int, optional): Number of consecutive intensity values 
            grouped into each bin. For example, 1 creates 256 bins 
            (0–255 individually), 2 creates 128 bins (0–1, 2–3, ...). 
            Must be >= 1. Defaults to 1.
        density (bool, optional): If True, normalize the histogram so the 
            total area equals 1. If False, return raw counts. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - Histogram values (counts or normalized probabilities).
            - Bin edges corresponding to the histogram.

    Raises:
        ValueError: If `values_per_bin` is less than 1 or larger than 256.
    """
    if (values_per_bin < 1) or (values_per_bin > 256):
        raise ValueError("values_per_bin must be >= 1 and <=256")

    bins = math.ceil(256 / values_per_bin)
    max_range = bins * values_per_bin

    hist, bin_edges = np.histogram(
        img, bins=bins, range=[0, max_range], density=density)
    return hist, bin_edges
