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


def compute_histogram(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the histogram of an image.

    Args:
        img (np.ndarray): Input grayscale image.

    Returns:
        tuple[np.ndarray, np.ndarray]: Histogram counts and bin edges.
    """
    hist, bin_edges = np.histogram(img, 256, [0, 256])
    return hist, bin_edges
