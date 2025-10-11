"""
Grayscale histogram descriptor.
"""

import cv2
import numpy as np
from src.descriptors.histogram import compute_histogram
from src.data.extract import read_image
from typing import Tuple


def convert_img_to_gray_scale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Grayscale image.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def compute_grayscale_histogram_from_array(
    img_bgr: np.ndarray,
    values_per_bin: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute grayscale histogram from image array.

    Args:
        img_bgr: Input image array in BGR format.
        values_per_bin: Number of intensity values per bin.

    Returns:
        hist: Normalized histogram.
        bin_edges: Bin edges for the histogram.
    """
    img_gray = convert_img_to_gray_scale(img_bgr)
    hist, bin_edges = compute_histogram(
        img_gray, values_per_bin=values_per_bin, density=True)

    return hist, bin_edges


def compute_grayscale_histogram(
    img_path: str,
    values_per_bin: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute grayscale histogram from image path.

    Args:
        img_path: Path to the image file.
        values_per_bin: Number of intensity values per bin.

    Returns:
        hist: Normalized histogram.
        bin_edges: Bin edges for the histogram.
    """
    img = read_image(img_path)

    return compute_grayscale_histogram_from_array(img, values_per_bin)
