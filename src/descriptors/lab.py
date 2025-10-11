"""
LAB concatenated histogram descriptor.
"""

from typing import Tuple
import cv2
import numpy as np
from src.descriptors.histogram import compute_histogram
from src.data.extract import read_image


def convert_img_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to LAB.

    Args:
        img_bgr (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: LAB image.
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return img_lab

def compute_lab_histogram_from_array(
    img_bgr: np.ndarray,
    values_per_bin: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LAB concatenated histogram from image array.

    Args:
        img_bgr: Input image array in BGR format.
        values_per_bin: Number of intensity values per bin.

    Returns:
        concat_hist: Normalized concatenated LAB histogram.
        bin_edges : Bin edges for the histograms (all channels share the same edges).
    """
    # Convert BGR to LAB
    img_lab = convert_img_to_lab(img_bgr)

    # Split channels (OpenCV uses BGR order)
    l_channel = img_lab[:, :, 0]
    a_channel = img_lab[:, :, 1]
    b_channel = img_lab[:, :, 2]

    # Compute histogram for each channel
    hist_l, bin_edges = compute_histogram(
        l_channel, values_per_bin=values_per_bin, density=True)
    hist_a, _ = compute_histogram(
        a_channel, values_per_bin=values_per_bin, density=True)
    hist_b, _ = compute_histogram(
        b_channel, values_per_bin=values_per_bin, density=True)

    # Concatenate in LAB order
    concat_hist = np.concatenate([hist_l, hist_a, hist_b])

    return concat_hist, bin_edges


def compute_lab_histogram(
    img_path: str,
    values_per_bin: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LAB concatenated histogram from image path.

    Args:
        img_path: Path to the image file.
        values_per_bin: Number of intensity values per bin.

    Returns:
        concat_hist: Normalized concatenated LAB histogram.
        bin_edges : Bin edges for the histograms (all channels share the same edges).
    """
    img = read_image(img_path)

    return compute_lab_histogram_from_array(img, values_per_bin) 