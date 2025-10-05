"""
HSV concatenated histogram descriptor.
"""

import cv2
import numpy as np 
from .histogram import compute_histogram
from src.data.extract import read_image
from typing import Tuple

def convert_img_to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to HSV.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: HSV image.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def compute_hsv_histogram(img_path: str, values_per_bin: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HSV concatenated histogram from image path.

    Args:
        img_path: Path to the image file.
        values_per_bin: Number of intensity values per bin.

    Returns:
        concat_hist: Normalized concatenated HSV histogram.
        bin_edges : Bin edges for the histograms (all channels share the same edges).
    """
    img = read_image(img_path)

    # Convert BGR to HSV
    img_hsv = convert_img_to_hsv(img)

    # Split channels (OpenCV uses BGR order)
    h_channel = img_hsv[:, :, 0]
    s_channel = img_hsv[:, :, 1]
    v_channel = img_hsv[:, :, 2]

    # Compute histogram for each channel
    hist_h, bin_edges = compute_histogram(h_channel, values_per_bin=values_per_bin, density=True)
    hist_s, _ = compute_histogram(s_channel, values_per_bin=values_per_bin, density=True)
    hist_v, _ = compute_histogram(v_channel, values_per_bin=values_per_bin, density=True)

    # Concatenate in HSV order
    concat_hist = np.concatenate([hist_h, hist_s, hist_v])

    return concat_hist, bin_edges