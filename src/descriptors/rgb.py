"""
RGB concatenated histogram descriptor.
"""

import cv2
import numpy as np
from .histogram import compute_histogram
from src.data.extract import read_image
from typing import Tuple

def convert_img_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: RGB image.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def compute_rgb_histogram(img_path: str, values_per_bin: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RGB concatenated histogram from image path.

    Args:
        img_path: Path to the image file.
        values_per_bin: Number of intensity values per bin.

    Returns:
        concat_hist : Concatenated histogram (R | G | B), each channel with bins depending on values_per_bin.
        bin_edges : Bin edges for the histograms (all channels share the same edges).
    """
    img_bgr = read_image(img_path)

    # Split channels (OpenCV uses BGR order)
    b_channel = img_bgr[:, :, 0]
    g_channel = img_bgr[:, :, 1]
    r_channel = img_bgr[:, :, 2]

    # Compute histogram for each channel
    hist_b, bin_edges = compute_histogram(b_channel, values_per_bin=values_per_bin, density=True)
    hist_g, _ = compute_histogram(g_channel, values_per_bin=values_per_bin, density=True)
    hist_r, _ = compute_histogram(r_channel, values_per_bin=values_per_bin, density=True)

    # Concatenate in RGB order (not BGR)
    concat_hist = np.concatenate([hist_r, hist_g, hist_b])

    return concat_hist, bin_edges
