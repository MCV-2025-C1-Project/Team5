"""
YCbCr concatenated histogram descriptor.
"""

from typing import Tuple
import cv2
import numpy as np
from src.descriptors.histogram import compute_histogram
from src.data.extract import read_image


def convert_img_to_ycbcr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to YCbCr.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: YCbCr image.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)


def compute_ycbcr_histogram(
    img_path: str,
    values_per_bin: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute YCbCr concatenated histogram from image path.

    Notes:
        OpenCV uses the YCrCb convention internally.
        We reorder to Y, Cb, Cr when concatenating.

    Args:
        img_path: Path to the image file.
        values_per_bin: Number of intensity values per bin.

    Returns:
        concat_hist: Normalized concatenated histogram [Y | Cb | Cr].
        bin_edges : Bin edges for the histograms (all channels share the same edges).
    """
    img = read_image(img_path)
    img_ycrcb = convert_img_to_ycbcr(img)      # channels: Y, Cr, Cb

    y = img_ycrcb[:, :, 0]
    cr = img_ycrcb[:, :, 1]
    cb = img_ycrcb[:, :, 2]

    # Compute per-channel histograms (8-bit ranges; Cb/Cr centered ~128 but still 0–255)
    hist_y,  bin_edges = compute_histogram(
        y,  values_per_bin=values_per_bin, density=True)
    hist_cb, _ = compute_histogram(
        cb, values_per_bin=values_per_bin, density=True)
    hist_cr, _ = compute_histogram(
        cr, values_per_bin=values_per_bin, density=True)

    # Concatenate in YCbCr order (not the YCrCb that OpenCV returns)
    concat_hist = np.concatenate([hist_y, hist_cb, hist_cr])

    return concat_hist, bin_edges
