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
    Compute HSV concatenated histogram from image path, scaling H to 0–255 so all channels
    share the same bin edges.

    Args:
        img_path: Path to the image file.
        values_per_bin: Number of intensity values per bin.

    Returns:
        hist_concat: Concatenated histogram [H | S | V] (length 3*N).
        bin_edges: Shared bin edges (length N+1).
    """
    img = read_image(img_path)              # BGR
    img_hsv = convert_img_to_hsv(img)       # HSV

    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

    hist_h, bin_edges = compute_histogram(h, values_per_bin=values_per_bin, density=True)
    hist_s, _         = compute_histogram(s,        values_per_bin=values_per_bin, density=True)
    hist_v, _         = compute_histogram(v,        values_per_bin=values_per_bin, density=True)

    hist_concat = np.concatenate([hist_h, hist_s, hist_v])

    return hist_concat, bin_edges

