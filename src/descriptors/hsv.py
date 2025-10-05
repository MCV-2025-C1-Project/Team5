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

def compute_hsv_histogram(img_path: str, values_per_bin: int = 1) -> np.ndarray:
    """
    Compute HSV concatenated histogram from image path.
    
    Args:
        img_path (str): Path to the image file.
        values_per_bin (int): Number of intensity values per bin.
        
    Returns:
        np.ndarray: Normalized concatenated HSV histogram.
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    # Convert BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Compute histogram for each channel
    histograms = []
    for i in range(3):  # H, S, V
        channel = img_hsv[:, :, i]
        hist, _ = compute_histogram(channel, values_per_bin=values_per_bin, density=True)
        histograms.append(hist)
    
    # Concatenate histograms
    concat_hist = np.concatenate(histograms)
    
    return concat_hist