import cv2
import numpy as np
from typing import Tuple

def convert_img_to_ycbcr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to YCbCr color space.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format (cv2.imread).

    Returns
    -------
    np.ndarray
        YCbCr image.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

def compute_histogram(img: np.ndarray,
                      bins: int = 256,
                      normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a 1-D histogram of the Cb and Cr channels in a YCbCr image.
    Similar to the Lab chroma histogram, this focuses on chrominance.

    Parameters
    ----------
    img : np.ndarray
        Input image already in YCbCr color space.
    bins : int, optional
        Number of histogram bins for each channel.
    normalize : bool, optional
        If True, divide by sum so the histogram sums to 1.

    Returns
    -------
    hist : np.ndarray
        Concatenated histogram of Cb and Cr channels.
    bin_edges : np.ndarray
        The edges of the histogram bins.
    """
    # Cb channel (index 1)
    cb = img[:, :, 1].flatten()

    # Cr channel (index 2)
    cr = img[:, :, 2].flatten()

    # Compute histograms for Cb and Cr
    hist_cb, bin_edges_cb = np.histogram(cb, bins=bins, range=(0, 256))
    hist_cr, bin_edges_cr = np.histogram(cr, bins=bins, range=(0, 256))

    # Concatenate histograms
    hist = np.concatenate([hist_cb, hist_cr])

    # Create concatenated bin edges (Cb bins + Cr bins offset)
    # We'll use simple sequential bins for visualization
    concat_bins = np.arange(len(hist) + 1).astype(np.float32)

    # Normalize if requested
    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()

    return hist.astype(np.float32), concat_bins
