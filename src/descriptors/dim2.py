"""
2D histogram descriptor.
"""

from typing import Tuple
import numpy as np

from src.data.extract import read_image
from src.descriptors.histogram import compute_histogram
from src.descriptors.ycbcr import convert_img_to_ycbcr


def compute_2d_histogram(
    img_path: str,
    values_per_bin: int = 1,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute luminance and chrominance 2D histograms from an image in YCbCr
        space.

    Args:
        img_path (str): Path to the input image file.
        values_per_bin (int, optional): Number of intensity values per bin.
            Smaller values yield finer granularity. Defaults to 1.
        **kwargs: Additional keyword arguments passed to `compute_histogram()`.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - hist_y: 1D histogram of the Y (luminance) channel.
            - bin_edges_y: Bin edges for the Y histogram.
            - hist_cbcr: 2D histogram of the chrominance (Cb, Cr) channels.
            - bin_edges_cbcr: Bin edges for the CbCr histogram.
    """
    img_bgr = read_image(img_path)
    img_ycbcr = convert_img_to_ycbcr(img_bgr)

    img_y = img_ycbcr[:, :, 0]
    cr = img_ycbcr[:, :, 1]
    cb = img_ycbcr[:, :, 2]

    # Combine in CbCr order (OpenCV returns YCrCb)
    img_cbcr = np.stack((cb, cr), axis=-1).reshape(-1, 2)

    hist_y, bin_edges_y = compute_histogram(
        img_y, values_per_bin=values_per_bin, dimension=1, **kwargs)
    hist_cbcr, bin_edges_cbcr = compute_histogram(
        img_cbcr, values_per_bin=values_per_bin, dimension=2, **kwargs)

    return hist_y, bin_edges_y, hist_cbcr, bin_edges_cbcr
