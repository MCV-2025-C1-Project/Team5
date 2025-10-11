"""
2D histogram descriptor (YCbCr-based).
"""

from typing import Tuple
import numpy as np

from src.data.extract import read_image
from src.descriptors.histogram import compute_histogram
from src.descriptors.ycbcr import convert_img_to_ycbcr
from src.descriptors.lab import convert_img_to_lab
from src.descriptors.hsv import convert_img_to_hsv


def compute_2d_histogram_from_array(
    img_bgr: np.ndarray,
    color_space: str = "ycbcr",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute luminance and chrominance histograms from an image in YCbCr space.

    This function builds:
      1) a 1D histogram for the Y (luminance) channel, and
      2) a 2D histogram for the chrominance (Cb, Cr) channels
    (note OpenCV loads YCrCb, so we reorder to CbCr).

    Args:
        img_bgr: (np.ndarray): Input image in BGR format.
        values_per_bin (int, optional): Number of consecutive intensity values
            grouped into each bin (e.g., 1→256 bins, 2→128 bins). Defaults to 1.
        **kwargs: Extra options forwarded to :func:`compute_histogram`
            (e.g., ``density=True``).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - hist_y: 1D histogram of the Y (luminance) channel.
            - bin_edges_y: Bin edges for the Y histogram.
            - hist_cbcr: 2D histogram of the (Cb, Cr) channels (Cb on axis-0, Cr on axis-1).
            - bin_edges_cbcr: Two arrays with the bin edges for Cb and Cr.

    Raises:
        ValueError: Propagated from :func:`compute_histogram` if invalid
            ``values_per_bin`` or arguments are provided.
    """
    if color_space.lower() == "ycbcr":
        img_ycbcr = convert_img_to_ycbcr(img_bgr)
        a = img_ycbcr[:, :, 0]  # Y
        b = img_ycbcr[:, :, 2]  # Cb
        c = img_ycbcr[:, :, 1]  # Cr
    elif color_space.lower() == "lab":
        img_lab = convert_img_to_lab(img_bgr)
        a = img_lab[:, :, 0]  # L
        b = img_lab[:, :, 1]  # A
        c = img_lab[:, :, 2]  # B
    elif color_space.lower() == "hsv":
        img_hsv = convert_img_to_hsv(img_bgr)
        a = img_hsv[:, :, 2]  # V
        b = img_hsv[:, :, 0]  # H
        c = img_hsv[:, :, 1]  # S
    else:
        raise ValueError(f"Unsupported color space '{color_space}'. "
                         "Expected 'ycbcr', 'lab', or 'hsv'.")

    bc = np.stack((b, c), axis=-1).reshape(-1, 2)

    hist_a, bin_edges_a = compute_histogram(
        a, dimension=1, density=False, **kwargs
    )
    hist_bc, bin_edges_bc = compute_histogram(
        bc, dimension=2, density=False, **kwargs
    )

    return hist_a, bin_edges_a, hist_bc, bin_edges_bc


def compute_2d_histogram(
    img_path: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a single feature vector by concatenating Y and CbCr histograms.

    Builds the Y (1D) and CbCr (2D) histograms in and returns a
    flattened feature vector: ``[hist_a, hist_ab.flatten()]``.

    Args:
        img_path (str): Path to the input image file.
        values_per_bin (int, optional): Number of consecutive intensity values
            grouped into each bin. Defaults to 1.
        **kwargs: Extra options forwarded to :func:`compute_histogram`
            (e.g., ``density=True``).

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]:
            - hist: 1D feature vector formed by concatenating the luminance
              histogram and the flattened chrominance histogram.
            - bin_edges: An empty array (no per-dimension edges are returned
              in this wrapper). If you need bin edges, call
              :func:`compute_2d_histogram_from_array` instead.

    Notes:
        - This wrapper intentionally omits bin edges for simplicity. If your
          downstream pipeline needs them (e.g., for visualization), use the
          more detailed function above.
    """
    img_bgr = read_image(img_path)

    hist_y, bin_edges_y, hist_cbcr, bin_edges_cbcr = compute_2d_histogram_from_array(
        img_bgr, **kwargs
    )
    hist = np.concatenate((hist_y.ravel(), hist_cbcr.ravel()))
    hist = hist / np.linalg.norm(hist)
    return hist, (bin_edges_y, bin_edges_cbcr)


# ---------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------


def compute_2d_histogram_ycbcr(
    img_path: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 2d YCbCr color histogram from an image.

    Wrapper around :func:`compute_2d_histogram` with ``color_space="ycbcr"``.

    Args:
        img_path (str): Path to the input image file.
        **kwargs: Additional keyword arguments passed to
            :func:`compute_2d_histogram`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - hist: 2D histogram of YCbCr color frequencies.
            - bin_edges: Bin edges for each YCbCr channel.
    """
    return compute_2d_histogram(img_path, color_space="ycbcr", **kwargs)


def compute_2d_histogram_lab(
    img_path: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 2D Lab color histogram from an image.

    Wrapper around :func:`compute_2d_histogram` with ``color_space="lab"``.

    Args:
        img_path (str): Path to the input image file.
        **kwargs: Additional keyword arguments passed to
            :func:`compute_2d_histogram`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - hist: 2D histogram of Lab color frequencies.
            - bin_edges: Bin edges for each Lab channel.
    """
    return compute_2d_histogram(img_path, color_space="lab", **kwargs)


def compute_2d_histogram_hsv(
    img_path: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 2d HSV color histogram from an image.

    Wrapper around :func:`compute_2d_histogram` with ``color_space="hsv"``.

    Args:
        img_path (str): Path to the input image file.
        **kwargs: Additional keyword arguments passed to
            :func:`compute_2d_histogram`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - hist: 2d histogram of HSV color frequencies.
            - bin_edges: Bin edges for each HSV channel.
    """
    return compute_2d_histogram(img_path, color_space="hsv", **kwargs)

def compute_2d_histogram_ycbcr_from_array(
    img_bgr: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 2D YCbCr color histogram from an image array.

    Wrapper around :func:`compute_2d_histogram_from_array` with ``color_space="ycbcr"``.

    Args:
        img_bgr (np.ndarray): Input image array in BGR format.
        **kwargs: Additional keyword arguments passed to
            :func:`compute_2d_histogram_from_array`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - hist: 2D histogram of YCbCr color frequencies.
            - bin_edges: Bin edges for each YCbCr channel.
    """
    return compute_2d_histogram_from_array(img_bgr, color_space="ycbcr", **kwargs)


def compute_2d_histogram_lab_from_array(
    img_bgr: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 2D Lab color histogram from an image array.

    Wrapper around :func:`compute_2d_histogram_from_array` with ``color_space="lab"``.

    Args:
        img_bgr (np.ndarray): Input image array in BGR format.
        **kwargs: Additional keyword arguments passed to
            :func:`compute_2d_histogram_from_array`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - hist: 2D histogram of Lab color frequencies.
            - bin_edges: Bin edges for each Lab channel.
    """
    return compute_2d_histogram_from_array(img_bgr, color_space="lab", **kwargs)


def compute_2d_histogram_hsv_from_array(
    img_bgr: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 2D HSV color histogram from an image array.

    Wrapper around :func:`compute_2d_histogram_from_array` with ``color_space="hsv"``.

    Args:
        img_bgr (np.ndarray): Input image array in BGR format.
        **kwargs: Additional keyword arguments passed to
            :func:`compute_2d_histogram_from_array`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - hist: 2D histogram of HSV color frequencies.
            - bin_edges: Bin edges for each HSV channel.
    """
    return compute_2d_histogram_from_array(img_bgr, color_space="hsv", **kwargs)

