"""
3D histogram descriptor.
"""

from typing import Tuple
import numpy as np

from src.data.extract import read_image
from src.descriptors.histogram import compute_histogram
from src.descriptors.rgb import convert_img_to_rgb
from src.descriptors.lab import convert_img_to_lab
from src.descriptors.hsv import convert_img_to_hsv


def compute_3d_histogram_from_array(
    img_bgr: np.ndarray,
    values_per_bin: int = 1,
    color_space: str = "rgb",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 3D color histogram from an image.

    Args:
        img_bgr: (np.ndarray): Input image in BGR format.
        values_per_bin (int, optional): Number of intensity values per bin.
            Smaller values yield finer granularity. Defaults to 1.
        color_space (str, optional): Color space to use for the histogram.
            Options are "rgb", "lab", or "hsv". Defaults to "rgb".
        **kwargs: Additional keyword arguments passed to `compute_histogram()`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - hist: 3D histogram of color frequencies.
            - bin_edges: Bin edges for each dimension.
    """
    if color_space.lower() == "rgb":
        img = convert_img_to_rgb(img_bgr)
    elif color_space.lower() == "lab":
        img = convert_img_to_lab(img_bgr)
    elif color_space.lower() == "hsv":
        img = convert_img_to_hsv(img_bgr)
    else:
        raise ValueError(f"Unsupported color space '{color_space}'. "
                         "Expected 'rgb', 'lab', or 'hsv'.")

    img_flattened = img.reshape(-1, img.shape[2])

    hist, bin_edges = compute_histogram(
        img_flattened, values_per_bin=values_per_bin, dimension=3, **kwargs)

    return hist, bin_edges


def compute_3d_histogram(
    img_path: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:

    img_bgr = read_image(img_path)

    return compute_3d_histogram_from_array(img_bgr, **kwargs)


# ---------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------


def compute_3d_histogram_rgb(
    img_path: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 3D RGB color histogram from an image.

    Wrapper around :func:`compute_3d_histogram` with ``color_space="rgb"``.

    Args:
        img_path (str): Path to the input image file.
        **kwargs: Additional keyword arguments passed to
            :func:`compute_3d_histogram`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - hist: 3D histogram of RGB color frequencies.
            - bin_edges: Bin edges for each RGB channel.
    """
    return compute_3d_histogram(img_path, color_space="rgb", **kwargs)


def compute_3d_histogram_lab(
    img_path: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 3D Lab color histogram from an image.

    Wrapper around :func:`compute_3d_histogram` with ``color_space="lab"``.

    Args:
        img_path (str): Path to the input image file.
        **kwargs: Additional keyword arguments passed to
            :func:`compute_3d_histogram`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - hist: 3D histogram of Lab color frequencies.
            - bin_edges: Bin edges for each Lab channel.
    """
    return compute_3d_histogram(img_path, color_space="lab", **kwargs)


def compute_3d_histogram_hsv(
    img_path: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 3D HSV color histogram from an image.

    Wrapper around :func:`compute_3d_histogram` with ``color_space="hsv"``.

    Args:
        img_path (str): Path to the input image file.
        **kwargs: Additional keyword arguments passed to
            :func:`compute_3d_histogram`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - hist: 3D histogram of HSV color frequencies.
            - bin_edges: Bin edges for each HSV channel.
    """
    return compute_3d_histogram(img_path, color_space="hsv", **kwargs)
