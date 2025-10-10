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


def compute_3d_histogram(
    img_path: str,
    values_per_bin: int = 1,
    color_space: str = "rgb",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 3D color histogram from an image.

    Args:
        img_path (str): Path to the input image file.
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
    img_bgr = read_image(img_path)

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
