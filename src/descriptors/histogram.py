"""
Histogram computation utilities for image descriptors.
"""

import numpy as np
import math


def compute_histogram(
    img: np.ndarray,
    values_per_bin: int = 1,
    density: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the histogram of a single-channel image with adjustable bin size.

    Args:
        img: Input single-channel image (grayscale or one channel).
        values_per_bin: Number of consecutive intensity values grouped into each bin.
            For example, 1 creates 256 bins (0–255 individually), 2 creates 128 bins.
            Must be >= 1. Defaults to 1.
        density: If True, normalize the histogram so the total area equals 1.
            If False, return raw counts. Defaults to True.

    Returns:
        Tuple containing:
            - Histogram values (counts or normalized probabilities).
            - Bin edges corresponding to the histogram.

    Raises:
        ValueError: If values_per_bin is less than 1 or larger than 256.
    """
    if (values_per_bin < 1) or (values_per_bin > 256):
        raise ValueError("values_per_bin must be >= 1 and <=256")

    bins = math.ceil(256 / values_per_bin)
    max_range = bins * values_per_bin

    hist, bin_edges = np.histogram(
        img, bins=bins, range=[0, max_range], density=density
    )
    return hist, bin_edges