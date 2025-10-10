"""
Histogram computation utilities for image descriptors.
"""

import math
import numpy as np


def compute_histogram(
    img: np.ndarray,
    values_per_bin: int = 1,
    density: bool = True,
    dimension: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a histogram for an image or feature array with flexible 
    dimensionality.

    This function computes a histogram over one or more dimensions (channels)
    of an image or feature array using ``numpy.histogramdd``. It supports
    configurable bin granularity and normalization.

    Args:
        img (np.ndarray): Input array representing pixel or feature values.
            For ``dimension=1``, it can be a 2D grayscale image or a flat vector.
            For higher dimensions (e.g. 2 or 3), it should have shape
            ``(n_samples, dimension)``.
        values_per_bin (int, optional): Number of consecutive intensity values
            grouped into each bin. For example, 1 → 256 bins, 2 → 128 bins.
            Must be between 1 and 256. Defaults to 1.
        density (bool, optional): If True, normalize the histogram so that the
            total volume equals 1. If False, return raw counts. Defaults to True.
        dimension (int, optional): Dimensionality of the histogram.
            Use 1 for grayscale, 2 for chrominance (e.g. CbCr), or 3 for color
            histograms. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - **hist**: Histogram values (counts or normalized probabilities).
              For multidimensional histograms, this will be an N-D array.
            - **bin_edges**: Array (or list of arrays) specifying the edges of
              each bin along every dimension.

    Raises:
        ValueError: If ``values_per_bin`` is less than 1 or greater than 256.
    """
    if (values_per_bin < 1) or (values_per_bin > 256):
        raise ValueError("values_per_bin must be >= 1 and <=256")

    bins = math.ceil(256 / values_per_bin)
    max_range = bins * values_per_bin

    # if we are computing 1D histogram, flatten all pixels before computing
    if dimension == 1:
        img = img.flatten()

    hist, bin_edges = np.histogramdd(
        img, bins=bins, range=[[0, max_range]]*dimension, density=density
    )

    if dimension == 1:
        bin_edges = bin_edges[0]

    return hist, bin_edges
