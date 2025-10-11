import numpy as np

from src.descriptors.block_histogram import block_based_histogram_from_array
from src.data.extract import read_image


def spatial_pyramid_histogram_from_array(
    img_bgr: np.ndarray,
    compute_histogram_func,
    levels: int = 3,
    values_per_bin: int = 1,
    **kwargs
) -> np.ndarray:
    """
    Compute a spatial pyramid descriptor from an image array by concatenating 
    histograms at multiple scales.

    This function calls the block_based_histogram_from_array function for different 
    grid sizes corresponding to pyramid levels and concatenates the results.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image array in BGR format.
    compute_histogram_func : callable
        The histogram computation function to be passed down.
    levels : int, optional
        The number of levels in the pyramid. `levels=3` will compute descriptors
        for 1x1, 2x2, and 4x4 grids. Defaults to 3.
    values_per_bin : int, optional
        The number of intensity values per bin. Defaults to 1.
    **kwargs : dict, optional
        Additional keyword arguments to pass to compute_histogram_func.

    Returns
    -------
    numpy.ndarray
        A single 1D feature vector for the entire spatial pyramid.
    """
    pyramid_descriptors = []

    for level in range(levels):
        grid_dim = 2**level
        grid_size = (grid_dim, grid_dim)

        level_descriptor = block_based_histogram_from_array(
            img_bgr=img_bgr,
            compute_histogram_func=compute_histogram_func,
            values_per_bin=values_per_bin,
            grid_size=grid_size,
            **kwargs
        )
        pyramid_descriptors.append(level_descriptor)
    
    return np.concatenate(pyramid_descriptors)


def spatial_pyramid_histogram(
    img_path: str,
    compute_histogram_func,
    levels: int = 3,
    values_per_bin: int = 1,
    **kwargs
) -> np.ndarray:
    """
    Compute a spatial pyramid descriptor by concatenating histograms at multiple scales.

    This function calls the block_based_histogram function for different grid
    sizes corresponding to pyramid levels and concatenates the results.

    Parameters
    ----------
    img_path : str
        Path to the input image file.
    compute_histogram_func : callable
        The histogram computation function to be passed down.
    levels : int, optional
        The number of levels in the pyramid. `levels=3` will compute descriptors
        for 1x1, 2x2, and 4x4 grids. Defaults to 3.
    values_per_bin : int, optional
        The number of intensity values per bin. Defaults to 1.
    **kwargs : dict, optional
        Additional keyword arguments to pass to compute_histogram_func.

    Returns
    -------
    numpy.ndarray
        A single 1D feature vector for the entire spatial pyramid.
    """
    img_bgr = read_image(img_path)
    
    return spatial_pyramid_histogram_from_array(
        img_bgr, compute_histogram_func, levels, values_per_bin, **kwargs
    )
