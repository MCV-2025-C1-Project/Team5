import numpy as np

from src.descriptors.block_histogram import block_based_histogram_from_array
from src.data.extract import read_image
from src.descriptors import lab, hsv, dim2, dim3


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

    Args:
        img_bgr (np.ndarray): Input image array in BGR format.
        compute_histogram_func (callable): The histogram computation function to be passed down.
            levels (int, optional): The number of levels in the pyramid. `levels=3` will compute descriptors
            for 1x1, 2x2, and 4x4 grids. Defaults to 3.
        values_per_bin (int, optional): The number of intensity values per bin. Defaults to 1.
        **kwargs (dict, optional): Additional keyword arguments to pass to compute_histogram_func.

    Returns:
        numpy.ndarray: A single 1D feature vector for the entire spatial pyramid.
    """
    pyramid_descriptors = []

    for level in range(levels):
        grid_dim = 2**level
        grid_size = (grid_dim, grid_dim)

        level_descriptor, bins = block_based_histogram_from_array(
            img_bgr=img_bgr,
            compute_histogram_func=compute_histogram_func,
            values_per_bin=values_per_bin,
            grid_size=grid_size,
            **kwargs
        )
        pyramid_descriptors.append(level_descriptor)

    return np.concatenate(pyramid_descriptors), bins


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

    Args:
        img_path (str): Path to the input image file.
        compute_histogram_func (callable): The histogram computation function to be passed down.
        levels (int, optional): The number of levels in the pyramid. `levels=3` will compute descriptors
            for 1x1, 2x2, and 4x4 grids. Defaults to 3.
        values_per_bin (int, optional): The number of intensity values per bin. Defaults to 1.
        **kwargs (dict, optional): Additional keyword arguments to pass to compute_histogram_func.

    Returns:
        numpy.ndarray: A single 1D feature vector for the entire spatial pyramid.
    """
    img_bgr = read_image(img_path)

    return spatial_pyramid_histogram_from_array(
        img_bgr, compute_histogram_func, levels, values_per_bin, **kwargs
    )


def spatial_pyramid_histogram_lab(img_path: str, **kwargs):
    """
    Compute a spatial pyramid histogram descriptor using the Lab color space.

    This function applies the spatial pyramid approach with Lab histograms, 
    capturing both global and local color information. Each pyramid level 
    subdivides the image into finer grids and concatenates the block histograms.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the Lab histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated spatial pyramid histogram descriptor in the Lab color space.
            - Bin edges or structure returned by the histogram function.
    """
    return spatial_pyramid_histogram(
        img_path,
        compute_histogram_func=lab.compute_lab_histogram_from_array,
        **kwargs
    )

def spatial_pyramid_histogram_hsv(img_path: str, **kwargs):
    """
    Compute a spatial pyramid histogram descriptor using the HSV color space.

    This function applies the spatial pyramid approach with HSV histograms, 
    capturing both global and coarse spatial color information. 

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the HSV histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated spatial pyramid histogram descriptor in the HSV color space.
            - Bin edges or structure returned by the histogram function.
    """
    return spatial_pyramid_histogram(
        img_path,
        compute_histogram_func=hsv.compute_hsv_histogram_from_array,
        **kwargs
    )

def spatial_pyramid_histogram_2d_lab(img_path: str, **kwargs):
    """
    Compute a spatial pyramid histogram descriptor using the 2D Lab color space.

    This function applies the spatial pyramid approach using 2D histograms 
    based on the Lab channels (e.g., a* and b*). Each pyramid level subdivides 
    the image into smaller grids, computes 2D histograms for each region, 
    and concatenates them into a single descriptor vector.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 2D Lab histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 2D spatial pyramid histogram descriptor in the Lab color space.
            - Bin edges or structure returned by the histogram function.
    """
    return spatial_pyramid_histogram(
        img_path,
        compute_histogram_func=dim2.compute_2d_histogram_lab_from_array,
        **kwargs
    )

def spatial_pyramid_histogram_2d_hsv(img_path: str, **kwargs):
    """
    Compute a spatial pyramid 2D HSV histogram descriptor .

    This function applies the spatial pyramid approach using 2D HSV histograms,
    capturing both global and coarse spatial color information. 

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 2D HSV histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 2D HSV spatial pyramid histogram descriptor.
            - Bin edges or structure returned by the histogram function.
    """
    return spatial_pyramid_histogram(
        img_path,
        compute_histogram_func=dim2.compute_2d_histogram_hsv_from_array,
        **kwargs
    )

def spatial_pyramid_histogram_3d_lab(img_path: str, **kwargs):
    """
    Compute a spatial pyramid histogram descriptor using the 3D Lab color space.

    This function applies the spatial pyramid approach using 3D histograms 
    that consider all Lab channels jointly (L*, a*, b*). 
    Each pyramid level subdivides the image into smaller grids, 
    computes 3D histograms for each region, and concatenates them 
    into a single, high-dimensional descriptor.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 3D Lab histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 3D spatial pyramid histogram descriptor in the Lab color space.
            - Bin edges or structure returned by the histogram function.
    """
    return spatial_pyramid_histogram(
        img_path,
        compute_histogram_func=dim3.compute_3d_histogram_lab_from_array,
        **kwargs
    )

def spatial_pyramid_histogram_3d_hsv(img_path: str, **kwargs):
    """
    Compute a spatial pyramid 3D HSV histogram descriptor.

    This function applies the spatial pyramid approach using 3D HSV histograms,
    which consider all HSV channels jointly (H, S, V). 

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 3D HSV histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 3D HSV spatial pyramid histogram descriptor.
            - Bin edges or structure returned by the histogram function.
    """
    return spatial_pyramid_histogram(
        img_path,
        compute_histogram_func=dim3.compute_3d_histogram_hsv_from_array,
        **kwargs
    )

