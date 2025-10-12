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


def spatial_pyramid_histogram_hsv_lvl2(img_path: str, **kwargs):
    """
    Compute a spatial pyramid histogram descriptor using the HSV color space (2 levels).

    This function applies the spatial pyramid approach with HSV histograms, 
    capturing both global and coarse spatial color information. 
    It computes histograms for 1x1 and 2x2 grids and concatenates them 
    into a single descriptor vector.

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
        levels=2,
        **kwargs
    )


def spatial_pyramid_histogram_hsv_lvl3(img_path: str, **kwargs):
    """
    Compute a spatial pyramid histogram descriptor using the HSV color space (3 levels).

    This function applies the spatial pyramid approach with HSV histograms, 
    capturing color and spatial information at multiple scales. 
    It computes histograms for 1x1, 2x2, and 4x4 grids and concatenates 
    them into a single descriptor vector.

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
        levels=3,
        **kwargs
    )


def spatial_pyramid_histogram_hsv_lvl4(img_path: str, **kwargs):
    """
    Compute a spatial pyramid histogram descriptor using the HSV color space (4 levels).

    This function applies the spatial pyramid approach with HSV histograms, 
    capturing very fine-grained spatial color information. 
    It computes histograms for 1x1, 2x2, 4x4, and 8x8 grids and concatenates 
    them into a single, high-dimensional descriptor vector.

    Note:
        Using 4 levels produces a large descriptor and may increase computation time.

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
        levels=4,
        **kwargs
    )

def spatial_pyramid_histogram_hsv_lvl5(img_path: str, **kwargs):
    """
    Compute a spatial pyramid histogram descriptor using the HSV color space (5 levels).

    This function applies the spatial pyramid approach with HSV histograms, 
    capturing very fine-grained spatial color information. 
    It computes histograms for 1x1, 2x2, 4x4, and 8x8 grids and concatenates 
    them into a single, high-dimensional descriptor vector.

    Note:
        Using 5 levels produces a large descriptor and may increase computation time.

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
        levels=5,
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


def spatial_pyramid_histogram_2d_hsv_lvl2(img_path: str, **kwargs):
    """
    Compute a spatial pyramid 2D HSV histogram descriptor with 2 levels (1x1 and 2x2).

    This function applies the spatial pyramid approach using 2D HSV histograms,
    capturing both global and coarse spatial color information. 
    Histograms from 1x1 and 2x2 grids are concatenated into a single descriptor vector.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 2D HSV histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 2D HSV spatial pyramid histogram descriptor (2 levels).
            - Bin edges or structure returned by the histogram function.
    """
    return spatial_pyramid_histogram(
        img_path,
        compute_histogram_func=dim2.compute_2d_histogram_hsv_from_array,
        levels=2,
        **kwargs
    )


def spatial_pyramid_histogram_2d_hsv_lvl3(img_path: str, **kwargs):
    """
    Compute a spatial pyramid 2D HSV histogram descriptor with 3 levels (1x1, 2x2, and 4x4).

    This function applies the spatial pyramid approach using 2D HSV histograms,
    capturing color and spatial information at multiple scales.
    Histograms from 1x1, 2x2, and 4x4 grids are concatenated into a single descriptor vector.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 2D HSV histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 2D HSV spatial pyramid histogram descriptor (3 levels).
            - Bin edges or structure returned by the histogram function.
    """
    return spatial_pyramid_histogram(
        img_path,
        compute_histogram_func=dim2.compute_2d_histogram_hsv_from_array,
        levels=3,
        **kwargs
    )


def spatial_pyramid_histogram_2d_hsv_lvl4(img_path: str, **kwargs):
    """
    Compute a spatial pyramid 2D HSV histogram descriptor with 4 levels (1x1, 2x2, 4x4, and 8x8).

    This function applies the spatial pyramid approach using 2D HSV histograms,
    capturing detailed spatial and color relationships. 
    Histograms from all levels are concatenated into a single descriptor vector.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 2D HSV histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 2D HSV spatial pyramid histogram descriptor (4 levels).
            - Bin edges or structure returned by the histogram function.
    """
    return spatial_pyramid_histogram(
        img_path,
        compute_histogram_func=dim2.compute_2d_histogram_hsv_from_array,
        levels=4,
        **kwargs
    )


def spatial_pyramid_histogram_2d_hsv_lvl5(img_path: str, **kwargs):
    """
    Compute a spatial pyramid 2D HSV histogram descriptor with 5 levels (1x1, 2x2, 4x4, 8x8, and 16x16).

    This function applies the spatial pyramid approach using 2D HSV histograms,
    capturing very fine-grained spatial and chromatic details.
    Histograms from all levels are concatenated into a single high-dimensional descriptor.

    Note:
        Using 5 levels produces a large feature vector and may increase computation time.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 2D HSV histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 2D HSV spatial pyramid histogram descriptor (5 levels).
            - Bin edges or structure returned by the histogram function.
    """
    return spatial_pyramid_histogram(
        img_path,
        compute_histogram_func=dim2.compute_2d_histogram_hsv_from_array,
        levels=5,
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
    Compute a spatial pyramid histogram descriptor using the 3D HSV color space.

    This function applies the spatial pyramid approach using 3D histograms 
    that consider all HSV channels jointly (H, S, V). 
    Each pyramid level subdivides the image into smaller grids, 
    computes 3D histograms for each region, and concatenates them 
    into a single, high-dimensional descriptor.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 3D HSV histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 3D spatial pyramid histogram descriptor in the HSV color space.
            - Bin edges or structure returned by the histogram function.
    """
    return spatial_pyramid_histogram(
        img_path,
        compute_histogram_func=dim3.compute_3d_histogram_hsv_from_array,
        **kwargs
    )
