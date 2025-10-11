import numpy as np
from typing import Tuple

from src.data.extract import read_image
from src.descriptors import lab, hsv, dim3, dim2


def block_based_histogram_from_array(
    img_bgr: np.ndarray,
    compute_histogram_func,
    values_per_bin: int = 1,
    grid_size: Tuple[int, int] = (4, 4),
    **kwargs
) -> np.ndarray:
    """
    Divide an image array into a grid and concatenate histograms from each block.

    This function implements the block-based histogram strategy to add spatial
    information to a feature descriptor.

    Args:
        img_bgr (np.ndarray): Input image array in BGR format.
        compute_histogram_func (callable): A function that takes an image region (a block) and returns its
            histogram descriptor as a 1D numpy.ndarray.
        values_per_bin (int, optional): Number of intensity values per bin. Defaults to 1.
        grid_size (tuple of (int, int), optional): A tuple (rows, cols) defining the number of blocks to divide the
            A tuple (rows, cols) defining the number of blocks to divide the
            image into. Defaults to (4, 4).
        **kwargs (dict, optional): Additional keyword arguments to pass to compute_histogram_func.

    Returns:
        numpy.ndarray: A single 1D feature vector representing the concatenation of all
            block histograms, ordered from left-to-right, top-to-bottom.
    """
    # Get the dimensions of the image
    h, w = img_bgr.shape[:2]
    grid_rows, grid_cols = grid_size

    # Calculate the approximate height and width of each block
    block_h = h // grid_rows
    block_w = w // grid_cols

    block_histograms = []

    for i in range(grid_rows):
        for j in range(grid_cols):
            # Extract the block
            row_start = i * block_h
            row_end = (i + 1) * block_h

            col_start = j * block_w
            col_end = (j + 1) * block_w

            # For the last block in the row/column, make sure it reaches
            # the end of the image, in case the dimensions are not divisible
            if i == grid_rows - 1:
                row_end = h
            if j == grid_cols - 1:
                col_end = w

            block = img_bgr[row_start:row_end, col_start:col_end]

            result = compute_histogram_func(
                block, values_per_bin=values_per_bin, **kwargs)
            
            # if the function returns 2 values (hist, bins)
            if len(result) == 2:
                hist, bins = result

            # if the function returns 4 values (hist_y, bins_y, hist_cbcr, bins_cbcr)
            elif len(result) == 4:
                hist_y, bins_y, hist_cbcr, bins_cbcr = result
                # combine luminance and chrominance histograms into a single feature vector
                hist = np.concatenate((hist_y.ravel(), hist_cbcr.ravel()))
                # keep both sets of bin edges for reference
                bins = (bins_y, bins_cbcr)
            block_histograms.append(hist)

    return np.concatenate(block_histograms), bins


def block_based_histogram(
    img_path: str,
    compute_histogram_func,
    values_per_bin: int = 1,
    grid_size: Tuple[int, int] = (4, 4),
    **kwargs
) -> np.ndarray:
    """
    Divide an image into a grid and concatenate histograms from each block.

    This function implements the block-based histogram strategy to add spatial
    information to a feature descriptor.

    Args:
        img_path (str): Path to the input image file.
        compute_histogram_func (callable): A function that takes an image region (a block) and returns its
            histogram descriptor as a 1D numpy.ndarray.
        values_per_bin (int, optional): Number of intensity values per bin. Defaults to 1.
        grid_size (tuple of (int, int), optional): A tuple (rows, cols) defining the number of blocks to divide the
            image into. Defaults to (4, 4).
        **kwargs (dict, optional): Additional keyword arguments to pass to compute_histogram_func.

    Returns:
        numpy.ndarray: A single 1D feature vector representing the concatenation of all
            spatial block histograms, ordered from left-to-right, top-to-bottom.
    """
    img_bgr = read_image(img_path)

    return block_based_histogram_from_array(
        img_bgr, compute_histogram_func, values_per_bin, grid_size, **kwargs
    )

def block_based_histogram_lab(img_path: str, **kwargs):
    """
    Compute block-based histograms using the Lab color space.

    This function divides the image into blocks and computes Lab histograms for each block,
    concatenating them into a single descriptor vector.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 1D histogram descriptor in the Lab color space.
            - Bin edges or structure returned by the histogram function.
    """
    return block_based_histogram(
        img_path,
        compute_histogram_func=lab.compute_lab_histogram_from_array,
        **kwargs
    )

def block_based_histogram_hsv(img_path: str, **kwargs):
    """
    Compute block-based histograms using the HSV color space.

    This function divides the image into blocks and computes HSV histograms for each block,
    concatenating them into a single descriptor vector.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 1D histogram descriptor in the HSV color space.
            - Bin edges or structure returned by the histogram function.
    """
    return block_based_histogram(
        img_path,
        compute_histogram_func=hsv.compute_hsv_histogram_from_array,
        **kwargs
    )

def block_based_histogram_2d_lab(img_path: str, **kwargs):
    """
    Compute block-based 2D histograms in the Lab color space.

    This function divides the image into spatial blocks and computes a 2D Lab histogram for each block,
    concatenating all block histograms into a single high-dimensional descriptor vector.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 2D histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 2D histogram descriptor in the Lab color space.
            - Bin edges or structure returned by the histogram function.
    """
    return block_based_histogram(
        img_path,
        compute_histogram_func=dim2.compute_2d_histogram_lab_from_array,
        **kwargs
    )

def block_based_histogram_2d_hsv(img_path: str, **kwargs):
    """
    Compute block-based 2D histograms in the HSV color space.

    This function divides the image into spatial blocks and computes a 2D HSV histogram for each block,
    concatenating all block histograms into a single high-dimensional descriptor vector.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 2D histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 2D histogram descriptor in the HSV color space.
            - Bin edges or structure returned by the histogram function.
    """
    return block_based_histogram(
        img_path,
        compute_histogram_func=dim2.compute_2d_histogram_hsv_from_array,
        **kwargs
    )

def block_based_histogram_3d_lab(img_path: str, **kwargs):
    """
    Compute block-based 3D histograms in the Lab color space.

    This function divides the image into spatial blocks and computes a 3D Lab histogram for each block,
    concatenating all block histograms into a single high-dimensional descriptor vector.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 3D histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 3D histogram descriptor in the Lab color space.
            - Bin edges or structure returned by the histogram function.
    """
    return block_based_histogram(
        img_path,
        compute_histogram_func=dim3.compute_3d_histogram_lab_from_array,
        **kwargs
    )


def block_based_histogram_3d_hsv(img_path: str, **kwargs):
    """
    Compute block-based 3D histograms in the HSV color space.

    This function divides the image into spatial blocks and computes a 3D HSV histogram for each block,
    concatenating all block histograms into a single high-dimensional descriptor vector.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (dict, optional): Additional parameters passed to the 3D histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Concatenated 3D histogram descriptor in the HSV color space.
            - Bin edges or structure returned by the histogram function.
    """
    return block_based_histogram(
        img_path,
        compute_histogram_func=dim3.compute_3d_histogram_hsv_from_array,
        **kwargs
    )
