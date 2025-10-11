import numpy as np
from typing import Tuple

from src.data.extract import read_image


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

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image array in BGR format.
    compute_histogram_func : callable
        A function that takes an image region (a block) and returns its
        histogram descriptor as a 1D numpy.ndarray.
    values_per_bin : int, optional
        Number of intensity values per bin. Defaults to 1.
    grid_size : tuple of (int, int), optional
        A tuple (rows, cols) defining the number of blocks to divide the
        image into. Defaults to (4, 4).
    **kwargs : dict, optional
        Additional keyword arguments to pass to compute_histogram_func.

    Returns
    -------
    numpy.ndarray
        A single 1D feature vector representing the concatenation of all
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

            hist, _ = compute_histogram_func(block, values_per_bin, **kwargs)
            block_histograms.append(hist)
    
    return np.concatenate(block_histograms)


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

    Parameters
    ----------
    img_path : str
        Path to the input image file.
    compute_histogram_func : callable
        A function that takes an image region (a block) and returns its
        histogram descriptor as a 1D numpy.ndarray.
    values_per_bin : int, optional
        Number of intensity values per bin. Defaults to 1.
    grid_size : tuple of (int, int), optional
        A tuple (rows, cols) defining the number of blocks to divide the
        image into. Defaults to (4, 4).
    **kwargs : dict, optional
        Additional keyword arguments to pass to compute_histogram_func.

    Returns
    -------
    numpy.ndarray
        A single 1D feature vector representing the concatenation of all
        block histograms, ordered from left-to-right, top-to-bottom.
    """
    img_bgr = read_image(img_path)
    
    return block_based_histogram_from_array(
        img_bgr, compute_histogram_func, values_per_bin, grid_size, **kwargs
    )
