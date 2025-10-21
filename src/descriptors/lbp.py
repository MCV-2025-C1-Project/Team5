from typing import Tuple, List
import numpy as np
from skimage.feature import local_binary_pattern
import cv2

from src.data.extract import read_image
from src.descriptors.grayscale import convert_img_to_gray_scale
from src.descriptors.lab import convert_img_to_lab
from src.descriptors.ycbcr import convert_img_to_ycbcr
from src.descriptors.hsv import convert_img_to_hsv

def compute_multiscale_lbp_for_block(
    block: np.ndarray,
    scales: List[Tuple[float, int]],
    method: str = 'uniform'
) -> np.ndarray:
    """Compute multiscale Local Binary Pattern histogram for a single block.

    This function extracts LBP features at multiple scales by varying the radius
    and number of sampling points. For each scale, it computes an LBP map and
    generates a normalized histogram. The histograms from all scales are
    concatenated to form a multiscale descriptor.

    Args:
        block (np.ndarray): 2D array representing a single image block/region.
        scales (List[Tuple[float, int]]): List of (radius, n_points) tuples
            defining the scales for LBP computation. For example, [(1, 8), (3, 24)]
            means LBP with radius=1 and 8 neighbors, plus radius=3 and 24 neighbors.
        method (str, optional): LBP method type. 'uniform' extracts only uniform
            patterns, reducing dimensionality. Defaults to 'uniform'.

    Returns:
        np.ndarray: 1D concatenated normalized histogram from all scales.
    """
    all_scale_histograms = []
    for radius, n_points in scales:
        # Compute LBP map for the current scale
        lbp_map = local_binary_pattern(block, P=n_points, R=radius, method=method)

        # Number of bins for 'uniform' method is n_points + 2
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp_map.ravel(), bins=n_bins, range=(0, n_bins))

        # Normalize histogram to make it scale-invariant
        hist = hist.astype("float")
        eps = 1e-7
        hist /= (hist.sum() + eps)

        all_scale_histograms.append(hist)

    return np.concatenate(all_scale_histograms)

def compute_lbp_descriptor_from_array(
    img_bgr: np.ndarray,
    color_space: str = 'grayscale',
    grid_size: Tuple[int, int] = (4, 4),
    scales: List[Tuple[float, int]] = [(1, 8), (3, 24)]
) -> np.ndarray:
    """Compute multiscale grid-based LBP descriptor from an image array.

    Converts the image to the specified color space, divides each channel into
    a grid of blocks, computes multiscale LBP histograms for each block, and
    concatenates them into a single normalized descriptor. This provides spatial
    information through the grid structure and texture information at multiple
    scales through LBP.

    Args:
        img_bgr (np.ndarray): Input image array in BGR format.
        color_space (str, optional): Color space for descriptor computation.
            Options: 'grayscale', 'lab', 'ycbcr', 'hsv'. Defaults to 'grayscale'.
        grid_size (Tuple[int, int], optional): Grid dimensions (rows, cols) for
            dividing each channel into spatial blocks. Defaults to (4, 4).
        scales (List[Tuple[float, int]], optional): List of (radius, n_points)
            tuples for multiscale LBP computation. Each tuple defines a scale
            with its sampling radius and number of neighbors. Defaults to
            [(1, 8), (3, 24)] for two scales: fine and coarse texture patterns.

    Returns:
        np.ndarray: Normalized 1D descriptor vector concatenating all channels,
            blocks, and scales.

    Raises:
        ValueError: If the specified color space is not supported.
    """
    channels_to_process = []
    space = color_space.lower()

    # Convert image to the specified color space
    if space == 'grayscale':
        channels_to_process = [convert_img_to_gray_scale(img_bgr)]
    elif space == 'lab':
        channels_to_process = cv2.split(convert_img_to_lab(img_bgr))
    elif space == 'ycbcr':
        channels_to_process = cv2.split(convert_img_to_ycbcr(img_bgr))
    elif space == 'hsv':
        channels_to_process = cv2.split(convert_img_to_hsv(img_bgr))
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
        
    all_channel_descriptors = []

    # Process each channel independently
    for channel in channels_to_process:
        h, w = channel.shape
        grid_rows, grid_cols = grid_size
        
        # Calculate block dimensions
        block_h = h // grid_rows
        block_w = w // grid_cols
        
        all_block_descriptors = []
        
        # Iterate through grid cells
        for i in range(grid_rows):
            for j in range(grid_cols):
                # Calculate block boundaries (handle remainder pixels in last row/col)
                row_start = i * block_h
                row_end = (i + 1) * block_h if i < grid_rows - 1 else h
                col_start = j * block_w
                col_end = (j + 1) * block_w if j < grid_cols - 1 else w
                
                # Extract block from channel
                block = channel[row_start:row_end, col_start:col_end]
                
                # Compute multiscale LBP histogram for this block
                block_hist = compute_multiscale_lbp_for_block(block, scales)
                all_block_descriptors.append(block_hist)
            
        # Concatenate all blocks for this channel
        channel_descriptor = np.concatenate(all_block_descriptors)
        all_channel_descriptors.append(channel_descriptor)
    
    # Concatenate all channels into final descriptor
    full_descriptor = np.concatenate(all_channel_descriptors)

    # L2 normalization for robustness
    norm = np.linalg.norm(full_descriptor)
    if norm > 0:
        full_descriptor = full_descriptor / norm
        
    return full_descriptor


def compute_lbp_descriptor(
    img_path: str, **kwargs
) -> np.ndarray:
    """Compute multiscale grid-based LBP descriptor from an image path.

    High-level wrapper that reads an image from disk and computes its LBP
    descriptor using the specified parameters. This is the main entry point
    for LBP feature extraction.

    Args:
        img_path (str): Path to the input image file.
        **kwargs: Additional keyword arguments passed to
            compute_lbp_descriptor_from_array (e.g., color_space, grid_size,
            scales).

    Returns:
        np.ndarray: Normalized 1D multiscale LBP descriptor vector.
    """
    img = read_image(img_path)
    return compute_lbp_descriptor_from_array(img, **kwargs)
