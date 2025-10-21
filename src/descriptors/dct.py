from typing import Tuple
import cv2
import numpy as np
from src.data.extract import read_image
from src.descriptors.grayscale import convert_img_to_gray_scale
from src.descriptors.lab import convert_img_to_lab
from src.descriptors.ycbcr import convert_img_to_ycbcr
from src.descriptors.hsv import convert_img_to_hsv


def extract_zigzag_coefficients(dct_matrix: np.ndarray, n_coeffs: int) -> np.ndarray:
    """Extract DCT coefficients in zigzag order from a 2D matrix.

    This function traverses a 2D DCT matrix in zigzag order (similar to JPEG
    encoding) to extract the most significant low-frequency coefficients.
    The zigzag pattern prioritizes coefficients from top-left to bottom-right,
    capturing energy-concentrated components first.

    Args:
        dct_matrix (np.ndarray): 2D array containing DCT coefficients.
        n_coeffs (int): Number of coefficients to extract. If the matrix has
            fewer elements, the result is zero-padded.

    Returns:
        np.ndarray: 1D array of extracted coefficients in zigzag order.
    """
    h, w = dct_matrix.shape
    zigzag_indices = []

    for diagonal in range(h + w - 1):
        if diagonal % 2 == 0:
            # Even diagonals: scan moves up and to the right
            i = min(diagonal, h - 1)
            j = max(0, diagonal - h + 1)
            
            while i >= 0 and j < w:
                zigzag_indices.append((i, j))
                i -= 1
                j += 1
        else:
            # Odd diagonals: scan moves down and to the left
            j = min(diagonal, w - 1)
            i = max(0, diagonal - w + 1)

            while j >= 0 and i < h:
                zigzag_indices.append((i, j))
                i += 1
                j -= 1
    
    coeffs = []
    for idx, (i, j) in enumerate(zigzag_indices):
        if idx >= n_coeffs:
            break
        coeffs.append(dct_matrix[i, j])
    
    # Zero-pad if matrix is smaller than n_coeffs
    while len(coeffs) < n_coeffs:
        coeffs.append(0.0)
    
    return np.array(coeffs)

def compute_dct_for_channel(
    channel_array: np.ndarray,
    block_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    zigzag_coeffs: int
) -> np.ndarray:
    """Compute DCT descriptor for a single image channel using grid-based blocks.

    Divides the channel into a grid of blocks, resizes each block to the specified
    block size, applies DCT, and extracts zigzag coefficients from each block.

    Args:
        channel_array (np.ndarray): 2D array representing a single image channel.
        block_size (Tuple[int, int]): Size (height, width) to resize each block
            before applying DCT.
        grid_size (Tuple[int, int]): Grid dimensions (rows, cols) for dividing
            the channel into blocks.
        zigzag_coeffs (int): Number of zigzag coefficients to extract from each
            DCT block.

    Returns:
        np.ndarray: 1D concatenated descriptor of all block DCT coefficients.
    """
    h, w = channel_array.shape
    grid_rows, grid_cols = grid_size
    
    block_h = h // grid_rows
    block_w = w // grid_cols

    all_block_descriptors = []

    for i in range(grid_rows):
        for j in range(grid_cols):
            row_start = i * block_h
            row_end = (i + 1) * block_h if i < grid_rows - 1 else h
            
            col_start = j * block_w
            col_end = (j + 1) * block_w if j < grid_cols - 1 else w
            block = channel_array[row_start:row_end, col_start:col_end]
            block_resized = cv2.resize(block, block_size)
            
            block_float = np.float32(block_resized)
            dct_block = cv2.dct(block_float)
            
            descriptor = extract_zigzag_coefficients(dct_block, zigzag_coeffs)
            all_block_descriptors.append(descriptor)
            
    return np.concatenate(all_block_descriptors)


def compute_block_dct_descriptor_from_array(
    img_bgr: np.ndarray,
    color_space: str = 'grayscale',
    block_size: Tuple[int, int] = (8, 8),
    grid_size: Tuple[int, int] = (4, 4),
    zigzag_coeffs: int = 16
) -> np.ndarray:
    """Compute block-based DCT descriptor from an image array.

    Converts the image to the specified color space, divides it into a grid of
    blocks, computes DCT for each block, extracts zigzag coefficients, and
    concatenates them into a single normalized descriptor.

    Args:
        img_bgr (np.ndarray): Input image array in BGR format.
        color_space (str, optional): Color space for descriptor computation.
            Options: 'grayscale', 'lab', 'ycbcr', 'hsv'. Defaults to 'grayscale'.
        block_size (Tuple[int, int], optional): Size to resize each block before
            DCT. Defaults to (8, 8).
        grid_size (Tuple[int, int], optional): Grid dimensions (rows, cols) for
            dividing the image. Defaults to (4, 4).
        zigzag_coeffs (int, optional): Number of zigzag coefficients to extract
            per block. Defaults to 16.

    Returns:
        np.ndarray: Normalized 1D descriptor vector concatenating all channels
            and blocks.

    Raises:
        ValueError: If the specified color space is not supported.
    """
    channels_to_process = []

    if color_space == 'grayscale':
        channels_to_process = [convert_img_to_gray_scale(img_bgr)]
    elif color_space == 'lab':
        channels_to_process = cv2.split(convert_img_to_lab(img_bgr))
    elif color_space == 'ycbcr':
        channels_to_process = cv2.split(convert_img_to_ycbcr(img_bgr))
    elif color_space == 'hsv':
        channels_to_process = cv2.split(convert_img_to_hsv(img_bgr))
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    
    all_descriptors = []
    for channel in channels_to_process:
        channel_descriptor = compute_dct_for_channel(
            channel, block_size, grid_size, zigzag_coeffs
        )
        all_descriptors.append(channel_descriptor)

    full_descriptor = np.concatenate(all_descriptors)
    
    norm = np.linalg.norm(full_descriptor)
    if norm > 0:
        full_descriptor = full_descriptor / norm
        
    return full_descriptor

def compute_block_dct_descriptor(
    img_path: str, **kwargs
) -> np.ndarray:
    """Compute block-based DCT descriptor from an image path.

    High-level wrapper that reads an image from disk and computes its DCT
    descriptor using the specified parameters.

    Args:
        img_path (str): Path to the input image file.
        **kwargs: Additional keyword arguments passed to
            compute_block_dct_descriptor_from_array (e.g., color_space, block_size,
            grid_size, zigzag_coeffs).

    Returns:
        np.ndarray: Normalized 1D DCT descriptor vector.
    """
    img = read_image(img_path)
    return compute_block_dct_descriptor_from_array(img, **kwargs)
