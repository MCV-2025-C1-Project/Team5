import numpy as np
import pywt
import cv2
from typing import Tuple
from src.data.extract import read_image
from src.descriptors.grayscale import convert_img_to_gray_scale
from src.descriptors.hsv import convert_img_to_hsv

# ==================================================================
#                             DWT
# ==================================================================

def compute_dwt_statistics(channel_array: np.ndarray, wavelet: str, levels: int) -> np.ndarray:
    """
    Compute multiple statistical features from Discrete Wavelet Transform (DWT) subbands.

    For each subband (approximation + details), several statistics are extracted:
    - Mean (average energy)
    - Variance (dispersion)
    - Standard deviation (amplitude of variation)

    Args:
        channel_array (np.ndarray): 2D single-channel image (grayscale or one color channel).
        wavelet (str): Wavelet type (e.g., 'haar', 'bior4.4', 'db2', 'sym4').
        levels (int): Number of decomposition levels for the DWT.

    Returns:
        np.ndarray: 1D feature vector with [mean, var, std] per subband.
    """
    # Normalize the input channel to [0,1]
    channel = np.float32(channel_array) / 255.0

    # Perform multi-level 2D DWT decomposition
    coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=levels)

    # Initialize feature list
    features = []

    # Iterate through approximation and detail coefficients
    for i, c in enumerate(coeffs):
        # coeffs[0] = approximation (LL)
        # coeffs[1:] = tuples of detail coefficients (LH, HL, HH)
        subbands = [c] if i == 0 else list(c)

        for sub in subbands:
            vals = sub.flatten()
            abs_vals = np.abs(vals)

            # Some subbands (especially at higher levels) can be constant or zero.
            # To avoid NaNs or runtime warnings, we set all features to zero for that subband.
            if np.all(abs_vals == 0):
                features.extend([0, 0, 0])
                continue

            # Compute basic statistical measures
            mean_val = np.mean(abs_vals)
            var_val = np.var(abs_vals)
            std_val = np.std(abs_vals)

            features.extend([mean_val, var_val, std_val])

    return np.array(features, dtype=np.float32)


def compute_dwt_descriptor_from_array(
    img_bgr: np.ndarray,
    color_space: str = 'grayscale',
    wavelet: str = 'haar',
    levels: int = 2
) -> np.ndarray:
    """Compute a global Wavelet-based texture descriptor from an image array.

    Converts image to specified color space, applies a multi-level 2D DWT to
    each channel, extracts average absolute energies from all subbands,
    concatenates and normalizes them.

    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        color_space (str, optional): Target color space. Options:
            'grayscale', 'lab', 'ycbcr', 'hsv'. Defaults to 'grayscale'.
        wavelet (str, optional): Wavelet type. Defaults to 'haar'.
        levels (int, optional): Number of DWT decomposition levels. Defaults to 2.

    Returns:
        np.ndarray: Normalized 1D descriptor concatenating all channel energies.
    """
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

    all_features = []
    for ch in channels_to_process:
        feats = compute_dwt_statistics(ch, wavelet, levels)
        all_features.append(feats)

    descriptor = np.concatenate(all_features)
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm

    return descriptor


def compute_dwt_descriptor(img_path: str, **kwargs) -> np.ndarray:
    """Compute a global Wavelet-based descriptor from an image file.

    Wrapper that reads an image from disk and applies the wavelet descriptor.

    Args:
        img_path (str): Path to the image.
        **kwargs: Passed to compute_wavelet_descriptor_from_array
            (e.g., color_space, wavelet, levels).

    Returns:
        np.ndarray: Normalized 1D descriptor vector.
    """
    img = read_image(img_path)
    return compute_dwt_descriptor_from_array(img, **kwargs)

# ==================================================================
#                         Block-based DWT
# ==================================================================

def compute_block_dwt_for_channel(
    channel_array: np.ndarray,
    block_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    wavelet: str,
    levels: int
) -> np.ndarray:
    """Compute block-based DWT descriptor for a single channel using compute_dwt_statistics.

    Each block is resized to a fixed size, then passed to compute_dwt_statistics
    to extract (mean, var, std) features from all DWT subbands.

    Args:
        channel_array (np.ndarray): 2D single-channel image.
        block_size (Tuple[int, int]): Target (height, width) for each block.
        grid_size (Tuple[int, int]): (rows, cols) defining the spatial grid.
        wavelet (str): Wavelet type (e.g., 'haar', 'bior4.4', 'db2').
        levels (int): Number of decomposition levels.

    Returns:
        np.ndarray: Concatenated descriptor from all blocks.
    """
    h, w = channel_array.shape
    grid_rows, grid_cols = grid_size
    block_h = h // grid_rows
    block_w = w // grid_cols

    all_block_features = []

    for i in range(grid_rows):
        for j in range(grid_cols):
            # Define block boundaries
            row_start = i * block_h
            row_end = h if i == grid_rows - 1 else (i + 1) * block_h
            col_start = j * block_w
            col_end = w if j == grid_cols - 1 else (j + 1) * block_w

            block = channel_array[row_start:row_end, col_start:col_end]
            block_resized = cv2.resize(block, block_size)
            block_float = np.float32(block_resized)

            block_features = compute_dwt_statistics(block_float, wavelet=wavelet, levels=levels)
            all_block_features.append(block_features)

    return np.concatenate(all_block_features).astype(np.float32)

def compute_block_dwt_descriptor_from_array(
    img_bgr: np.ndarray,
    color_space: str = 'grayscale',
    block_size: Tuple[int, int] = (64, 64),
    grid_size: Tuple[int, int] = (4, 4),
    wavelet: str = 'haar',
    levels: int = 2
) -> np.ndarray:
    """Compute block-based DWT descriptor from an image array.

    Converts the image to the specified color space, divides it into a grid of
    blocks, applies multi-level DWT on each block, extracts (mean, var, std)
    statistics from all subbands, concatenates and normalizes the result.

    Args:
        img_bgr (np.ndarray): Input image array in BGR format.
        color_space (str, optional): Color space for descriptor computation.
            Options: 'grayscale', 'lab', 'ycbcr', 'hsv'. Defaults to 'grayscale'.
        block_size (Tuple[int, int], optional): Block size (height, width) for resizing before DWT.
            Defaults to (64, 64).
        grid_size (Tuple[int, int], optional): Grid dimensions (rows, cols). Defaults to (4, 4).
        wavelet (str, optional): Wavelet type. Defaults to 'haar'.
        levels (int, optional): DWT decomposition levels. Defaults to 2.

    Returns:
        np.ndarray: Normalized 1D descriptor concatenating all channels and blocks.
    """
    # Convert to selected color space
    if color_space == 'grayscale':
        channels_to_process = [convert_img_to_gray_scale(img_bgr)]
    elif color_space == 'hsv':
        channels_to_process = cv2.split(convert_img_to_hsv(img_bgr))
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

    all_descriptors = []
    for channel in channels_to_process:
        channel_descriptor = compute_block_dwt_for_channel(
            channel,
            block_size=block_size,
            grid_size=grid_size,
            wavelet=wavelet,
            levels=levels
        )
        all_descriptors.append(channel_descriptor)

    full_descriptor = np.concatenate(all_descriptors)

    norm = np.linalg.norm(full_descriptor)
    if norm > 0:
        full_descriptor = full_descriptor / norm

    return full_descriptor


def compute_block_dwt_descriptor(img_path: str, **kwargs) -> np.ndarray:
    """Compute block-based DWT descriptor from an image path.

    Wrapper that reads an image from disk and applies the block-based DWT descriptor.

    Args:
        img_path (str): Path to the input image file.
        **kwargs: Additional keyword arguments passed to
            compute_block_dwt_descriptor_from_array (e.g., color_space, block_size,
            grid_size, wavelet, levels).

    Returns:
        np.ndarray: Normalized 1D DWT descriptor vector.
    """
    img = read_image(img_path)
    return compute_block_dwt_descriptor_from_array(img, **kwargs)
