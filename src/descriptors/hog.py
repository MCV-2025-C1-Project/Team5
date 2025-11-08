import cv2
import numpy as np
from typing import Tuple, Callable, Optional

from src.data.extract import read_image
from src.descriptors.grayscale import convert_img_to_gray_scale

def compute_hog_descriptor_from_array(
    img_bgr: np.ndarray,
    keypoint_detector: Optional[Callable] = None,
    win_size: Tuple[int, int] = (64, 128),
    block_size: Tuple[int, int] = (16, 16),
    block_stride: Tuple[int, int] = (8, 8),
    cell_size: Tuple[int, int] = (8, 8),
    nbins: int = 9,
    deriv_aperture: int = 1,
    win_sigma: float = -1.0,
    histogram_norm_type: int = 0,
    l2_hys_threshold: float = 0.2,
    gamma_correction: bool = True,
    nlevels: int = 64,
    signed_gradients: bool = False,
    patch_size: int = 64,
    aggregation: str = 'none',
    nfeatures: Optional[int] = None,
    **keypoint_params
) -> Tuple[list, np.ndarray]:
    """Compute HOG (Histogram of Oriented Gradients) descriptors from an image array.

    Extracts HOG features either from the entire image or from patches around detected
    keypoints. HOG captures edge or gradient structure that is characteristic of local
    shape and appearance, following Dalal & Triggs (2005).

    Args:
        img_bgr: Input image array in BGR format (H, W, 3), dtype uint8 or float.
        keypoint_detector: Optional callable that returns a list of `cv2.KeyPoint`.
            If None, computes HOG over the entire image resized to `win_size`.
            Different detectors may return different types:
            - DoG: returns list/tuple of keypoints
            - Harris: returns tuple (keypoints, response_map)
            - Harris-Laplacian: returns list of keypoints
        win_size: Detection window size (width, height) in pixels. The image or
            patch is resized to this dimension before computing HOG.
        block_size: Block size (width, height) for normalization. Each block
            contains multiple cells and is independently normalized.
        block_stride: Step size (width, height) for sliding the blocks across
            the detection window. Smaller strides create more overlapping blocks.
        cell_size: Cell size (width, height). Each cell accumulates a local
            histogram of gradient orientations.
        nbins: Number of orientation bins in each cell histogram. Typically 9
            for unsigned gradients (0-180°) or 18 for signed (0-360°).
        deriv_aperture: Aperture size for the Sobel derivative operator (1, 3, 5, 7).
        win_sigma: Gaussian smoothing sigma for the detection window. Set to -1
            for automatic calculation based on block size.
        histogram_norm_type: Block normalization method (0=L2-Hys, 1=L2, 2=L1, 3=L1-sqrt).
        l2_hys_threshold: Clipping threshold for L2-Hys normalization (typically 0.2).
        gamma_correction: If True, apply gamma (power-law) correction to the input
            image to normalize illumination.
        nlevels: Number of detection window scale levels for multi-scale detection.
        signed_gradients: If True, use signed gradients (0-360°); otherwise unsigned (0-180°).
        patch_size: Size of square patches (pixels) extracted around each keypoint.
            Only used when `keypoint_detector` is provided.
        aggregation: Method to combine multiple descriptors when using keypoints:
            - "mean": Average all descriptors into a single vector.
            - "concat": Concatenate all descriptors into one long vector.
            - "none": Return all descriptors separately (2D array).
        nfeatures: Optional limit on the number of keypoints to process. If provided,
            selects the top `nfeatures` keypoints by response strength.
        **keypoint_params: Additional parameters forwarded to the keypoint detector.

    Returns:
        A tuple `(keypoints, descriptor)` where:
            keypoints: List of `cv2.KeyPoint` used for descriptor extraction.
                Empty list if using the full image approach.
            descriptor: HOG descriptor(s). Shape depends on `aggregation`:
                - "mean": 1D array of shape (D,)
                - "concat": 1D array of shape (N*D,)
                - "none": 2D array of shape (N, D)
                Returns zeros if no valid descriptors are computed.

    Raises:
        ValueError: If `aggregation` is not one of {"mean", "concat", "none"}.
    """
    # Convert to grayscale since HOG operates on intensity gradients
    img_gray = convert_img_to_gray_scale(img_bgr)
    
    # Create HOG descriptor object with specified parameters.
    # This configures the feature extraction pipeline but doesn't compute descriptors yet.
    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=nbins,
        _derivAperture=deriv_aperture,
        _winSigma=win_sigma,
        _histogramNormType=histogram_norm_type,
        _L2HysThreshold=l2_hys_threshold,
        _gammaCorrection=gamma_correction,
        _nlevels=nlevels,
        _signedGradient=signed_gradients
    )
    
    descriptors_list = []
    keypoints = []
    
    if keypoint_detector is not None:
        # === Keypoint-based approach ===
        # Detect interest points and compute HOG on patches around them.
        # This is useful for matching and recognition tasks.
        
        # Detect keypoints using the provided detector function.
        # Different keypoint detectors have different return signatures:
        # - DoG: returns list of keypoints
        # - Harris: returns tuple (keypoints, response_map)
        # - Harris-Laplacian: returns list of keypoints
        result = keypoint_detector(img_bgr, **keypoint_params)
        
        # Handle different return types from various keypoint detectors
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
            # This is the specific signature for the Harris detector
            detected_keypoints = result[0]
        else:
            # This handles DoG (tuple of keypoints) and Harris-Laplacian (list of keypoints)
            detected_keypoints = result
        
        if not detected_keypoints:
            # Return zeros if no keypoints detected
            descriptor_size = hog.getDescriptorSize()
            if aggregation == 'mean':
                return [], np.zeros(descriptor_size, dtype=np.float32)
            return [], np.array([])
        
        # Limit the number of keypoints if nfeatures is specified.
        # Selects the strongest keypoints by response value (quality/strength).
        if nfeatures is not None and len(detected_keypoints) > nfeatures:
            # Sort keypoints by response (strength) in descending order and take the top nfeatures
            detected_keypoints = sorted(detected_keypoints, key=lambda kp: kp.response, reverse=True)[:nfeatures]
        
        # Extract square patches around each keypoint and compute HOG for each.
        h, w = img_gray.shape
        half_patch = patch_size // 2
        
        for kp in detected_keypoints:
            # Get keypoint coordinates (rounded to nearest pixel)
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Check if patch is within image bounds to avoid boundary errors
            if (x - half_patch < 0 or x + half_patch >= w or
                y - half_patch < 0 or y + half_patch >= h):
                continue
            
            # Extract square patch centered at keypoint
            patch = img_gray[y - half_patch:y + half_patch,
                           x - half_patch:x + half_patch]
            
            # Resize patch to win_size if necessary for HOG computation
            if patch.shape != (win_size[1], win_size[0]):
                patch = cv2.resize(patch, win_size)
            
            try:
                # Compute HOG descriptor for this patch
                desc = hog.compute(patch)
                if desc is not None:
                    descriptors_list.append(desc.flatten())
                    keypoints.append(kp)
            except cv2.error:
                # Skip patches that cause errors (e.g., invalid size)
                continue
        
        if not descriptors_list:
            # Return zeros if no valid descriptors were computed
            descriptor_size = hog.getDescriptorSize()
            if aggregation == 'mean':
                return [], np.zeros(descriptor_size, dtype=np.float32)
            return [], np.array([])
        
        # Stack all descriptors into a 2D array (N, D)
        descriptors = np.array(descriptors_list, dtype=np.float32)
        
    else:
        # === Full image approach ===
        # Compute a single HOG descriptor for the entire image.
        # Useful for global image classification tasks.
        
        h, w = img_gray.shape
        
        # Resize image to match win_size if necessary.
        # HOG requires a specific window size for computation.
        if (h, w) != (win_size[1], win_size[0]):
            img_resized = cv2.resize(img_gray, win_size)
        else:
            img_resized = img_gray
        
        try:
            # Compute HOG descriptor for the entire image
            desc = hog.compute(img_resized)
            if desc is None:
                # Return zeros if computation failed
                descriptor_size = hog.getDescriptorSize()
                if aggregation == 'mean':
                    return [], np.zeros(descriptor_size, dtype=np.float32)
                return [], np.array([])
            # Reshape to (1, D) for consistency with keypoint-based approach
            descriptors = desc.flatten().reshape(1, -1)
        except cv2.error:
            # Return zeros if an error occurred during computation
            descriptor_size = hog.getDescriptorSize()
            if aggregation == 'mean':
                return [], np.zeros(descriptor_size, dtype=np.float32)
            return [], np.array([])
    
    # === Apply descriptor aggregation ===
    # Combine multiple descriptors into a single representation or keep them separate.
    if aggregation == 'mean':
        # Average all descriptors element-wise into a single descriptor.
        # Reduces dimensionality and provides a global representation.
        aggregated = np.mean(descriptors, axis=0)
        return keypoints, aggregated.astype(np.float32)
    elif aggregation == 'concat':
        # Concatenate all descriptors into one long vector.
        # Preserves all local information but increases dimensionality.
        concatenated = descriptors.flatten()
        return keypoints, concatenated.astype(np.float32)
    elif aggregation == 'none':
        # Return all descriptors separately without aggregation.
        # Useful for bag-of-words or when individual descriptors are needed.
        return keypoints, descriptors
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def compute_hog_descriptor(
    img_path: str,
    keypoint_detector: Optional[Callable] = None,
    **kwargs
) -> Tuple[list, np.ndarray]:
    """Compute HOG descriptors from an image file path.

    Convenience wrapper that reads an image from disk and forwards it to
    `compute_hog_descriptor_from_array`. All HOG and keypoint parameters
    are passed through via `**kwargs`.

    Args:
        img_path: Path to the input image file (absolute or relative).
        keypoint_detector: Optional callable for keypoint detection.
            If None, computes HOG over the entire image.
        **kwargs: Additional arguments forwarded to `compute_hog_descriptor_from_array`.
            Examples: win_size, block_size, nbins, aggregation, patch_size, etc.

    Returns:
        A tuple `(keypoints, descriptor)` where:
            keypoints: List of `cv2.KeyPoint` (empty if using full image).
            descriptor: HOG descriptor(s), shape depends on aggregation mode.
    """
    img_bgr = read_image(img_path)
    return compute_hog_descriptor_from_array(
        img_bgr,
        keypoint_detector=keypoint_detector,
        **kwargs
    )
