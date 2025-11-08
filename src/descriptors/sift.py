
from typing import Tuple, Callable, Optional

import cv2
import numpy as np

from src.data.extract import read_image
from src.descriptors.grayscale import convert_img_to_gray_scale

SIFT_DESCRIPTOR_DIM = 128


def compute_sift_descriptor_from_array(
    img_bgr: np.ndarray,
    keypoint_detector: Optional[Callable] = None,
    nfeatures: int = 5000,
    nOctaveLayers: int = 3,
    contrastThreshold: float = 0.04,
    edgeThreshold: float = 10.0,
    sigma: float = 1.6,
    aggregation: str = 'none',
    **keypoint_params
) -> Tuple[list, np.ndarray]:
    """
    Compute SIFT descriptors from image array.

    Args:
        img_bgr: Input image array in BGR format.
        keypoint_detector: Custom keypoint detection function (e.g., DoG detector).
            If None, uses SIFT's native DoG detector.
        nfeatures: Number of best features to retain (0 = no limit).
        nOctaveLayers: Number of layers in each octave.
        contrastThreshold: Threshold to filter weak features.
        edgeThreshold: Threshold to filter edge-like features.
        sigma: Sigma of Gaussian applied to input image.
        aggregation: Aggregation method ('mean' or 'none').
        **keypoint_params: Parameters passed to custom keypoint detector.

    Returns:
        keypoints: List of detected keypoints.
        descriptor: Aggregated descriptor (1D) or all descriptors (2D).
    """
    img_gray = convert_img_to_gray_scale(img_bgr)

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma
    )

    if keypoint_detector is not None:
        keypoints = keypoint_detector(img_bgr, **keypoint_params)
    else:
        keypoints = sift.detect(img_gray, None)

    if len(keypoints) > nfeatures:
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[
            :nfeatures]

    if not keypoints:
        if aggregation == 'mean':
            return [], np.zeros(SIFT_DESCRIPTOR_DIM, dtype=np.float32)
        return [], np.array([])

    keypoints, descriptors = sift.compute(img_gray, keypoints)

    if descriptors is None:
        if aggregation == 'mean':
            return [], np.zeros(SIFT_DESCRIPTOR_DIM, dtype=np.float32)
        return [], np.array([])

    if aggregation == 'mean':
        aggregated = np.mean(descriptors, axis=0)
        return keypoints, aggregated.astype(np.float32)
    elif aggregation == 'none':
        return keypoints, descriptors
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def compute_sift_descriptor(
    img_path: str,
    keypoint_detector: Optional[Callable] = None,
    **kwargs
) -> Tuple[list, np.ndarray]:
    """
    Compute SIFT descriptors from image file path.

    Args:
        img_path: Path to input image file.
        keypoint_detector: Custom keypoint detection function.
            If None, uses SIFT's native detector.
        **kwargs: Arguments passed to compute_sift_descriptor_from_array.

    Returns:
        keypoints: List of detected keypoints.
        descriptor: Aggregated descriptor (1D) or all descriptors (2D).
    """
    img_bgr = read_image(img_path)
    return compute_sift_descriptor_from_array(
        img_bgr,
        keypoint_detector=keypoint_detector,
        **kwargs
    )
