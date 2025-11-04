import cv2
import numpy as np

from src.data.extract import read_image
from src.descriptors.grayscale import convert_img_to_gray_scale


def detect_dog_keypoints_from_array(
        img_bgr: np.ndarray,
        num_scales: int = 5,
        sigma_base: float = 1.6,
        contrast_threshold: float = 0.03,
        edge_threshold: float = 10.0
) -> np.ndarray:
    """Detect keypoints using Difference of Gaussians (DoG) from an image array.

    Args:
        img_bgr (np.ndarray): Input image array in BGR format.
        num_scales (int, optional): Number of scales (octave layers) in the DoG
            pyramid. More scales capture finer details. Defaults to 5.
        sigma_base (float, optional): Initial Gaussian blur sigma for the base
            scale. Controls smoothing amount. Defaults to 1.6.
        contrast_threshold (float, optional): Threshold for filtering low-contrast
            keypoints. Higher values yield fewer, more stable keypoints. Defaults to 0.03.
        edge_threshold (float, optional): Threshold for eliminating edge-like
            responses. Higher values retain more edge features. Defaults to 10.0.

    Returns:
        np.ndarray: List of detected keypoints (cv2.KeyPoint objects). Each keypoint
            contains position (pt), scale (size), orientation (angle), response strength,
            and octave information.
    """
    img_gray = convert_img_to_gray_scale(img_bgr)

    # Create SIFT detector (uses DoG internally)
    sift = cv2.SIFT_create(
        nOctaveLayers = num_scales,
        contrastThreshold = contrast_threshold,
        edgeThreshold = edge_threshold,
        sigma= sigma_base
    )

    keypoints = sift.detect(img_gray, None)
    
    return keypoints

def detect_dog_keypoints(
        img_path: str,
        num_scales: int = 5,
        sigma_base: float = 1.6,
        contrast_threshold: float = 0.03,
        edge_threshold: float = 10.0
) -> np.ndarray:
    """Detect DoG keypoints from an image file path.

    Args:
        img_path (str): Path to the input image file.
        num_scales (int, optional): Number of DoG scales per octave. Defaults to 5.
        sigma_base (float, optional): Initial Gaussian sigma value. Defaults to 1.6.
        contrast_threshold (float, optional): Minimum contrast for keypoint
            acceptance. Defaults to 0.03.
        edge_threshold (float, optional): Threshold to filter edge-like responses.
            Defaults to 10.0.

    Returns:
        np.ndarray: List of detected keypoints with spatial and scale information.
    """
    img_bgr = read_image(img_path)

    return detect_dog_keypoints_from_array(
        img_bgr=img_bgr,
        num_scales=num_scales,
        sigma_base=sigma_base,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold
    )
