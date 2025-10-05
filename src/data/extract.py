import cv2
import numpy as np


def read_image(image_filepath: str) -> np.ndarray:
    """Read an image from a file path as an RGB array.

    Args:
        image_filepath (str): Path to the image file.

    Returns:
        np.ndarray: Image array in RGB format if successful.
    """
    img_bgr = cv2.imread(image_filepath)
    if img_bgr is None:
        raise ValueError(f"Could not load image from {image_filepath}")

    return img_bgr 