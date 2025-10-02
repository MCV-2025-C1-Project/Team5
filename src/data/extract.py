import cv2
import numpy as np


def read_image(image_filepath: str) -> np.ndarray:
    """Read an image from a file path.

    Args:
        image_filepath (str): Path to the image file.

    Returns:
        np.ndarray: Image array in BGR format if successful.
    """
    img = cv2.imread(image_filepath)
    return img
