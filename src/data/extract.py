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
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
