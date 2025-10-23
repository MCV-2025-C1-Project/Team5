import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# METRICS

def compute_psne(img_gt, img):
    return peak_signal_noise_ratio(img_gt, img)


def compute_ssim(img_gt, img):
    if len(img.shape) > 2:
        return structural_similarity(img_gt, img, channel_axis=2)
    else:
        return structural_similarity(img_gt, img)


# Noise addition


def add_noise(img: np.ndarray, noise_type: str = "gaussian", amount: float = 0.02, mean: float = 0.0, var: float = 0.01) -> np.ndarray:
    """
    Add different types of noise to an image.

    Args:
        img (np.ndarray): Input image in range [0, 255].
        noise_type (str): Type of noise to add. Options: 'gaussian', 'salt_pepper', 'uniform'.
        amount (float): Noise intensity or fraction of pixels affected (used for salt_pepper and uniform).
        mean (float): Mean of Gaussian noise.
        var (float): Variance of Gaussian noise.

    Returns:
        np.ndarray: Noisy image in uint8 format.
    """
    img = img.astype(np.float32) / 255.0  # normalize to [0, 1]
    noisy = img.copy()

    if noise_type == "gaussian":
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, img.shape)
        noisy = img + gauss

    elif noise_type == "salt_pepper":
        noisy = img.copy()
        num_salt = np.ceil(amount * img.size * 0.5)
        num_pepper = np.ceil(amount * img.size * 0.5)

        # Salt (white) noise
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape[:2]]
        noisy[coords[0], coords[1]] = 1

        # Pepper (black) noise
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape[:2]]
        noisy[coords[0], coords[1]] = 0

    elif noise_type == "uniform":
        uniform_noise = np.random.uniform(-amount, amount, img.shape)
        noisy = img + uniform_noise

    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    # Clip values and rescale
    noisy = np.clip(noisy, 0, 1)
    noisy = (noisy * 255).astype(np.uint8)

    return noisy
