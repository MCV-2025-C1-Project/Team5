
from scipy.fft import fft2, fftshift
import pywt
import numpy as np
from scipy.ndimage import laplace


from src.descriptors import ycbcr, grayscale


def noise_detector_laplace(
    img_bgr: np.ndarray,
    threshold: int = 45,
    convert_to_ycbcr: bool = True
) -> bool:
    """Detects image noise using the Laplacian operator on the Y (luma) channel.

    The function computes the Laplacian of the luminance component in YCbCr space
    and classifies the image as noisy if the 75th percentile of absolute Laplacian
    values exceeds a fixed threshold.

    Args:
        img_bgr (np.ndarray): Input BGR image (uint8).

    Returns:
        bool: True if the image is considered noisy, False otherwise.
    """
    if convert_to_ycbcr:
        # convert to YCbCr
        img_ycbcr = ycbcr.convert_img_to_ycbcr(img_bgr)
        # keep the y channel onyly
        img = img_ycbcr[:, :, 0]
    else:
        img = grayscale.convert_img_to_gray_scale(img)

    # convert to integers 32 bits to be able to represent the differences
    img = img.astype(np.int32)

    # compute laplacian of the image
    laplace_img = laplace(img)
    laplace_img = np.abs(laplace_img.ravel())

    if np.quantile(laplace_img, .75) < threshold:  # no noise
        return False
    else:  # noise
        return True


def noise_detector_grad(
    img_bgr: np.ndarray,
    threshold: float = 20.0,
    convert_to_ycbcr: bool = True
) -> bool:
    """Detects image noise by analyzing first- and second-order gradients.

    The detector computes horizontal and vertical gradients of the Y channel
    and measures the variability of consecutive gradient differences. Images
    with high gradient irregularity are labeled as noisy.

    Args:
        img_bgr (np.ndarray): Input BGR image (uint8).

    Returns:
        bool: True if the image is considered noisy, False otherwise.
    """
    if convert_to_ycbcr:
        # convert to YCbCr
        img_ycbcr = ycbcr.convert_img_to_ycbcr(img_bgr)
        # keep the y channel onyly
        img = img_ycbcr[:, :, 0]
    else:
        img = grayscale.convert_img_to_gray_scale(img_bgr)

    # convert to integers 32 bits to be able to represent the differences
    img = img.astype(np.int32)

    # compute gradient of the image
    grad_i_img_y = img[1:, :] - img[:-1, :]
    grad_j_img_y = img[:, 1:] - img[:, :-1]

    # difference among consecutive values of gradients
    diff_1_grad_i = grad_i_img_y[1:, :] - grad_i_img_y[:-1, :]
    diff_1_grad_j = grad_j_img_y[1:, :] - grad_j_img_y[:-1, :]

    diff_1_grad = np.abs(
        np.concat((diff_1_grad_i.ravel(), diff_1_grad_j.ravel())))

    if np.quantile(diff_1_grad, .75) < threshold:  # no noise
        return False
    else:  # noise
        return True


def estimate_noise_wavelet(
    img: np.ndarray,
    threshold: float = 5,
    convert_to_ycbcr: bool = True
) -> tuple[float, bool]:
    """Estimates noise level using a single-level wavelet transform.

    The diagonal detail coefficients of a Haar ('db1') wavelet are used to
    estimate the standard deviation of additive noise. The image is classified
    as noisy if the estimated σ exceeds the given threshold.

    Args:
        img (np.ndarray): Input grayscale or color image.
        threshold (float, optional): Noise-level threshold for classification.
            Defaults to 5.

    Returns:
        tuple[float, bool]:
            - Estimated noise standard deviation (float).
            - Boolean flag indicating if the image is noisy.
    """
    if convert_to_ycbcr:
        # convert to YCbCr
        img_ycbcr = ycbcr.convert_img_to_ycbcr(img)
        # keep the y channel onyly
        img = img_ycbcr[:, :, 0]
    else:
        img = grayscale.convert_img_to_gray_scale(img)

    coeffs = pywt.dwt2(img, 'db1')
    cA, (cH, cV, cD) = coeffs
    # Use diagonal detail coefficients
    sigma = np.median(np.abs(cD)) / 0.6745  # estimated noise std
    return sigma, sigma > threshold  # True = noisy


def estimate_noise_fft(
    img: np.ndarray,
    threshold: float = 3.5,
    convert_to_ycbcr: bool = True
) -> tuple[float, bool]:
    """Estimates noise by comparing high- and low-frequency spectral energy.

    Performs a 2D FFT of the grayscale image and computes the ratio of energy
    in high-frequency bands to that in low-frequency bands. A high ratio
    indicates potential noise presence.

    Args:
        img (np.ndarray): Input grayscale or color image.
        threshold (float, optional): Frequency energy ratio threshold.
            Defaults to 3.5.

    Returns:
        tuple[float, bool]:
            - Ratio of high- to low-frequency energy (float).
            - Boolean flag indicating if the image is noisy.
    """
    if convert_to_ycbcr:
        # convert to YCbCr
        img_ycbcr = ycbcr.convert_img_to_ycbcr(img)
        # keep the y channel onyly
        img = img_ycbcr[:, :, 0]
    else:
        img = grayscale.convert_img_to_gray_scale(img)

    f = fftshift(fft2(img))
    magnitude = np.abs(f)

    # Split spectrum in low vs high frequencies
    h, w = img.shape
    center = (h // 2, w // 2)
    r = min(center) // 4  # low-frequency radius
    y, x = np.ogrid[:h, :w]
    mask = (x - center[1])**2 + (y - center[0])**2 <= r**2

    low_energy = magnitude[mask].sum()
    high_energy = magnitude[~mask].sum()
    ratio = high_energy / (low_energy + 1e-8)
    return ratio, ratio > threshold  # True = noisy
