
import cv2
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, median_filter

from src.descriptors import ycbcr


def apply_gaussian_filter(img: np.ndarray) -> np.ndarray:
    """Apply a Gaussian filter to an image.

    Applies a Gaussian smoothing filter to reduce noise and detail. 
    If the input image has multiple channels, the filter is applied 
    along the first two spatial axes.

    Args:
        img (np.ndarray): Input grayscale or color image.

    Returns:
        np.ndarray: Denoised image after Gaussian filtering.
    """
    if len(img.shape) > 2:
        return gaussian_filter(img, sigma=1, radius=2, axes=(0, 1))
    else:
        return gaussian_filter(img, sigma=1, radius=2)


def apply_median_filter(img: np.ndarray) -> np.ndarray:
    """Apply a median filter to an image.

    Performs median filtering to remove impulsive (salt-and-pepper) noise.
    If the image has multiple channels, the filter is applied over the 
    first two spatial dimensions.

    Args:
        img (np.ndarray): Input grayscale or color image.

    Returns:
        np.ndarray: Denoised image after median filtering.
    """
    if len(img.shape) > 2:
        return median_filter(img, size=3, axes=(0, 1))
    else:
        return median_filter(img, size=3)


def apply_median_filter_ycbcr(img_bgr: np.ndarray) -> np.ndarray:
    """Apply a median filter in the YCbCr color space.

    Converts the input BGR image to YCbCr, applies median filtering 
    on each channel, and converts the result back to BGR. 
    This approach helps preserve color consistency while denoising.

    Args:
        img_bgr (np.ndarray): Input image in BGR color space.

    Returns:
        np.ndarray: Denoised image in BGR color space.
    """
    img_ycbcr = ycbcr.convert_img_to_ycbcr(img_bgr)
    filtered_img_ycbcr = apply_median_filter(img_ycbcr)
    filtered_img_bgr = ycbcr.reconvert_img_to_bgr(filtered_img_ycbcr)
    return filtered_img_bgr


def _mad_sigma(band: np.ndarray) -> float:
    """Median Absolute Deviation estimator of noise σ from a detail band."""
    return float(np.median(np.abs(band)) / 0.6745)


def _soft(x: np.ndarray, T: float) -> np.ndarray:
    """Soft-thresholding."""
    return np.sign(x) * np.maximum(np.abs(x) - T, 0.0)


def _levels_heuristic(h: int, w: int) -> int:
    """Heuristic for wavelet levels based on image size."""
    return max(2, min(5, int(np.floor(np.log2(min(h, w))) - 2)))


def denoise_wavelet_gray(img_gray01: np.ndarray,
                         wavelet: str = "db2",
                         levels: int | None = None,
                         rule: str = "bayes") -> tuple[np.ndarray, float]:
    """Wavelet denoise (grayscale) with BayesShrink/VisuShrink; returns (denoised, sigma).

    Args:
      img_gray01: Grayscale image in [0,1].
      wavelet: PyWavelets wavelet name (e.g., 'db2', 'sym4').
      levels: Decomposition levels; if None, chosen heuristically.
      rule: 'bayes' (adaptive) or 'visu' (universal threshold).

    Returns:
      (denoised_gray01, sigma_est)
    """
    x = np.clip(img_gray01.astype(np.float32), 0, 1)
    h, w = x.shape
    if levels is None:
        levels = _levels_heuristic(h, w)

    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=levels, mode="symmetric")
    cA, detail_levels = coeffs[0], coeffs[1:]

    # Noise estimate from finest diagonal band
    cH1, cV1, cD1 = detail_levels[-1]
    sigma = _mad_sigma(cD1)

    new_details = []
    N = x.size
    eps = 1e-8

    for (cH, cV, cD) in detail_levels:
        if rule.lower().startswith("visu"):
            T = sigma * np.sqrt(2.0 * np.log(N))
            cH_t = _soft(cH, T)
            cV_t = _soft(cV, T)
            cD_t = _soft(cD, T)
        else:  # BayesShrink per subband
            def bayes_T(sub):
                var_w = float(np.var(sub, ddof=1))
                sigma_x = np.sqrt(max(var_w - sigma**2, 0.0))
                return (sigma**2) / (sigma_x + eps) if sigma_x > 0 else np.max(np.abs(sub)) + 1.0
            TH, TV, TD = bayes_T(cH), bayes_T(cV), bayes_T(cD)
            cH_t = _soft(cH, TH)
            cV_t = _soft(cV, TV)
            cD_t = _soft(cD, TD)
        new_details.append((cH_t, cV_t, cD_t))

    den = pywt.waverec2([cA] + new_details, wavelet=wavelet, mode="symmetric")
    return np.clip(den, 0.0, 1.0), float(sigma)


def _cycle_spin_gray(x01: np.ndarray, spins: int, fn) -> np.ndarray:
    """Cycle-spinning for shift-invariance (grayscale)."""
    if spins <= 1:
        return fn(x01)
    H, W = x01.shape
    acc = np.zeros_like(x01, dtype=np.float32)
    for dy in range(spins):
        for dx in range(spins):
            s = np.roll(np.roll(x01, dy, axis=0), dx, axis=1)
            den = fn(s)
            acc += np.roll(np.roll(den, -dy, axis=0), -dx, axis=1)
    return acc / (spins * spins)


def _match_shape(ref, arr):
    h, w = ref.shape[:2]
    return arr[:h, :w]


def denoise_wavelet(img_bgr: np.ndarray,
                    wavelet: str = "db2",
                    levels: int | None = None,
                    luma_rule: str = "bayes",
                    chroma_rule: str = "bayes",
                    chroma_strength: float = 0.4,
                    spins_luma: int = 1,
                    spins_chroma: int = 1) -> np.ndarray:
    """Edge-preserving wavelet denoise for color images in YCrCb space.

    Strategy:
      - Convert BGR→YCrCb.
      - Stronger denoise on Y (luma); light denoise on Cr/Cb (optional).
      - Convert back to BGR.

    Args:
      img_bgr: Input image (uint8 or float in [0,1]).
      wavelet: Wavelet family for all channels (e.g., 'db2', 'sym4').
      levels: Wavelet levels; if None, chosen per-channel heuristically.
      luma_rule: 'bayes' or 'visu' for Y channel.
      chroma_rule: 'bayes' or 'visu' for chroma channels.
      chroma_strength: Blend factor [0..1] for chroma denoise amount
        (0 = keep original chroma; 1 = fully denoised chroma).
      spins_luma: Cycle-spins for Y (>=1).
      spins_chroma: Cycle-spins for Cr/Cb (>=1).

    Returns:
      Denoised BGR image (uint8).
    """
    # BGR → YCrCb
    ycrcb = ycbcr.convert_img_to_ycbcr(img_bgr)
    Y_u8, Cr_u8, Cb_u8 = cv2.split(ycrcb)

    # Prepare [0,1] floats
    Y = Y_u8.astype(np.float32) / 255.0
    Cr = Cr_u8.astype(np.float32) / 255.0
    Cb = Cb_u8.astype(np.float32) / 255.0

    # Luma denoise (stronger)
    def _den_luma(gray01):
        den, _ = denoise_wavelet_gray(
            gray01, wavelet=wavelet, levels=levels, rule=luma_rule)
        return den

    Y_den = _cycle_spin_gray(Y, spins_luma, _den_luma)
    Y_den = _match_shape(Y,  Y_den)

    # Chroma denoise (optional and lighter)
    if chroma_strength > 0:
        def _den_ch(gray01):
            lev = _levels_heuristic(
                *gray01.shape) if levels is None else max(1, levels - 1)
            den, _ = denoise_wavelet_gray(
                gray01, wavelet=wavelet, levels=lev, rule=chroma_rule)
            return den
        Cr_den = _cycle_spin_gray(Cr, spins_chroma, _den_ch)
        Cb_den = _cycle_spin_gray(Cb, spins_chroma, _den_ch)
        # ensure to keep the same shape as the original img
        Cr_den = _match_shape(Cr, Cr_den)
        Cb_den = _match_shape(Cb, Cb_den)
        # Blend to avoid color bleeding
        Cr_out = (1 - chroma_strength) * Cr + chroma_strength * Cr_den
        Cb_out = (1 - chroma_strength) * Cb + chroma_strength * Cb_den
    else:
        Cr_out, Cb_out = Cr, Cb

    # Repack and convert back
    ycrcb_out = cv2.merge([
        (np.clip(Y_den, 0, 1) * 255).astype(np.uint8),
        (np.clip(Cr_out, 0, 1) * 255).astype(np.uint8),
        (np.clip(Cb_out, 0, 1) * 255).astype(np.uint8),
    ])
    bgr_out = ycbcr.reconvert_img_to_bgr(ycrcb_out)
    return bgr_out
