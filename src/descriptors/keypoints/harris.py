import cv2
import numpy as np
from typing import List, Iterable

from src.data.extract import read_image
from src.descriptors.grayscale import convert_img_to_gray_scale


def local_maxima(
    resp: np.ndarray, nms_ksize: int = 3, thresh: float | None = None
) -> np.ndarray:
    """Finds local maxima in a response map above a given threshold.

    Applies non-maximum suppression using morphological dilation and
    returns a boolean mask indicating the peak locations.

    Args:
        resp (np.ndarray): Input response map (e.g., corner or saliency map).
        nms_ksize (int, optional): Kernel size for non-maximum suppression.
            Larger values produce sparser detections. Defaults to 3.
        thresh (float | None, optional): Absolute threshold. If None, defaults
            to 1% of the maximum response value.

    Returns:
        np.ndarray: Boolean array where True marks local maxima above threshold.
    """
    if thresh is None:
        thresh = 0.01 * float(resp.max())
    k = max(1, int(nms_ksize))
    dil = cv2.dilate(resp, np.ones((k, k), np.uint8))
    peaks = (resp == dil) & (resp > thresh)
    return peaks


def detect_harris_keypoints_from_array(
    img_bgr: np.ndarray,
    blockSize: int = 3,
    ksize: int = 3,
    k: float = 0.04,
    thresh_rel: float = 0.4,
    nms_ksize: int = 3,
    **kwargs
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """
    Detect Harris keypoints from an image array and return cv2.KeyPoint objects.

    Args:
        img_bgr (np.ndarray): Input BGR image.
        blockSize (int, optional): Neighborhood size for the structure tensor.
            Defaults to 3.
        ksize (int, optional): Aperture parameter for the Sobel operator.
            Defaults to 3.
        k (float, optional): Harris detector free parameter (0.04-0.06).
            Defaults to 0.04.
        thresh_rel (float, optional): Relative threshold for corner response.
            Defaults to 0.4.
        nms_ksize (int, optional): Kernel size for non-maximum suppression.
            Defaults to 3.

    Returns:
        tuple[list[cv2.KeyPoint], np.ndarray]: List of detected keypoints and
            the Harris response map.
    """

    img_gray = convert_img_to_gray_scale(img_bgr)
    img_gray = img_gray.astype(np.float32) / 255.0

    R = cv2.cornerHarris(img_gray, blockSize=blockSize, ksize=ksize, k=k)
    # Normalize to [0,1] for stability (optional)
    Rn = cv2.normalize(R, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    peaks = local_maxima(Rn, nms_ksize=nms_ksize, thresh=thresh_rel * Rn.max())

    # Convert to cv2.KeyPoint list
    ys, xs = np.where(peaks)
    keypoints = [
        cv2.KeyPoint(
            float(x),
            float(y),
            size=blockSize,
            response=float(Rn[y, x])
        )
        for y, x in zip(ys, xs)
    ]

    # Sort keypoints by response descending
    keypoints.sort(key=lambda k: k.response, reverse=True)
    return keypoints


def detect_harris_keypoints(
    img_path: str,
    **kwargs
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detects Harris keypoints from an image path.

    Reads an image from the provided path and applies the Harris corner
    detection algorithm with optional parameters forwarded to
    `detect_harris_keypoints_from_array`.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (Any): Additional parameters passed to
            `detect_harris_keypoints_from_array`, such as
            `blockSize`, `ksize`, `k`, `thresh_rel`, or `nms_ksize`.

    Returns:
        List[cv2.KeyPoint]: List of detected Harris keypoints.
    """

    img_bgr = read_image(img_path)
    return detect_harris_keypoints_from_array(img_bgr, **kwargs)


def detect_harris_laplacian_keypoints_from_array(
    img_bgr: np.ndarray,
    sigmas: Iterable[float] = (
        1, 1.2, 1.6, 2.2, 3.0, 4.2, 6.0, 8.0, 16.0, 24.0, 30.0, 50.0),
    blockSize: int = 3,
    ksize: int = 3,
    k: float = 0.04,
    harris_thresh_rel: float = 0.01,
    harris_thresh_pct: float = 99.5,
    base_nms_ksize: int = 3,
    **kwargs
) -> List[cv2.KeyPoint]:
    """Harris-Laplacian keypoints with per-scale thresholds and NMS scaling.

    Steps per σ:
      1) Blur image with Gaussian(σ).
      2) Harris response on blurred image; normalize per-scale.
      3) Spatial NMS with window ~ σ.
      4) Scale-normalized Laplacian LoG = σ² * Laplacian(Gσ * I).
      5) Keep locations that are Harris peaks AND |LoG| scale-maxima.

    Args:
        img_bgr: BGR image (uint8 or float).
        sigmas: Gaussian scales.
        blockSize: Harris structure tensor neighborhood.
        ksize: Sobel aperture for gradients (3, 5, 7).
        k: Harris free parameter.
        harris_thresh_rel: Relative-to-max per-scale Harris threshold.
        harris_thresh_pct: Robust percentile per-scale (e.g., 99.5) combined with rel.
        base_nms_ksize: Base NMS kernel; actual kernel scales with σ.

    Returns:
        List[cv2.KeyPoint]: Keypoints with .pt=(x,y), .size≈3σ, .response=|LoG|.
    """
    # Grayscale [0,1]
    gray = convert_img_to_gray_scale(img_bgr).astype(np.float32)
    if gray.max() > 1.0:
        gray /= 255.0

    sigmas = list(sigmas)
    H_masks = []          # per-scale Harris peak masks (bool)
    LoGs = []             # per-scale scale-normalized Laplacian (float32)
    # per-scale normalized Harris (for debugging/thresholding)
    H_norms = []

    for sigma in sigmas:
        # 1) smoothing at scale
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # 2) Harris response (then normalize to [0,1] per-scale)
        H = cv2.cornerHarris(blur, blockSize=blockSize, ksize=ksize, k=k)
        Hn = cv2.normalize(H, None, 0, 1, cv2.NORM_MINMAX)

        # 3) per-scale adaptive threshold and NMS window ~ σ
        rel_th = float(harris_thresh_rel) * float(Hn.max())
        pct_th = float(np.percentile(Hn, harris_thresh_pct))
        th = max(rel_th, pct_th)

        # scale NMS: base + 2*σ rounded to nearest odd
        nms_k = int(max(base_nms_ksize, 2 * sigma + 1))
        if nms_k % 2 == 0:
            nms_k += 1

        peaks = local_maxima(Hn, nms_ksize=nms_k, thresh=th)

        # 4) scale-normalized Laplacian
        lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        log = (sigma ** 2) * lap

        H_masks.append(peaks.astype(bool))
        LoGs.append(log.astype(np.float32))
        H_norms.append(Hn)

    H_masks = np.stack(H_masks, axis=0)     # [S,H,W], bool
    LoGs = np.stack(LoGs, axis=0)           # [S,H,W], float32

    # 5) scale-space maxima in |LoG|
    absLoG = np.abs(LoGs)
    S = len(sigmas)

    keypoints: List[cv2.KeyPoint] = []
    H, W = H_masks.shape[1:]

    for s in range(S):
        prevA = absLoG[s - 1] if s - 1 >= 0 else -np.inf
        nextA = absLoG[s + 1] if s + 1 < S else -np.inf
        curA = absLoG[s]

        scale_max = (curA >= prevA) & (curA >= nextA)
        mask = scale_max & H_masks[s]

        ys, xs = np.where(mask)
        sigma = float(sigmas[s])
        # visualization scale; tweak (2-4)σ if you prefer
        size = float(3.0 * sigma)

        for y, x in zip(ys, xs):
            kp = cv2.KeyPoint(
                x=float(x),
                y=float(y),
                size=size,
                response=float(curA[y, x])
            )
            keypoints.append(kp)

    # Sort keypoints by response descending
    keypoints.sort(key=lambda k: k.response, reverse=True)
    return keypoints


def detect_harris_laplacian_keypoints(
    img_path: str,
    **kwargs
) -> List[cv2.KeyPoint]:
    """Detects Harris-Laplacian keypoints from an image path.

    Reads an image from the given path, processes it across multiple scales
    using the Harris-Laplacian detector, and returns OpenCV keypoint objects.

    Args:
        img_path (str): Path to the input image file.
        **kwargs (Any): Additional parameters passed to
            `detect_harris_laplacian_keypoints_from_array`, such as
            `sigmas`, `blockSize`, `k`, `harris_thresh_rel`, or `nms_ksize`.

    Returns:
        List[cv2.KeyPoint]: List of multi-scale keypoints with position, size,
            and response values.
    """
    img_bgr = read_image(img_path)
    return detect_harris_laplacian_keypoints_from_array(img_bgr, **kwargs)
