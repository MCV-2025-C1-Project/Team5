import cv2
import numpy as np

def convert_img_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to CIELab (Lab) color space.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format (cv2.imread).

    Returns
    -------
    np.ndarray
        Lab image (OpenCV convention: 
        L ∈ [0,255], 
        a,b ∈ [0,255] with 128 as neutral).
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

def compute_histogram(img_lab: np.ndarray,
                      bins: int = 64,
                      c_max: float = 150.0,
                      normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 1-D histogram of C*_ab (the chroma) in an image already in Lab space.

    C*_ab = sqrt(a^2 + b^2) represents how "colourful" a pixel is, independent of lightness.

    Parameters
    ----------
    img_lab : np.ndarray
        Input image in Lab format (OpenCV convention:
        L ∈ [0,255], a,b ∈ [0,255] with 128 as neutral).
    bins : int, optional
        Number of histogram bins for C*_ab.
    c_max : float, optional
        Maximum chroma value for clipping. Values above c_max are clipped.
    normalize : bool, optional
        If True, divide by sum so the histogram sums to 1.

    Returns
    -------
    hist : np.ndarray of shape (bins,)
        Histogram values (counts or normalised).
    bin_edges : np.ndarray of shape (bins+1,)
        The edges of the histogram bins (useful for plotting).
    """
    # Extract a* and b* channels and re-center around 0.
    a = img_lab[:, :, 1].astype(np.float32) - 128.0
    b = img_lab[:, :, 2].astype(np.float32) - 128.0

    # Compute chroma C*_ab = sqrt(a^2 + b^2).
    c = np.sqrt(a * a + b * b)

    # Clip extreme values so the histogram range stays [0,c_max].
    c = np.clip(c, 0, c_max)

    # Compute histogram of chroma values 
    hist, bin_edges = np.histogram(c, bins=bins, range=(0, c_max))

    # If normalize=True and the histogram is non-empty,
    # divide by its sum so that it adds up to 1 (L1 normalization)
    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()

    return hist.astype(np.float32), bin_edges.astype(np.float32)

   