import cv2
import numpy as np 

def convert_img_to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to HSV color space.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format (cv2.imread).

    Returns
    -------
    np.ndarray
        HSV image (OpenCV convention: H∈[0,180), S,V∈[0,255]).
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def compute_histogram(img: np.ndarray,
                      bins: int = 72,
                      s_min: float = 0.10,
                      density: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 1-D histogram of the Hue channel in an HSV image,
    weighted by Saturation so that strongly coloured pixels have more influence.

    Parameters
    ----------
    img : np.ndarray
        Input image already in HSV color space (H∈[0,180), S,V∈[0,255]).
    bins : int, optional
        Number of histogram bins for Hue (OpenCV Hue values are 0..180).
    s_min : float, optional
        Minimum saturation threshold (0..1). Pixels below this threshold are ignored.
    density : bool, optional
        If True, normalize the histogram so that the area under it integrates to 1.  
        If False, return raw weighted counts. Defaults to True.

    Returns
    -------
    hist : np.ndarray of shape (bins,)
        Histogram values (counts or normalized).
    bin_edges : np.ndarray of shape (bins+1,)
        The edges of the histogram bins.
    """
    # Hue channel [0,180)
    h = img[:, :, 0].astype(np.float32)

    # Saturation channel scaled to [0,1]
    s = img[:, :, 1].astype(np.float32) / 255.0

    # Assign a weight to each pixel:
    #   - If its saturation is >= s_min, use the saturation value as weight.
    #   - If its saturation is < s_min, give it weight 0 (ignore it).
    weights = np.where(s >= s_min, s, 0.0)

    # Weighted Hue histogram with 'bins' bins over [0,180)
    hist, bin_edges = np.histogram(
        h,
        bins=bins,
        range=(0, 180),
        weights=weights,
        density=density
    )

    return hist.astype(np.float32), bin_edges.astype(np.float32)