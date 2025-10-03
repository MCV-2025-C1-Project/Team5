import numpy as np
from .gray_level_histogram import compute_histogram

def compute_rgb_histogram(
    img: np.ndarray, 
    values_per_bin: int = 1, 
    density: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute and concatenate the histograms of the R, G, and B channels of an image.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format (as loaded by cv2.imread).
    values_per_bin : int, optional
        Number of consecutive intensity values grouped into each bin. Defaults to 1 (256 bins).
    density : bool, optional
        If True, normalize each channel histogram so that the area under the histogram integrates to 1.

    Returns
    -------
    hist_concat : np.ndarray
        Concatenated histogram (R | G | B), each channel with bins depending on values_per_bin.
    bin_edges : np.ndarray
        Bin edges for the histograms (all channels share the same edges).
    """
    # Split channels (OpenCV uses BGR order)
    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]

    # Compute histogram for each channel
    hist_b, bin_edges = compute_histogram(b_channel, values_per_bin=values_per_bin, density=density)
    hist_g, _ = compute_histogram(g_channel, values_per_bin=values_per_bin, density=density)
    hist_r, _ = compute_histogram(r_channel, values_per_bin=values_per_bin, density=density)

    # Concatenate in RGB order (not BGR)
    hist_concat = np.concatenate([hist_r, hist_g, hist_b])

    return hist_concat.astype(np.float32), bin_edges.astype(np.float32)
