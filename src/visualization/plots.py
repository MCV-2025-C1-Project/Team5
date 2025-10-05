"""
Visualization utilities: display images (BGR→RGB) and histograms.
"""


import matplotlib.pyplot as plt
import numpy as np
import cv2


def display_image(img_bgr: np.ndarray) -> None:
    """Display an image using Matplotlib.

    Args:
        img (np.ndarray): Image array to display.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None)
    plt.show()


def display_histogram(hist: np.ndarray, bins: np.ndarray) -> None:
    """Display a histogram as a bar plot.

    Args:
        hist (np.ndarray): Histogram values.
        bins (np.ndarray): Bin edges.

    Raises:
        ValueError: If histogram and bin lengths are incompatible.
    """
    # bin edges -> remove last edge to have one bin for each value
    if len(hist) == len(bins)-1:
        bins = bins[:-1]
    elif len(hist) != len(bins):
        raise ValueError(
            f"Histogram length ({len(hist)}) does not match bins length ({len(bins)})"
        )

    plt.bar(bins, hist)
    plt.show()

def display_rgb_histogram(hist_concat: np.ndarray, bin_edges: np.ndarray) -> None:
    """Display concatenated RGB histogram as a bar plot, superposing the three channels.

    Args:
        hist_concat (np.ndarray): Concatenated histogram values for R, G, B channels (length 3*N).
        bin_edges (np.ndarray): Bin edges for a single channel (length N+1 for N bins).

    Raises:
        ValueError: If histogram and bin_edges lengths are incompatible.
    """
    n_bins = len(bin_edges) - 1
    if hist_concat.shape[0] != 3 * n_bins:
        raise ValueError(
            f"Expected concatenated histogram of length {3 * n_bins}, got {hist_concat.shape[0]}"
        )

    bins = bin_edges[:-1]  # Use left edges for bar positions

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot each channel's histogram superposed
    ax.bar(bins, hist_concat[:n_bins], color='r', width=np.diff(bin_edges), label='Red', alpha=0.5)
    ax.bar(bins, hist_concat[n_bins:2 * n_bins], color='g', width=np.diff(bin_edges), label='Green', alpha=0.5)
    ax.bar(bins, hist_concat[2 * n_bins:], color='b', width=np.diff(bin_edges), label='Blue', alpha=0.5)

    ax.set_xlabel("Bin")
    ax.set_ylabel("Frequency")
    ax.set_title("RGB Histogram (superposed channels)")
    ax.legend()
    plt.tight_layout()
    plt.show()
