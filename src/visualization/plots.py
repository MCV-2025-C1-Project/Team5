"""
Visualization utilities: display images and histograms.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2


def display_image(img_bgr: np.ndarray) -> None:
    """
    Display an image using Matplotlib.

    Args:
        img (np.ndarray): Image array to display.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None)
    plt.show()


def display_histogram(hist: np.ndarray, bin_edges: np.ndarray) -> None:
    """Display a histogram as a bar plot.

    Args:
        hist (np.ndarray): Histogram values.
        bin_edges (np.ndarray): Bin edges.

    Raises:
        ValueError: If histogram and bin lengths are incompatible.
    """
    # bin edges -> remove last edge to have one bin for each value
    if len(hist) == len(bin_edges)-1:
        bins = bin_edges[:-1]
    elif len(hist) != len(bin_edges):
        raise ValueError(
            f"Histogram length ({len(hist)}) does not match bins length ({len(bin_edges)})"
        )

    plt.bar(bins, hist, width=np.diff(bin_edges))
    plt.show()


def display_rgb_histogram(hist_concat: np.ndarray, bin_edges: np.ndarray) -> None:
    """
    Display concatenated RGB histogram as a bar plot, superposing the three channels.

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
    ax.bar(bins, hist_concat[:n_bins], color='r',
           width=np.diff(bin_edges), label='Red', alpha=0.5)
    ax.bar(bins, hist_concat[n_bins:2 * n_bins], color='g',
           width=np.diff(bin_edges), label='Green', alpha=0.5)
    ax.bar(bins, hist_concat[2 * n_bins:], color='b',
           width=np.diff(bin_edges), label='Blue', alpha=0.5)

    ax.set_xlabel("Bin")
    ax.set_ylabel("Frequency")
    ax.set_title("RGB Histogram (superposed channels)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def display_lab_histogram(hist_concat: np.ndarray, bin_edges: np.ndarray) -> None:
    """
    Display concatenated LAB histogram as a bar plot, superposing the three channels.

    Args:
        hist_concat (np.ndarray): Concatenated histogram values for L, a, b channels (length 3*N).
        bin_edges (np.ndarray): Bin edges for a single channel (length N+1 for N bins).

    Raises:
        ValueError: If histogram and bin_edges lengths are incompatible.
    """
    n_bins = len(bin_edges) - 1
    if hist_concat.shape[0] != 3 * n_bins:
        raise ValueError(
            f"Expected concatenated histogram of length {3 * n_bins}, got {hist_concat.shape[0]}"
        )

    bins = bin_edges[:-1]
    widths = np.diff(bin_edges)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bins, hist_concat[:n_bins],           width=widths, alpha=0.5, label='L', color='#333333')
    ax.bar(bins, hist_concat[n_bins:2*n_bins],   width=widths, alpha=0.5, label='a', color='#D55E00')
    ax.bar(bins, hist_concat[2*n_bins:],         width=widths, alpha=0.5, label='b', color='#0072B2')

    ax.set_xlabel("Bin")
    ax.set_ylabel("Frequency")
    ax.set_title("LAB Histogram (superposed channels)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def display_hsv_histogram(hist_concat: np.ndarray, bin_edges: np.ndarray) -> None:
    """
    Display concatenated HSV histogram as a bar plot, superposing the three channels.

    Args:
        hist_concat (np.ndarray): Concatenated histogram values for H, S, V channels (length 3*N).
        bin_edges (np.ndarray): Bin edges for a single channel (length N+1 for N bins).

    Raises:
        ValueError: If histogram and bin_edges lengths are incompatible.
    """
    n_bins = len(bin_edges) - 1
    if hist_concat.shape[0] != 3 * n_bins:
        raise ValueError(
            f"Expected concatenated histogram of length {3 * n_bins}, got {hist_concat.shape[0]}"
        )

    bins = bin_edges[:-1]
    widths = np.diff(bin_edges)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bins, hist_concat[:n_bins],         width=widths, alpha=0.5, label='H', color='#8E44AD')  # purple
    ax.bar(bins, hist_concat[n_bins:2*n_bins], width=widths, alpha=0.5, label='S', color='#27AE60')  # green
    ax.bar(bins, hist_concat[2*n_bins:],       width=widths, alpha=0.5, label='V', color='#F1C40F')  # yellow

    ax.set_xlabel("Bin")
    ax.set_ylabel("Frequency")
    ax.set_title("HSV Histogram (superposed channels)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def display_ycbcr_histogram(hist_concat: np.ndarray, bin_edges: np.ndarray) -> None:
    """
    Display concatenated YCbCr histogram as a bar plot, superposing the three channels.

    Note: Assumes hist_concat is ordered as [Y | Cb | Cr] (length 3*N), and that
    OpenCV conversion was BGR→YCrCb but channels were reordered to Y, Cb, Cr.

    Args:
        hist_concat (np.ndarray): Concatenated histogram values for Y, Cb, Cr (length 3*N).
        bin_edges (np.ndarray): Bin edges for a single channel (length N+1 for N bins).

    Raises:
        ValueError: If histogram and bin_edges lengths are incompatible.
    """
    n_bins = len(bin_edges) - 1
    if hist_concat.shape[0] != 3 * n_bins:
        raise ValueError(
            f"Expected concatenated histogram of length {3 * n_bins}, got {hist_concat.shape[0]}"
        )

    bins = bin_edges[:-1]
    widths = np.diff(bin_edges)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bins, hist_concat[:n_bins],           width=widths, alpha=0.5, label='Y',  color='#444444')
    ax.bar(bins, hist_concat[n_bins:2*n_bins],   width=widths, alpha=0.5, label='Cb', color='#1f77b4')
    ax.bar(bins, hist_concat[2*n_bins:],         width=widths, alpha=0.5, label='Cr', color='#d62728')

    ax.set_xlabel("Bin")
    ax.set_ylabel("Frequency")
    ax.set_title("YCbCr Histogram (superposed channels)")
    ax.legend()
    plt.tight_layout()
    plt.show()

