import matplotlib.pyplot as plt
import numpy as np


def display_image(img: np.ndarray) -> None:
    """Display an image using Matplotlib.

    Args:
        img (np.ndarray): Image array to display.
    """
    plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
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
