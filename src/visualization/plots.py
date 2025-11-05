"""
Visualization utilities: display images and histograms.
"""

from typing import List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import cv2


def display_image(img_bgr: np.ndarray, ax=None, title: str = None, **kwargs:None) -> None:
    """
    Display an image using Matplotlib.

    Args:
        img_bgr (np.ndarray): Image array in BGR format.
        ax (matplotlib.axes.Axes, optional): Axis to draw the image on. If None, creates a new figure.
        title (str, optional): Title to display above the image.
        **kwargs: Additional arguments forwarded to plt.imshow() or ax.imshow().
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if ax is not None:
        ax.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None, **kwargs)
        if title:
            ax.set_title(title)
        ax.axis("off")
        return ax
    else:
        plt.figure(figsize=(12, 6))
        plt.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None, **kwargs)
        if title:
            plt.title(title)
        plt.axis("off")
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
    ax.bar(bins, hist_concat[:n_bins],
           width=widths, alpha=0.5, label='L', color='#333333')
    ax.bar(bins, hist_concat[n_bins:2*n_bins],
           width=widths, alpha=0.5, label='a', color='#D55E00')
    ax.bar(bins, hist_concat[2*n_bins:],
           width=widths, alpha=0.5, label='b', color='#0072B2')

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
    ax.bar(bins, hist_concat[:n_bins],         width=widths,
           alpha=0.5, label='H', color='#8E44AD')  # purple
    ax.bar(bins, hist_concat[n_bins:2*n_bins], width=widths,
           alpha=0.5, label='S', color='#27AE60')  # green
    ax.bar(bins, hist_concat[2*n_bins:],       width=widths,
           alpha=0.5, label='V', color='#F1C40F')  # yellow

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
    ax.bar(bins, hist_concat[:n_bins],
           width=widths, alpha=0.5, label='Y',  color='#444444')
    ax.bar(bins, hist_concat[n_bins:2*n_bins],
           width=widths, alpha=0.5, label='Cb', color='#1f77b4')
    ax.bar(bins, hist_concat[2*n_bins:],
           width=widths, alpha=0.5, label='Cr', color='#d62728')

    ax.set_xlabel("Bin")
    ax.set_ylabel("Frequency")
    ax.set_title("YCbCr Histogram (superposed channels)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def display_2d_histogram(hist: np.ndarray, bin_edges: np.ndarray) -> None:
    """Display a 2D histogram as a 3D bar plot.

    Args:
        hist (np.ndarray): 2D array representing histogram frequencies.
        bin_edges (np.ndarray): Sequence of arrays containing the bin edges
            for each histogram dimension.

    Returns:
        None: Displays a 3D histogram visualization.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(
        bin_edges[0][:-1] + 0.25,
        bin_edges[1][:-1] + 0.25,
        indexing="ij"
    )
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    # Construct arrays with the dimensions for all the bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.set_zlabel("Frequency")

    plt.show()


def draw_keypoints(
        img_bgr: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        color: Tuple[int, int, int] = (0, 255, 0),
        draw_rich: bool = False,
        **kwargs: Any
) -> None:
    """Draws cv2.KeyPoint objects on an image and displays it.

    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        keypoints (List[cv2.KeyPoint]): List of keypoints detected (e.g., from Harris or LoG).
        color (Tuple[int, int, int], optional): Color used to draw the keypoints (BGR). Defaults to green.
        draw_rich (bool, optional): If True, draws the keypoints with size and orientation
            (like in SIFT/ORB visualization). Defaults to False.
        **kwargs (Any): Additional keyword arguments forwarded to `display_image()`.

    Returns:
        None
    """
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_rich else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    out = cv2.drawKeypoints(img_bgr, keypoints, None, color=color, flags=flags)
    display_image(out, **kwargs)

def draw_matches(
    img1: np.ndarray,
    kps1: List[cv2.KeyPoint],
    img2: np.ndarray,
    kps2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    title: str = "",
    resize: bool = False,
    resize_height: int = 600,
    draw_rich: bool = False,
    **kwargs: Any
) -> np.ndarray:
    """
    Draw matches between two images with optional resizing (and keypoint scaling) for better visualization.

    Args:
        img1 (np.ndarray): Query image (BGR).
        kps1 (List[cv2.KeyPoint]): Keypoints from the query image.
        img2 (np.ndarray): Database image (BGR).
        kps2 (List[cv2.KeyPoint]): Keypoints from the database image.
        matches (List[cv2.DMatch]): Matches to draw.
        title (str, optional): Title for the plot.
        resize (bool, optional): Whether to resize both images to a common height for display. Defaults to False.
        resize_height (int, optional): Target height when resizing. Defaults to 600.
        draw_rich (bool, optional): If True, draws keypoints with size and orientation. Defaults to False.
        **kwargs (Any): Extra arguments passed to display_image().

    Returns:
        np.ndarray: The final image with matches drawn.
    """

    def scale_keypoints(keypoints: List[cv2.KeyPoint], scale: float) -> List[cv2.KeyPoint]:
        """Scale keypoint coordinates and sizes by a factor."""
        scaled = []
        for kp in keypoints:
            scaled.append(cv2.KeyPoint(
                x=kp.pt[0] * scale,
                y=kp.pt[1] * scale,
                size=kp.size * scale,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id
            ))
        return scaled

    def resize_to_height(img: np.ndarray, kps: List[cv2.KeyPoint], height: int):
        """Resize image and scale keypoints proportionally."""
        h, w = img.shape[:2]
        scale = height / float(h)
        img_resized = cv2.resize(img, (int(w * scale), height))
        kps_scaled = scale_keypoints(kps, scale)
        return img_resized, kps_scaled

    # Resize if True
    if resize:
        img1, kps1 = resize_to_height(img1, kps1, resize_height)
        img2, kps2 = resize_to_height(img2, kps2, resize_height)

    # Draw matches
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_rich else cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    img_matches = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=flags)

    # Display
    display_image(img_matches, title=title, **kwargs)
    return img_matches




