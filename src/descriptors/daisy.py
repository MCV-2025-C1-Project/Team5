

from collections import defaultdict
import math
from typing import List, Tuple, Callable, Optional

from skimage._shared.utils import check_nD
from skimage._shared.filters import gaussian
from skimage.util.dtype import img_as_float
import numpy as np
from numpy import arctan2, exp, pi, sqrt
import cv2

from src.visualization import plots
from src.data.extract import read_image
from src.descriptors.grayscale import convert_img_to_gray_scale


def compute_daisy_descriptor_from_array(
    img_bgr: np.ndarray,
    keypoint_detector: Callable[..., List[cv2.KeyPoint]],
    top_n: int = 500,
    adjust_to_size: bool = True,
    normalization: str = "l2",
    orientations: int = 8,
    visualize: bool = True,
    visualization_title: str = "DAISY descriptors",
    **keypoint_params
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Compute DAISY descriptors on an image array sampled at detected keypoints.

    This converts the image to grayscale in [0, 1], detects keypoints using
    `keypoint_detector`, computes DAISY on a dense grid, and samples
    descriptors at (rounded) keypoint locations. Optionally visualizes.

    Args:
        img_bgr: Input image in BGR format (H, W, 3), dtype uint8 or float.
        keypoint_detector: Callable that returns a list of `cv2.KeyPoint`.
            Extra kwargs are forwarded via `**keypoint_params`.
        top_n: Keep only the first `top_n` keypoints (after detector order).
        adjust_to_size: If True, group keypoints by size and compute DAISY
            with `radius=size` per group for better scale matching.
        normalization: Descriptor normalization mode: {"l1", "l2", "daisy", "off"}.
        orientations: Number of orientation bins; used by "daisy" normalization.
        visualize: If True, draws DAISY glyphs over the image.
        visualization_title: Title for the visualization window/figure.
        **keypoint_params: Extra parameters forwarded to `keypoint_detector`.

    Returns:
        A tuple `(keypoints, descs)` where:
            keypoints: List of the (possibly re-ordered) `cv2.KeyPoint`.
            descs: Array of shape (N, D) with float descriptors.

    Raises:
        ValueError: If normalization mode is invalid or shape assumptions fail.
    """

    # Convert to grayscale in [0,1]
    gray = convert_img_to_gray_scale(img_bgr)/255.0

    if visualize:
        # convert image to rgb
        descs_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)/255.0
    else:
        descs_img = None

    kwargs = {
        "normalization": normalization,
        "orientations": orientations,
        "visualize": visualize
    }

    keypoints = keypoint_detector(img_bgr, **keypoint_params)
    keypoints = keypoints[:top_n]
    visualization_title = f"Top {top_n} {visualization_title}"

    if adjust_to_size:
        # split keypoints by size
        keypoints_sizes = defaultdict(list)
        for keypoint in keypoints:
            keypoints_sizes[math.ceil(keypoint.size)].append(keypoint)

        # compute daisy descriptors for each size and concatenate
        descs = []
        keypoints_sorted_by_size = []
        for size, keypoints_size in keypoints_sizes.items():
            keypoints_sorted_by_size += keypoints_size
            descs_size, descs_img = daisy(
                gray,
                keypoints_size,
                radius=size,
                descs_img=descs_img,
                **kwargs
            )
            descs.append(descs_size)
        descs = np.vstack(descs)

        # sort descriptors in the initial keypoints order
        pairs = list(zip(keypoints_sorted_by_size, descs))
        pairs.sort(key=lambda x: x[0].response, reverse=True)
        keypoints, descs = zip(*pairs)

    else:
        descs, descs_img = daisy(
            gray, keypoints, descs_img=descs_img, **kwargs)

    descs = np.asarray(descs)
    descs = normalize_descriptors(descs, **kwargs)

    if visualize:
        plots.display_daisy_descriptors(descs_img, visualization_title)

    return keypoints, np.asarray(descs)


def daisy(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    radius: int = 15,
    rings: int = 3,
    histograms: int = 8,
    orientations: int = 8,
    normalization: str = 'l1',
    sigmas: Optional[List[float]] = None,
    ring_radii: Optional[List[float]] = None,
    visualize: bool = True,
    descs_img: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract DAISY descriptors at keypoint locations.

    Follows Tola et al. (2010) to build orientation histograms over concentric
    rings, with Gaussian smoothing in space and circular smoothing in angle.
    The dense descriptor volume is sampled at (rounded) keypoint coordinates.

    Args:
        image: Grayscale image (H, W) in float or uint8; converted to float in [0, 1].
        keypoints: List of `cv2.KeyPoint` whose (x, y) define sampling points.
        radius: Outer radius of the descriptor (pixels).
        rings: Number of concentric rings (excluding center).
        histograms: Number of angular samples per ring.
        orientations: Orientation bins per histogram.
        normalization: One of {"l1", "l2", "daisy", "off"} for descriptor norm.
        sigmas: Spatial Gaussian sigmas; length must be `rings + 1` if given
            (center + each ring). Overrides `rings` if provided.
        ring_radii: Radii per ring; length must be `rings`. Overrides `radius`
            and `rings` if provided.
        visualize: If True, returns an RGB image with DAISY glyphs drawn.
        descs_img: Optional RGB image to draw on (H, W, 3) in [0, 1] or [0, 255].

    Returns:
        A tuple `(descs, descs_img)` where:
            descs: Array of shape (N, D) with descriptors for valid keypoints.
            descs_img: Visualization RGB image if `visualize=True`, else None.

    Raises:
        ValueError: If `sigmas`/`ring_radii` lengths are inconsistent or
            `normalization` is invalid.
    """

    check_nD(image, 2, 'img')

    image = img_as_float(image)
    float_dtype = image.dtype

    # Validate parameters.
    if (
        sigmas is not None
        and ring_radii is not None
        and len(sigmas) - 1 != len(ring_radii)
    ):
        raise ValueError('`len(sigmas)-1 != len(ring_radii)`')
    if ring_radii is not None:
        rings = len(ring_radii)
        radius = ring_radii[-1]
    if sigmas is not None:
        rings = len(sigmas) - 1
    if sigmas is None:
        sigmas = [radius * (i + 1) / float(2 * rings) for i in range(rings)]
    if ring_radii is None:
        ring_radii = [radius * (i + 1) / float(rings) for i in range(rings)]
    if normalization not in ['l1', 'l2', 'daisy', 'off']:
        raise ValueError('Invalid normalization method.')

    # Compute image derivatives.
    dx = np.zeros(image.shape, dtype=float_dtype)
    dy = np.zeros(image.shape, dtype=float_dtype)
    dx[:, :-1] = np.diff(image, n=1, axis=1)
    dy[:-1, :] = np.diff(image, n=1, axis=0)

    # Compute gradient orientation and magnitude and their contribution
    # to the histograms.
    grad_mag = sqrt(dx**2 + dy**2)
    grad_ori = arctan2(dy, dx)
    orientation_kappa = orientations / pi
    orientation_angles = [2 * o * pi /
                          orientations - pi for o in range(orientations)]
    hist = np.empty((orientations,) + image.shape, dtype=float_dtype)
    for i, o in enumerate(orientation_angles):
        # Weigh bin contribution by the circular normal distribution
        hist[i, :, :] = exp(orientation_kappa * np.cos(grad_ori - o))
        # Weigh bin contribution by the gradient magnitude
        hist[i, :, :] = np.multiply(hist[i, :, :], grad_mag)

    # Smooth orientation histograms for the center and all rings.
    sigmas = [sigmas[0]] + sigmas
    hist_smooth = np.empty((rings + 1,) + hist.shape, dtype=float_dtype)
    for i in range(rings + 1):
        for j in range(orientations):
            hist_smooth[i, j, :, :] = gaussian(
                hist[j, :, :], sigma=sigmas[i], mode='reflect'
            )

    # Assemble descriptor grid.
    theta = [2 * pi * j / histograms for j in range(histograms)]
    desc_dims = (rings * histograms + 1) * orientations
    descs = np.empty(
        (desc_dims, image.shape[0] - 2 * radius, image.shape[1] - 2 * radius),
        dtype=float_dtype,
    )
    descs[:orientations, :, :] = hist_smooth[0,
                                             :, radius:-radius, radius:-radius]
    idx = orientations
    for i in range(rings):
        for j in range(histograms):
            y_min = radius + int(round(ring_radii[i] * math.sin(theta[j])))
            y_max = descs.shape[1] + y_min
            x_min = radius + int(round(ring_radii[i] * math.cos(theta[j])))
            x_max = descs.shape[2] + x_min
            descs[idx: idx + orientations, :, :] = hist_smooth[
                i + 1, :, y_min:y_max, x_min:x_max
            ]
            idx += orientations

    # Extract integer pixel coordinates from keypoints
    xs = np.round([kp.pt[0] for kp in keypoints]).astype(int)
    ys = np.round([kp.pt[1] for kp in keypoints]).astype(int)

    # Clip to valid bounds
    H, W = descs.shape[1:]
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys = xs[valid], ys[valid]

    # Extract descriptors at keypoint locations
    descs = descs[:, ys, xs]
    descs = descs.swapaxes(0, 1)

    if visualize:
        descs_img = plots.draw_daisy_descriptors(
            descs_img, descs, sigmas, orientation_angles, ring_radii, rings,
            theta, orientations, histograms, xs, ys)

    return descs, descs_img


def normalize_descriptors(
    descs: np.ndarray,
    normalization: str,
    orientations: int = 8,
    **kwargs
) -> np.ndarray:
    """Normalize a batch of DAISY descriptors.

    Args:
        descs: Descriptor array of shape (N, D).
        normalization: Mode {"l1", "l2", "daisy", "off"}.
            - "l1": L1-normalize each descriptor.
            - "l2": L2-normalize each descriptor.
            - "daisy": L2-normalize each orientation block of size `orientations`.
            - "off": Return descriptors unchanged.
        orientations: Orientation bins used to define block size for "daisy".

    Returns:
        Normalized descriptors with the same shape as `descs`.

    Raises:
        ValueError: If "daisy" is selected and `D` is not a multiple of `orientations`.
    """
    # Normalize descriptors.
    if normalization != 'off':
        if normalization == 'l1':
            descs /= (descs.sum(axis=-1, keepdims=True) + 1e-10)
        elif normalization == 'l2':
            descs /= (np.sqrt((descs * descs).sum(axis=-1, keepdims=True)) + 1e-10)
        elif normalization == 'daisy':
            # Expect last dim C to be a multiple of `orientations`
            C = descs.shape[-1]
            if C % orientations != 0:
                raise ValueError(
                    f"C={C} is not a multiple of orientations={orientations} "
                    f"for 'daisy' normalization.")
            B = C // orientations  # number of orientation-blocks

            # Reshape to (..., B, orientations), normalize per last axis, then reshape back
            new_shape = (*descs.shape[:-1], B, orientations)
            x = descs.reshape(new_shape)
            denom = np.sqrt((x * x).sum(axis=-1, keepdims=True)) + 1e-10
            x = x / denom
            descs = x.reshape(descs.shape)

    return descs


def compute_daisy_descriptor(
    img_path: str,
    **kwargs
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Compute DAISY descriptors from an image path.

    Reads the image (BGR), detects keypoints, computes DAISY, and returns
    keypoints with their descriptors. Extra kwargs are forwarded to
    `compute_daisy_descriptor_from_array`.

    Args:
        img_path: Path to the input image file.
        **kwargs: Forwarded to `compute_daisy_descriptor_from_array`
            (e.g., keypoint_detector, top_n, adjust_to_size, normalization, etc.).

    Returns:
        A tuple `(keypoints, descs)` where:
            keypoints: List of detected `cv2.KeyPoint`.
            descs: Array of shape (N, D) with DAISY descriptors.
    """
    img_bgr = read_image(img_path)
    return compute_daisy_descriptor_from_array(
        img_bgr,
        **kwargs
    )
