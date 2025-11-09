import numpy as np
import os
import cv2
from scipy.ndimage import (
    binary_opening, binary_closing, binary_fill_holes, gaussian_filter
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
import argparse
warnings.filterwarnings('ignore')

from src.data.extract import read_image
from src.metrics.mask_metrics import (
    compute_confusion_counts_mask,
    precision_mask,
    recall_mask,
    f1_mask
)
from src.visualization.viz import (
    apply_mask_to_image_mask,
    create_visualization_mask,
    visualize_color_pipeline_steps_mask,
    create_summary_plots_mask,
    save_results_mask
)
from src.descriptors.grayscale import convert_img_to_gray_scale
from src.descriptors.lab import convert_img_to_lab
from src.descriptors.hsv import convert_img_to_hsv


def apply_gaussian_filter(image: np.ndarray, sigma: float = 1.0, radius: int = 2) -> np.ndarray:
    """
    Apply Gaussian filter to denoise image.

    Args:
        image (np.ndarray): Input image (H, W, C).
        sigma (float): Standard deviation for Gaussian kernel.
        radius (int): Radius of Gaussian kernel (truncate parameter).

    Returns:
        np.ndarray: Filtered image.
    """
    # Apply Gaussian filter to each channel
    filtered = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        filtered[:, :, c] = gaussian_filter(
            image[:, :, c].astype(np.float32),
            sigma=sigma,
            truncate=radius
        )

    return filtered.astype(image.dtype)


def _estimate_bg_from_borders(image_lab: np.ndarray, border: int = 20) -> Dict[str, np.ndarray]:
    """
    Estimate background color statistics from the image borders in LAB space.

    Args:
        image_lab (np.ndarray): Image in LAB color space.
        border (int): Width of border pixels to sample for background estimation.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing 'median' and 'mad' statistics for each channel.
    """
    H, W, _ = image_lab.shape
    b = max(1, min(border, min(H, W)//4))

    top    = image_lab[:b, :, :]
    bottom = image_lab[-b:, :, :]
    left   = image_lab[:, :b, :]
    right  = image_lab[:, -b:, :]

    borders = np.concatenate(
        [top.reshape(-1,3), bottom.reshape(-1,3), left.reshape(-1,3), right.reshape(-1,3)],
        axis=0
    )

    med = np.median(borders, axis=0)                            # (3,)
    mad = np.median(np.abs(borders - med), axis=0) + 1e-6       # (3,) avoid /0
    return {"median": med.astype(np.float32), "mad": mad.astype(np.float32)}

def _lab_robust_distance_weighted(L: np.ndarray, A: np.ndarray, B: np.ndarray,
                                  med: np.ndarray, mad: np.ndarray,
                                  wL: float = 0.6, wA: float = 1.0, wB: float = 1.0) -> np.ndarray:
    """
    Weighted distance to BG center in LAB, normalized by MAD per channel.

    Args:
        L (np.ndarray): Lightness channel.
        A (np.ndarray): A channel (green-red).
        B (np.ndarray): B channel (blue-yellow).
        med (np.ndarray): Median values for each LAB channel.
        mad (np.ndarray): MAD (Median Absolute Deviation) values for each LAB channel.
        wL (float): Weight for L channel to downweight illumination.
        wA (float): Weight for A channel.
        wB (float): Weight for B channel.

    Returns:
        np.ndarray: Weighted distance to background center.
    """
    dL = (L - med[0]) / mad[0]
    dA = (A - med[1]) / mad[1]
    dB = (B - med[2]) / mad[2]
    dist = np.sqrt((wL*dL)**2 + (wA*dA)**2 + (wB*dB)**2)
    return dist

def _hue_circular_distance(h: np.ndarray, h0: float) -> np.ndarray:
    """
    Circular distance on hue (OpenCV H in [0,180)).

    Args:
        h (np.ndarray): Hue values in range [0,180).
        h0 (float): Reference hue value.

    Returns:
        np.ndarray: Circular distance in degrees in range [0,90].
    """
    dh = np.abs(h - h0)
    dh = np.minimum(dh, 180.0 - dh)
    return dh


def _morphological_area_opening(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Area opening morphology to remove small blobs.

    Args:
        mask (np.ndarray): Binary mask to process.
        min_area (int): Minimum area threshold for blob removal.

    Returns:
        np.ndarray: Processed mask with small blobs removed.
    """
    if min_area <= 0:
        return mask

    # disk area ~ pi r^2  -> r ~ sqrt(min_area/pi)
    r = max(1, int(np.sqrt(float(min_area) / np.pi)))
    se = np.ones((2*r+1, 2*r+1), dtype=bool)

    # Remove small bright objects
    mask = binary_opening(mask, structure=se)

    # Fill small holes by the same area scale on the inverse
    inv = ~mask
    inv = binary_opening(inv, structure=se)
    mask = ~inv

    return mask

def _post_clean(mask: np.ndarray,
                open_size: int = 5,
                close_size: int = 7,
                min_area: int = 200) -> np.ndarray:
    """
    Post-process mask with morphological operations and area filtering.

    Args:
        mask (np.ndarray): Binary mask to clean.
        open_size (int): Size of opening structuring element.
        close_size (int): Size of closing structuring element.
        min_area (int): Minimum area threshold for component retention.

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    if open_size > 0:
        mask = binary_opening(mask, structure=np.ones((open_size, open_size)))
    if close_size > 0:
        mask = binary_closing(mask, structure=np.ones((close_size, close_size)))

    mask = binary_fill_holes(mask)

    # Area opening for large component
    mask = _morphological_area_opening(mask, min_area=min_area)

    return mask


def detect_gaps_improved(signal: np.ndarray, 
                         min_gap_size: int = 5,
                         min_gap_ratio: float = 0.015) -> List[Tuple[int, int]]:
    """
    Improved gap detection with adaptive thresholds.
    
    Args:
        signal: 1D binary signal (0=background, 1=foreground)
        min_gap_size: Minimum gap size in pixels
        min_gap_ratio: Minimum gap size as ratio of total signal length
    
    Returns:
        List of (gap_start, gap_end) tuples sorted by gap size
    """
    # Adaptive minimum gap size - more lenient
    adaptive_min_gap = max(min_gap_size, int(len(signal) * min_gap_ratio))
    
    # Find transitions
    diff = np.diff(np.concatenate([[0], signal, [0]]))
    starts = np.where(diff == 1)[0]  # Foreground starts
    ends = np.where(diff == -1)[0]    # Foreground ends
    
    if len(starts) == 0 or len(ends) == 0:
        return []
    
    # Find gaps between foreground regions
    gaps = []
    for i in range(len(ends) - 1):
        gap_start = ends[i]
        gap_end = starts[i + 1]
        gap_size = gap_end - gap_start
        
        if gap_size >= adaptive_min_gap:
            gaps.append((gap_start, gap_end, gap_size))
    
    # Sort by gap size (largest first)
    gaps.sort(key=lambda x: x[2], reverse=True)
    
    return [(g[0], g[1]) for g in gaps]


def analyze_gap_quality(image: np.ndarray, 
                        mask: np.ndarray,
                        gap_start: int, 
                        gap_end: int,
                        axis: int = 1) -> float:
    """
    Analyze the quality of a detected gap to determine if it's a real separation.
    
    Args:
        image: Original image
        mask: Segmentation mask
        gap_start: Gap start position
        gap_end: Gap end position
        axis: 0 for vertical gap (columns), 1 for horizontal gap (rows)
    
    Returns:
        Quality score 0-1, where higher means more likely a real gap
    """
    H, W = mask.shape
    gap_size = gap_end - gap_start
    
    # Extract gap region
    if axis == 1:  # Vertical gap (side-by-side paintings)
        gap_region = mask[:, gap_start:gap_end]
        gap_image = image[:, gap_start:gap_end]
    else:  # Horizontal gap (stacked paintings)
        gap_region = mask[gap_start:gap_end, :]
        gap_image = image[gap_start:gap_end, :]
    
    if gap_region.size == 0:
        return 0.0
    
    # Check 1: How much background is in the gap? (most important)
    fg_pixels = np.sum(gap_region)
    total_pixels = gap_region.size
    bg_ratio = 1.0 - (fg_pixels / (total_pixels + 1e-6))
    
    # If gap is mostly foreground, it's probably not a real gap
    if bg_ratio < 0.3:
        return 0.1
    
    # Check 2: Gap size relative to image dimension
    if axis == 1:
        size_ratio = gap_size / W
    else:
        size_ratio = gap_size / H
    
    # Larger gaps are more likely real separations
    size_score = min(1.0, size_ratio / 0.05)  # Normalize around 5% of image
    
    # Check 3: Color consistency in gap
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    stats = _estimate_bg_from_borders(lab, border=20)
    bg_median = stats["median"]
    
    # Get gap colors
    if axis == 1:
        gap_colors = lab[:, gap_start:gap_end].reshape(-1, 3)
    else:
        gap_colors = lab[gap_start:gap_end, :].reshape(-1, 3)
    
    if len(gap_colors) > 10:
        gap_color_median = np.median(gap_colors, axis=0)
        color_distance = np.linalg.norm(gap_color_median - bg_median)
        # If gap color is similar to background, higher score
        color_score = max(0.0, 1.0 - color_distance / 30.0)
    else:
        color_score = 0.5
    
    # Combine scores with weights
    # Background ratio is most important (70%), then gap size (20%), then color (10%)
    quality_score = 0.7 * bg_ratio + 0.2 * size_score + 0.1 * color_score
    
    return quality_score


def detect_multiple_paintings_smart(image: np.ndarray,
                                    mask: np.ndarray,
                                    min_gap_size: int = 5,
                                    min_gap_ratio: float = 0.015,
                                    quality_threshold: float = 0.45,
                                    debug: bool = False) -> Tuple[int, Optional[int], Optional[str]]:
    """
    Detect if there are multiple paintings and where to split.
    
    Returns:
        (num_paintings, split_position, split_type)
        - num_paintings: 1 or 2
        - split_position: column or row index to split at (None if single painting)
        - split_type: 'vertical' or 'horizontal' (None if single painting)
    """
    H, W = mask.shape
    
    # Project mask to signals
    vertical_signal = np.sum(mask, axis=0)  # Sum rows -> column signal
    vertical_signal = (vertical_signal > 0).astype(np.int32)
    
    horizontal_signal = np.sum(mask, axis=1)  # Sum columns -> row signal
    horizontal_signal = (horizontal_signal > 0).astype(np.int32)
    
    # Detect gaps in both directions
    vertical_gaps = detect_gaps_improved(vertical_signal, min_gap_size, min_gap_ratio)
    horizontal_gaps = detect_gaps_improved(horizontal_signal, min_gap_size, min_gap_ratio)
    
    if debug:
        print(f"  Found {len(vertical_gaps)} vertical gaps, {len(horizontal_gaps)} horizontal gaps")
    
    # Analyze gap quality for vertical gaps (side-by-side paintings)
    best_vertical_gap = None
    best_vertical_quality = 0.0
    
    for gap_start, gap_end in vertical_gaps[:5]:  # Check top 5 gaps
        quality = analyze_gap_quality(image, mask, gap_start, gap_end, axis=1)
        if debug and quality > 0.3:
            print(f"  Vertical gap [{gap_start}:{gap_end}] quality: {quality:.3f}")
        if quality > best_vertical_quality:
            best_vertical_quality = quality
            best_vertical_gap = (gap_start, gap_end)
    
    # Analyze gap quality for horizontal gaps (stacked paintings)
    best_horizontal_gap = None
    best_horizontal_quality = 0.0
    
    for gap_start, gap_end in horizontal_gaps[:5]:  # Check top 5 gaps
        quality = analyze_gap_quality(image, mask, gap_start, gap_end, axis=0)
        if debug and quality > 0.3:
            print(f"  Horizontal gap [{gap_start}:{gap_end}] quality: {quality:.3f}")
        if quality > best_horizontal_quality:
            best_horizontal_quality = quality
            best_horizontal_gap = (gap_start, gap_end)
    
    if debug:
        print(f"  Best vertical: {best_vertical_quality:.3f}, Best horizontal: {best_horizontal_quality:.3f}")
        print(f"  Threshold: {quality_threshold}")
    
    # Decision: choose the best gap if quality exceeds threshold
    # Prefer vertical split (side-by-side) if both are similar
    if best_vertical_quality >= quality_threshold and best_vertical_quality >= best_horizontal_quality - 0.05:
        # Vertical split (side-by-side)
        split_pos = (best_vertical_gap[0] + best_vertical_gap[1]) // 2
        if debug:
            print(f"  → VERTICAL SPLIT at column {split_pos}")
        return 2, split_pos, 'vertical'
    
    elif best_horizontal_quality >= quality_threshold:
        # Horizontal split (stacked)
        split_pos = (best_horizontal_gap[0] + best_horizontal_gap[1]) // 2
        if debug:
            print(f"  → HORIZONTAL SPLIT at row {split_pos}")
        return 2, split_pos, 'horizontal'
    
    else:
        # Single painting
        if debug:
            print(f"  → SINGLE PAINTING (quality below threshold)")
        return 1, None, None


def _project_mask_to_signal(mask: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Project a 2D mask to a 1D signal by summing along the specified axis.

    Args:
        mask (np.ndarray): Binary mask (H, W).
        axis (int): Axis to sum along. 0 for column-wise (horizontal split), 1 for row-wise (vertical split).

    Returns:
        np.ndarray: 1D signal representing the projection.
    """
    # Sum along the axis and normalize to binary (0 or 1)
    signal = np.sum(mask, axis=axis)
    # Convert to binary: 0 if all zeros, 1 if at least some foreground
    signal = (signal > 0).astype(np.int32)
    return signal


def _project_mask_to_signal_thick(mask: np.ndarray, row_thickness: int = 50) -> np.ndarray:
    """
    Project mask to 1D signal using thick row detection (improved method).

    For each column, if ANY pixel in that column has foreground, signal=1, else signal=0.
    This provides more robust gap detection than single-pixel rows.

    Args:
        mask (np.ndarray): Binary mask (H, W).
        row_thickness (int): Row thickness parameter (informational, actual check is per-column).

    Returns:
        np.ndarray: 1D signal array of length W.
    """
    H, W = mask.shape
    signal = []

    # Process column by column
    for col in range(W):
        # Check if any pixel in this column (across all rows) has foreground
        if np.any(mask[:, col] > 0):
            signal.append(1)
        else:
            signal.append(0)

    return np.array(signal)


def _detect_gap_pattern(signal: np.ndarray, min_gap_size: int = 10) -> tuple:
    """
    Detect if there are multiple paintings by analyzing the 1D signal pattern.

    Looks for patterns like:
    - Single painting: 0-1-0 or 0-0-0 or 1-0 or 0-1
    - Two paintings: 0-1-0-1-0 or 1-0-1 or 1-0-1-0 or 0-1-0-1

    Args:
        signal (np.ndarray): 1D binary signal.
        min_gap_size (int): Minimum size of gap (in pixels) to consider as separation.

    Returns:
        tuple: (num_paintings, gap_start, gap_end) where gap_start and gap_end define the gap position.
               Returns (1, -1, -1) if single painting detected.
    """
    # Find transitions from 0 to 1 (painting starts) and 1 to 0 (painting ends)
    diff = np.diff(np.concatenate([[0], signal, [0]]))
    starts = np.where(diff == 1)[0]  # Indices where painting starts
    ends = np.where(diff == -1)[0]   # Indices where painting ends

    num_regions = len(starts)

    if num_regions <= 1:
        # Single painting or no painting
        return (1, -1, -1)

    # Find gaps between regions
    gaps = []
    for i in range(num_regions - 1):
        gap_start = ends[i]
        gap_end = starts[i + 1]
        gap_size = gap_end - gap_start
        if gap_size >= min_gap_size:
            gaps.append((gap_start, gap_end, gap_size))

    if len(gaps) == 0:
        # No significant gap found
        return (1, -1, -1)

    # Find the largest gap (most likely the gap between two paintings)
    largest_gap = max(gaps, key=lambda x: x[2])
    gap_start, gap_end, _ = largest_gap

    return (2, gap_start, gap_end)


def _detect_vertical_gap(signal: np.ndarray, min_gap_size: int = 50) -> Tuple[int, int, int]:
    """
    Detect vertical gap (background region) in signal for vertical-only splitting.

    This is an improved version focused on detecting side-by-side paintings only.

    Args:
        signal (np.ndarray): 1D binary signal.
        min_gap_size (int): Minimum gap size to consider as split (pixels).

    Returns:
        Tuple[int, int, int]: (num_regions, gap_start, gap_end)
            - num_regions: 1 for single painting, 2 for two paintings
            - gap_start, gap_end: Gap boundaries (-1, -1 if single painting)
    """
    # Find transitions from 0 to 1 (region starts) and 1 to 0 (region ends)
    diff = np.diff(np.concatenate([[0], signal, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    num_regions = len(starts)

    if num_regions <= 1:
        # Single painting or no painting
        return (1, -1, -1)

    # Find gaps between regions
    gaps = []
    for i in range(num_regions - 1):
        gap_start = ends[i]
        gap_end = starts[i + 1]
        gap_size = gap_end - gap_start
        if gap_size >= min_gap_size:
            gaps.append((gap_start, gap_end, gap_size))

    if len(gaps) == 0:
        # No significant gap found
        return (1, -1, -1)

    # Return largest gap (most likely separation between paintings)
    largest_gap = max(gaps, key=lambda x: x[2])
    return (2, largest_gap[0], largest_gap[1])


def _find_split_position(gap_start: int, gap_end: int) -> int:
    """
    Find the split position at the middle of the gap.

    Args:
        gap_start (int): Start index of the gap.
        gap_end (int): End index of the gap.

    Returns:
        int: Split position (middle of the gap).
    """
    return (gap_start + gap_end) // 2


def crop_to_mask_bounds(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop image and mask to the bounding box of the foreground.

    This function finds the tight bounding box around the foreground pixels
    and crops both the image and mask to this region.

    Args:
        image (np.ndarray): Image to crop (H, W, C).
        mask (np.ndarray): Binary mask (H, W).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cropped (image, mask) tuple.
    """
    # Find bounding box of foreground
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No foreground, return original
        return image, mask

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box
    cropped_image = image[rmin:rmax+1, cmin:cmax+1]
    cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]

    return cropped_image, cropped_mask


def visualize_signal_analysis(image: np.ndarray,
                              initial_mask: np.ndarray,
                              min_gap_size: int = 10,
                              save_path: str = None) -> plt.Figure:
    """
    Visualize the signal analysis for detecting multiple paintings.

    Args:
        image (np.ndarray): Original image in BGR format.
        initial_mask (np.ndarray): Initial segmentation mask.
        min_gap_size (int): Minimum gap size for detection.
        save_path (str): Path to save the figure (optional).

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    # Project to signals
    horizontal_signal = _project_mask_to_signal(initial_mask, axis=0)
    vertical_signal = _project_mask_to_signal(initial_mask, axis=1)

    # Detect gaps
    num_h, gap_h_start, gap_h_end = _detect_gap_pattern(horizontal_signal, min_gap_size)
    num_v, gap_v_start, gap_v_end = _detect_gap_pattern(vertical_signal, min_gap_size)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # Initial mask
    axes[0, 1].imshow(initial_mask, cmap='gray')
    axes[0, 1].set_title("Initial Segmentation Mask")
    axes[0, 1].axis('off')

    # Horizontal signal (for detecting side-by-side paintings)
    axes[1, 0].plot(horizontal_signal, linewidth=2, color='blue')
    axes[1, 0].set_title(f"Horizontal Signal - Paintings: {num_h}")
    axes[1, 0].set_xlabel("Column Index")
    axes[1, 0].set_ylabel("Signal Value (0 or 1)")
    axes[1, 0].set_ylim([-0.1, 1.5])
    axes[1, 0].grid(True, alpha=0.3)
    if num_h == 2:
        axes[1, 0].axvline(gap_h_start, color='r', linestyle='--', label='Gap Start', linewidth=2)
        axes[1, 0].axvline(gap_h_end, color='g', linestyle='--', label='Gap End', linewidth=2)
        split_pos = _find_split_position(gap_h_start, gap_h_end)
        axes[1, 0].axvline(split_pos, color='purple', linestyle='-', linewidth=3, label=f'Split @ {split_pos}')
        axes[1, 0].legend()

    # Vertical signal (for detecting stacked paintings)
    axes[1, 1].plot(vertical_signal, linewidth=2, color='green')
    axes[1, 1].set_title(f"Vertical Signal - Paintings: {num_v}")
    axes[1, 1].set_xlabel("Row Index")
    axes[1, 1].set_ylabel("Signal Value (0 or 1)")
    axes[1, 1].set_ylim([-0.1, 1.5])
    axes[1, 1].grid(True, alpha=0.3)
    if num_v == 2:
        axes[1, 1].axvline(gap_v_start, color='r', linestyle='--', label='Gap Start', linewidth=2)
        axes[1, 1].axvline(gap_v_end, color='g', linestyle='--', label='Gap End', linewidth=2)
        split_pos = _find_split_position(gap_v_start, gap_v_end)
        axes[1, 1].axvline(split_pos, color='purple', linestyle='-', linewidth=3, label=f'Split @ {split_pos}')
        axes[1, 1].legend()

    plt.suptitle("Signal Analysis for Multi-Painting Detection", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def visualize_split_comparison(image: np.ndarray,
                                initial_mask: np.ndarray,
                                final_mask: np.ndarray,
                                split_info: dict,
                                save_path: str = None) -> plt.Figure:
    """
    Visualize comparison between initial segmentation and split-then-segment result.

    Args:
        image (np.ndarray): Original image in BGR format.
        initial_mask (np.ndarray): Segmentation before splitting.
        final_mask (np.ndarray): Segmentation after split-segment-merge.
        split_info (dict): Information about the split (type, position, etc.).
        save_path (str): Path to save the figure (optional).

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create segmented images
    initial_segmented = image_rgb.copy()
    initial_segmented[initial_mask == 0] = 0

    final_segmented = image_rgb.copy()
    final_segmented[final_mask == 0] = 0

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Initial (before split)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(initial_mask, cmap='gray')
    axes[0, 1].set_title("Initial Mask (Before Split)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(initial_segmented)
    axes[0, 2].set_title("Initial Segmentation", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Final (after split-segment-merge)
    axes[1, 0].imshow(image_rgb)

    # Draw split line on the image
    if split_info['type'] == 'horizontal':
        split_pos = split_info['position']
        axes[1, 0].axvline(split_pos, color='red', linestyle='--', linewidth=3, label=f'Split at col {split_pos}')
        axes[1, 0].legend(loc='upper right')
    elif split_info['type'] == 'vertical':
        split_pos = split_info['position']
        axes[1, 0].axhline(split_pos, color='red', linestyle='--', linewidth=3, label=f'Split at row {split_pos}')
        axes[1, 0].legend(loc='upper right')

    split_type_str = split_info['type'].capitalize() if split_info['type'] != 'none' else 'No Split'
    axes[1, 0].set_title(f"Split Strategy: {split_type_str}", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(final_mask, cmap='gray')
    axes[1, 1].set_title("Final Mask (After Split-Segment-Merge)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(final_segmented)
    axes[1, 2].set_title("Final Segmentation", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle("Comparison: Before Split vs After Split-Segment-Merge", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def detect_and_split_paintings(image: np.ndarray,
                                mask: np.ndarray,
                                row_thickness: int = 50,
                                min_gap_size: int = 50) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect if image has multiple paintings and split vertically if needed (improved method).

    This is a simplified version that only performs vertical splitting (side-by-side paintings)
    using the improved thick row signal detection method.

    Args:
        image (np.ndarray): Original image.
        mask (np.ndarray): Segmentation mask.
        row_thickness (int): Row thickness for signal detection (for robust gap detection).
        min_gap_size (int): Minimum gap size to consider as split (pixels).

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of (image_crop, mask_crop) tuples.
    """
    # Project mask to signal using thick row method
    signal = _project_mask_to_signal_thick(mask, row_thickness)

    # Detect vertical gap
    num_paintings, gap_start, gap_end = _detect_vertical_gap(signal, min_gap_size)

    if num_paintings == 1:
        # Single painting - return whole image and mask
        return [(image, mask)]

    # Two paintings - split at gap center
    split_col = (gap_start + gap_end) // 2

    left_image = image[:, :split_col]
    right_image = image[:, split_col:]

    left_mask = mask[:, :split_col]
    right_mask = mask[:, split_col:]

    return [(left_image, left_mask), (right_image, right_mask)]


def segment_multiple_paintings(image: np.ndarray,
                                border: int = 20,
                                dist_percentile: float = 92.0,
                                dist_margin: float = 0.5,
                                min_area: int = 200,
                                opening_size: int = 5,
                                closing_size: int = 7,
                                wL: float = 0.6, wA: float = 1.0, wB: float = 1.0,
                                sat_min: float = 30.0,
                                hue_percentile: float = 92.0,
                                hue_margin_deg: float = 6.0,
                                min_gap_size: int = 5,
                                min_gap_ratio: float = 0.015,
                                quality_threshold: float = 0.45,
                                return_debug_info: bool = False,
                                debug: bool = False):
    """
    Segment multiple paintings in an image using smart gap detection.
    
    Returns list of masks - one for single painting, multiple for multiple paintings.
    """
    H, W = image.shape[:2]
    
    # Step 1: Initial segmentation using segment_background
    initial_mask = segment_background(
        image, border=border, dist_percentile=dist_percentile,
        dist_margin=dist_margin, wL=wL, wA=wA, wB=wB,
        sat_min=sat_min, hue_percentile=hue_percentile,
        hue_margin_deg=hue_margin_deg,
        min_area=min_area, opening_size=opening_size, closing_size=closing_size
    )
    
    # Step 2: Detect multiple paintings with smart gap quality analysis
    num_paintings, split_pos, split_type = detect_multiple_paintings_smart(
        image, initial_mask, min_gap_size, min_gap_ratio, quality_threshold, debug=debug
    )
    
    # Initialize debug info
    debug_info = {
        'initial_mask': initial_mask,
        'num_paintings': num_paintings,
        'split_type': split_type if split_type else 'none',
        'split_position': split_pos
    }
    
    if num_paintings == 1:
        # Single painting
        if return_debug_info:
            return [initial_mask], debug_info
        return [initial_mask]
    
    # Step 3: Split and re-segment each part
    if split_type == 'vertical':
        # Side-by-side split
        left_image = image[:, :split_pos]
        right_image = image[:, split_pos:]
        
        # Segment each part independently
        left_mask = segment_background(
            left_image, border=border, dist_percentile=dist_percentile,
            dist_margin=dist_margin, wL=wL, wA=wA, wB=wB,
            sat_min=sat_min, hue_percentile=hue_percentile,
            hue_margin_deg=hue_margin_deg,
            min_area=min_area // 2, opening_size=opening_size, closing_size=closing_size
        )
        
        right_mask = segment_background(
            right_image, border=border, dist_percentile=dist_percentile,
            dist_margin=dist_margin, wL=wL, wA=wA, wB=wB,
            sat_min=sat_min, hue_percentile=hue_percentile,
            hue_margin_deg=hue_margin_deg,
            min_area=min_area // 2, opening_size=opening_size, closing_size=closing_size
        )
        
        # Check minimum area requirement (3% of total image)
        total_pixels = H * W
        min_painting_area = total_pixels * 0.03
        
        left_area = np.sum(left_mask)
        right_area = np.sum(right_mask)
        
        if debug:
            print(f"  Left area: {left_area} ({left_area/total_pixels*100:.1f}%), "
                  f"Right area: {right_area} ({right_area/total_pixels*100:.1f}%)")
        
        # Return based on areas
        if left_area < min_painting_area and right_area >= min_painting_area:
            right_full = np.zeros((H, W), dtype=np.uint8)
            right_full[:, split_pos:] = right_mask
            masks = [right_full]
        elif right_area < min_painting_area and left_area >= min_painting_area:
            left_full = np.zeros((H, W), dtype=np.uint8)
            left_full[:, :split_pos] = left_mask
            masks = [left_full]
        elif left_area < min_painting_area and right_area < min_painting_area:
            masks = [initial_mask]
        else:
            # Both paintings are large enough
            left_full = np.zeros((H, W), dtype=np.uint8)
            left_full[:, :split_pos] = left_mask
            right_full = np.zeros((H, W), dtype=np.uint8)
            right_full[:, split_pos:] = right_mask
            masks = [left_full, right_full]
        
        if return_debug_info:
            return masks, debug_info
        return masks
    
    else:  # horizontal split
        # Stacked split
        top_image = image[:split_pos, :]
        bottom_image = image[split_pos:, :]
        
        # Segment each part independently
        top_mask = segment_background(
            top_image, border=border, dist_percentile=dist_percentile,
            dist_margin=dist_margin, wL=wL, wA=wA, wB=wB,
            sat_min=sat_min, hue_percentile=hue_percentile,
            hue_margin_deg=hue_margin_deg,
            min_area=min_area // 2, opening_size=opening_size, closing_size=closing_size
        )
        
        bottom_mask = segment_background(
            bottom_image, border=border, dist_percentile=dist_percentile,
            dist_margin=dist_margin, wL=wL, wA=wA, wB=wB,
            sat_min=sat_min, hue_percentile=hue_percentile,
            hue_margin_deg=hue_margin_deg,
            min_area=min_area // 2, opening_size=opening_size, closing_size=closing_size
        )
        
        # Check minimum area requirement (1% of total image for horizontal)
        total_pixels = H * W
        min_painting_area = total_pixels * 0.015
        
        top_area = np.sum(top_mask)
        bottom_area = np.sum(bottom_mask)
        
        if debug:
            print(f"  Top area: {top_area} ({top_area/total_pixels*100:.1f}%), "
                  f"Bottom area: {bottom_area} ({bottom_area/total_pixels*100:.1f}%)")
        
        # Return based on areas
        if top_area < min_painting_area and bottom_area >= min_painting_area:
            bottom_full = np.zeros((H, W), dtype=np.uint8)
            bottom_full[split_pos:, :] = bottom_mask
            masks = [bottom_full]
        elif bottom_area < min_painting_area and top_area >= min_painting_area:
            top_full = np.zeros((H, W), dtype=np.uint8)
            top_full[:split_pos, :] = top_mask
            masks = [top_full]
        elif top_area < min_painting_area and bottom_area < min_painting_area:
            masks = [initial_mask]
        else:
            # Both paintings are large enough
            top_full = np.zeros((H, W), dtype=np.uint8)
            top_full[:split_pos, :] = top_mask
            bottom_full = np.zeros((H, W), dtype=np.uint8)
            bottom_full[split_pos:, :] = bottom_mask
            masks = [top_full, bottom_full]
        
        if return_debug_info:
            return masks, debug_info
        return masks


def segment_background(image: np.ndarray,
                                border: int = 20,
                                dist_percentile: float = 92.0,
                                dist_margin: float = 0.5,
                                min_area: int = 200,
                                opening_size: int = 5,
                                closing_size: int = 7,
                                wL: float = 0.6, wA: float = 1.0, wB: float = 1.0,
                                sat_min: float = 30.0,
                                hue_percentile: float = 92.0,
                                hue_margin_deg: float = 6.0,
                                apply_gaussian: bool = False,
                                sigma: float = 1.0,
                                radius: int = 2) -> np.ndarray:
    """
    Segment foreground from background using LAB and HSV color-based methods.

    Args:
        image (np.ndarray): Input image in BGR format.
        border (int): Width of border pixels for background estimation.
        dist_percentile (float): Percentile for LAB distance threshold.
        dist_margin (float): Margin added to LAB distance threshold.
        min_area (int): Minimum area for morphological filtering.
        opening_size (int): Size of morphological opening operation.
        closing_size (int): Size of morphological closing operation.
        wL (float): Weight for L channel in LAB distance.
        wA (float): Weight for A channel in LAB distance.
        wB (float): Weight for B channel in LAB distance.
        sat_min (float): Minimum saturation threshold for hue-based segmentation.
        hue_percentile (float): Percentile for hue distance threshold.
        hue_margin_deg (float): Margin in degrees added to hue distance threshold.
        apply_gaussian (bool): Whether to apply Gaussian filtering for denoising.
        sigma (float): Standard deviation for Gaussian kernel.
        radius (int): Radius of Gaussian kernel (truncate parameter).

    Returns:
        np.ndarray: Binary mask where foreground is 1 and background is 0.
    """
    # Apply Gaussian filter for denoising if requested
    if apply_gaussian:
        image = apply_gaussian_filter(image, sigma=sigma, radius=radius)

    lab = convert_img_to_lab(image).astype(np.float32)
    hsv = convert_img_to_hsv(image).astype(np.float32)

    L = lab[..., 0]
    A = lab[..., 1]
    B = lab[..., 2]
    Hh = hsv[..., 0]  # [0,180)
    Ss = hsv[..., 1]  # [0,255]

    # 1) Background model from borders (LAB)
    stats = _estimate_bg_from_borders(lab, border=border)
    med = stats["median"]
    mad = stats["mad"]  # robust scales

    # Border mask (for threshold estimation)
    H, W = L.shape
    b = max(1, min(border, min(H, W)//4))
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:b, :] = True
    border_mask[-b:, :] = True
    border_mask[:, :b] = True
    border_mask[:, -b:] = True

    # 3) Robust weighted LAB distance (downweight L)
    dist_lab = _lab_robust_distance_weighted(L, A, B, med, mad, wL=wL, wA=wA, wB=wB)

    # 4) Threshold from *border* distances (+ margin)
    border_dists = dist_lab[border_mask]
    if border_dists.size == 0:
        thr_lab = np.percentile(dist_lab, dist_percentile) + dist_margin
    else:
        thr_lab = np.percentile(border_dists, dist_percentile) + dist_margin
    mask_lab = dist_lab > thr_lab  # FG by LAB

    # 5) Hue fallback (HSV), purely color-based
    border_hues = Hh[border_mask]
    h0 = np.median(border_hues) if border_hues.size > 0 else np.median(Hh)
    hue_dist = _hue_circular_distance(Hh, h0)  # degrees in [0,90]
    border_hued = hue_dist[border_mask]
    thr_hue = (np.percentile(hue_dist, hue_percentile) + hue_margin_deg
               if border_hued.size == 0
               else np.percentile(border_hued, hue_percentile) + hue_margin_deg)
    mask_hue = (hue_dist > thr_hue) & (Ss >= sat_min)

    # Combine color cues
    mask = mask_lab | mask_hue

    # 6) Morph-only cleanup
    mask = _post_clean(mask, open_size=opening_size, close_size=closing_size, min_area=min_area)

    return mask.astype(np.uint8)


def run_experiments_on_dataset(image_folder: str,
                               experiments_config: List[Dict],
                               max_images: int = None,
                               output_folder: str = "results/method_color_only/",
                               save_outputs: bool = True) -> List[Dict]:
    """
    Run all experiments on the dataset and save outputs.

    Args:
        image_folder (str): Path to folder containing input images.
        experiments_config (List[Dict]): List of experiment configurations with name, func, and params.
        max_images (int): Maximum number of images to process. None processes all images.
        output_folder (str): Path to output folder for results.
        save_outputs (bool): Whether to save output masks, segmented images, and visualizations.

    Returns:
        List[Dict]: List of experiment results with average metrics and per-image results.
    """
    # Get image files
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')])
    if max_images:
        image_files = image_files[:max_images]

    print(f"Running {len(experiments_config)} experiments on {len(image_files)} images...")

    all_results = []

    for exp_idx, exp_config in enumerate(experiments_config):
        print(f"\n[{exp_idx+1}/{len(experiments_config)}] Running: {exp_config['name']}")

        # Create output directories for this experiment
        exp_output_folder = os.path.join(output_folder, exp_config['name'])
        if save_outputs:
            os.makedirs(os.path.join(exp_output_folder, 'masks'), exist_ok=True)
            os.makedirs(os.path.join(exp_output_folder, 'segmented'), exist_ok=True)
            os.makedirs(os.path.join(exp_output_folder, 'visualizations'), exist_ok=True)

        image_results = []

        for img_idx, img_file in enumerate(image_files):
            # Load image using read_image from src.data.extract
            img_path = os.path.join(image_folder, img_file)
            image = read_image(img_path)

            # Load GT mask (same filename but .png)
            gt_file = os.path.splitext(img_file)[0] + '.png'
            gt_path = os.path.join(image_folder, gt_file)

            gt_mask = None
            if os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    gt_mask = (gt_mask > 127).astype(np.uint8)

            # Run experiment
            is_multi_painting = exp_config['func'].__name__ == 'segment_multiple_paintings'

            if is_multi_painting:
                # Returns list of masks
                masks_list = exp_config['func'](image, **exp_config['params'])
                # For evaluation, merge all masks
                pred_mask = np.zeros_like(masks_list[0])
                for m in masks_list:
                    pred_mask = np.logical_or(pred_mask, m).astype(np.uint8)
            else:
                masks_list = None
                pred_mask = exp_config['func'](image, **exp_config['params'])

            # Calculate metrics if GT available
            metrics = None
            if gt_mask is not None:
                tp, fp, tn, fn = compute_confusion_counts_mask(pred_mask, gt_mask)

                precision = precision_mask(tp, fp)
                recall = recall_mask(tp, fn)
                f1 = f1_mask(tp, fp, fn)

                metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn
                }

                result = {
                    'image': img_file,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn
                }
                image_results.append(result)

            # Create segmented output
            segmented = apply_mask_to_image_mask(image, pred_mask)

            if save_outputs:
                base_name = os.path.splitext(img_file)[0]
                
                # Save combined mask
                mask_filename = base_name + '.png'
                mask_path = os.path.join(exp_output_folder, 'masks', mask_filename)
                cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))

                # Save combined segmented output
                seg_path = os.path.join(exp_output_folder, 'segmented', img_file)
                cv2.imwrite(seg_path, segmented)
                
                # If multiple paintings detected, save separate masks and segmented images
                if is_multi_painting and masks_list is not None and len(masks_list) > 1:
                    for idx, individual_mask in enumerate(masks_list):
                        # Save individual mask as <id>_<N>.png
                        individual_mask_filename = f"{base_name}_{idx}.png"
                        individual_mask_path = os.path.join(exp_output_folder, 'masks', individual_mask_filename)
                        cv2.imwrite(individual_mask_path, (individual_mask * 255).astype(np.uint8))
                        
                        # Save individual segmented painting (no background)
                        individual_segmented = apply_mask_to_image_mask(image, individual_mask)
                        individual_seg_filename = f"{base_name}_{idx}.jpg"
                        individual_seg_path = os.path.join(exp_output_folder, 'segmented', individual_seg_filename)
                        cv2.imwrite(individual_seg_path, individual_segmented)

                # Create and save visualization
                viz_filename = os.path.splitext(img_file)[0] + '_viz.png'
                viz_path = os.path.join(exp_output_folder, 'visualizations', viz_filename)

                title = f"{exp_config['name']} - {img_file}"
                if metrics:
                    title += f"\nP={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"

                # Convert BGR to RGB for visualization
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)

                fig = create_visualization_mask(
                    original=image_rgb,
                    mask=pred_mask,
                    segmented=segmented_rgb,
                    gt_mask=gt_mask,
                    metrics=metrics,
                    title=title
                )
                fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            print(f"  [{img_idx+1}/{len(image_files)}] Processed {img_file}", end='\r')

        print()  # New line after progress

        # Calculate average metrics
        if image_results:
            avg_precision = np.mean([r['precision'] for r in image_results])
            avg_recall = np.mean([r['recall'] for r in image_results])
            avg_f1 = 2*((avg_precision*avg_recall)/(avg_precision+avg_recall))

            exp_summary = {
                'experiment_name': exp_config['name'],
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1,
                'image_results': image_results,
                'config': exp_config
            }

            all_results.append(exp_summary)

            print(f"  Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")

            if save_outputs:
                print(f"  Outputs saved to: {exp_output_folder}")

    return all_results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Color-only background removal and segmentation with smart multi-painting detection"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="data/raw/qsd2_w3/",
        help="Path to folder containing input images (default: data/raw/qsd2_w3/)"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results/seg/",
        help="Path to folder for output results (default: results/seg/)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: None, process all)"
    )
    parser.add_argument(
        "--multi_painting",
        action="store_true",
        help="Enable multi-painting segmentation mode (detects and segments multiple paintings)"
    )
    parser.add_argument(
        "--min_gap_size",
        type=int,
        default=5,
        help="Minimum gap size (in pixels) between paintings for multi-painting mode (default: 5)"
    )
    parser.add_argument(
        "--min_gap_ratio",
        type=float,
        default=0.015,
        help="Minimum gap size as ratio of image dimension (default: 0.015 = 1.5%%)"
    )
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.45,
        help="Quality threshold for gap detection (0-1, default: 0.45)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    args = parser.parse_args()

    print(f"Image folder: {args.image_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Multi-painting mode: {args.multi_painting}")
    if args.multi_painting:
        print(f"Min gap size: {args.min_gap_size}")
        print(f"Min gap ratio: {args.min_gap_ratio}")
        print(f"Quality threshold: {args.quality_threshold}")

    # Configuration
    IMAGE_FOLDER = args.image_folder
    OUTPUT_FOLDER = args.output_folder
    MAX_IMAGES = args.max_images

    # Experiment configuration
    if args.multi_painting:
        experiments = [{
            'name': 'multi_painting_smart',
            'func': segment_multiple_paintings,
            'params': {
                'border': 20,
                'dist_percentile': 92.0,
                'dist_margin': 0.5,
                'min_area': 200,
                'opening_size': 5,
                'closing_size': 7,
                'wL': 0.6, 'wA': 1.0, 'wB': 1.0,
                'sat_min': 30.0,
                'hue_percentile': 92.0,
                'hue_margin_deg': 6.0,
                'min_gap_size': args.min_gap_size,
                'min_gap_ratio': args.min_gap_ratio,
                'quality_threshold': args.quality_threshold,
                'debug': args.debug
            }
        }]
    else:
        experiments = [{
            'name': 'single_painting',
            'func': segment_background,
            'params': {
                'border': 20,
                'dist_percentile': 92.0,
                'dist_margin': 0.5,
                'min_area': 200,
                'opening_size': 5,
                'closing_size': 7,
                'wL': 0.6, 'wA': 1.0, 'wB': 1.0,
                'sat_min': 30.0,
                'hue_percentile': 92.0,
                'hue_margin_deg': 6.0
            }
        }]

    # Run experiments with full output saving
    results = run_experiments_on_dataset(
        image_folder=IMAGE_FOLDER,
        experiments_config=experiments,
        max_images=MAX_IMAGES,
        output_folder=OUTPUT_FOLDER,
        save_outputs=True
    )

    # Save results and plots
    save_results_mask(results, OUTPUT_FOLDER)

    # Print final results
    if results:
        result = results[0]
        print(f"\nOverall Result: {result['experiment_name']}")
        print(f"  Precision: {result['avg_precision']:.4f}")
        print(f"  Recall:    {result['avg_recall']:.4f}")
        print(f"  F1 Score:  {result['avg_f1']:.4f}")
        print(f"\nOutputs saved to: {OUTPUT_FOLDER}")
        print(f"  - Masks: {OUTPUT_FOLDER}/masks/")
        print(f"  - Segmented: {OUTPUT_FOLDER}/segmented/")
        print(f"  - Visualizations: {OUTPUT_FOLDER}/visualizations/")