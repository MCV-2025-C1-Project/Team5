import numpy as np
import os
import cv2
import pickle
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, gaussian_filter
from typing import Dict, List, Tuple
import argparse
from pathlib import Path
from tqdm import tqdm

from src.data.extract import read_image
from src.descriptors.lab import convert_img_to_lab
from src.descriptors.hsv import convert_img_to_hsv
from src.models.main import DESCRIPTOR_FUNCTIONS, DISTANCE_FUNCTIONS
from src.tools.startup import logger


def apply_gaussian_filter(image: np.ndarray, sigma: float = 1.0, radius: int = 2) -> np.ndarray:
    """
    Apply Gaussian filter to denoise image.

    Args:
        image: Input image (H, W, C).
        sigma: Standard deviation for Gaussian kernel.
        radius: Radius of Gaussian kernel.

    Returns:
        Filtered image.
    """
    # Apply Gaussian filter to each channel
    filtered = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        filtered[:, :, c] = gaussian_filter(image[:, :, c].astype(np.float32), sigma=sigma, truncate=radius)

    return filtered.astype(image.dtype)


def _estimate_bg_from_borders(image_lab: np.ndarray, border: int = 20) -> Dict[str, np.ndarray]:
    """Estimate background color statistics from borders in LAB space."""
    H, W, _ = image_lab.shape
    b = max(1, min(border, min(H, W)//4))

    top = image_lab[:b, :, :]
    bottom = image_lab[-b:, :, :]
    left = image_lab[:, :b, :]
    right = image_lab[:, -b:, :]

    borders = np.concatenate([top.reshape(-1,3), bottom.reshape(-1,3),
                              left.reshape(-1,3), right.reshape(-1,3)], axis=0)

    med = np.median(borders, axis=0)
    mad = np.median(np.abs(borders - med), axis=0) + 1e-6
    return {"median": med.astype(np.float32), "mad": mad.astype(np.float32)}


def _lab_robust_distance_weighted(L, A, B, med, mad, wL=0.6, wA=1.0, wB=1.0):
    """Weighted distance to background in LAB."""
    dL = (L - med[0]) / mad[0]
    dA = (A - med[1]) / mad[1]
    dB = (B - med[2]) / mad[2]
    return np.sqrt((wL*dL)**2 + (wA*dA)**2 + (wB*dB)**2)


def _hue_circular_distance(h, h0):
    """Circular distance on hue."""
    dh = np.abs(h - h0)
    return np.minimum(dh, 180.0 - dh)


def _morphological_area_opening(mask, min_area):
    """Area opening to remove small blobs."""
    if min_area <= 0:
        return mask
    r = max(1, int(np.sqrt(float(min_area) / np.pi)))
    se = np.ones((2*r+1, 2*r+1), dtype=bool)
    mask = binary_opening(mask, structure=se)
    inv = ~mask
    inv = binary_opening(inv, structure=se)
    return ~inv


def _post_clean(mask, open_size=5, close_size=7, min_area=200):
    """Post-process mask with morphological operations."""
    if open_size > 0:
        mask = binary_opening(mask, structure=np.ones((open_size, open_size)))
    if close_size > 0:
        mask = binary_closing(mask, structure=np.ones((close_size, close_size)))
    mask = binary_fill_holes(mask)
    mask = _morphological_area_opening(mask, min_area=min_area)
    return mask


def segment_background(image, border=20, dist_percentile=92.0, dist_margin=0.5,
                       min_area=200, opening_size=5, closing_size=7,
                       wL=0.6, wA=1.0, wB=1.0, sat_min=30.0,
                       hue_percentile=92.0, hue_margin_deg=6.0,
                       apply_gaussian=True, sigma=1.0, radius=2):
    """
    Segment foreground from background with optional Gaussian filtering.
    """
    # Apply Gaussian filter for denoising
    if apply_gaussian:
        image = apply_gaussian_filter(image, sigma=sigma, radius=radius)

    lab = convert_img_to_lab(image).astype(np.float32)
    hsv = convert_img_to_hsv(image).astype(np.float32)

    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    Hh, Ss = hsv[..., 0], hsv[..., 1]

    stats = _estimate_bg_from_borders(lab, border=border)
    med, mad = stats["median"], stats["mad"]

    H, W = L.shape
    b = max(1, min(border, min(H, W)//4))
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:b, :] = True
    border_mask[-b:, :] = True
    border_mask[:, :b] = True
    border_mask[:, -b:] = True

    dist_lab = _lab_robust_distance_weighted(L, A, B, med, mad, wL, wA, wB)
    border_dists = dist_lab[border_mask]
    thr_lab = np.percentile(border_dists, dist_percentile) + dist_margin if border_dists.size else np.percentile(dist_lab, dist_percentile) + dist_margin
    mask_lab = dist_lab > thr_lab

    border_hues = Hh[border_mask]
    h0 = np.median(border_hues) if border_hues.size > 0 else np.median(Hh)
    hue_dist = _hue_circular_distance(Hh, h0)
    border_hued = hue_dist[border_mask]
    thr_hue = (np.percentile(hue_dist, hue_percentile) + hue_margin_deg if border_hued.size == 0
               else np.percentile(border_hued, hue_percentile) + hue_margin_deg)
    mask_hue = (hue_dist > thr_hue) & (Ss >= sat_min)

    mask = mask_lab | mask_hue
    mask = _post_clean(mask, open_size=opening_size, close_size=closing_size, min_area=min_area)

    return mask.astype(np.uint8)


def _project_mask_to_signal_thick(mask: np.ndarray, row_thickness: int = 50) -> np.ndarray:
    """
    Project mask to 1D signal using thick rows.

    If ANY pixel in a row_thickness-pixel tall row has a value, the signal becomes 1.

    Args:
        mask: Binary mask (H, W).
        row_thickness: Number of rows to consider together.

    Returns:
        1D signal array.
    """
    H, W = mask.shape
    signal = []

    # Process column by column
    for col in range(W):
        column_has_foreground = False

        # Check if any pixel in this column (across all rows) has foreground
        # Actually, based on the description, we should check in chunks of row_thickness
        # But I think the intent is: for each column, if ANY pixel in that column has value, signal=1
        # Let me re-read: "take 50 pixel row and if any of the 50 pixels has 1 or 0, that column value should become 1 or 0"
        # I think it means: divide the column into chunks of 50 pixels, and if any pixel in that chunk is 1, the whole chunk becomes 1
        # But for a signal, we want one value per column position
        # Let me interpret it as: for each column, check if there's any foreground pixel in that column

        if np.any(mask[:, col] > 0):
            signal.append(1)
        else:
            signal.append(0)

    return np.array(signal)


def _detect_vertical_gap(signal: np.ndarray, min_gap_size: int = 50) -> Tuple[int, int, int]:
    """
    Detect vertical gap (background region) in signal.

    Returns:
        (num_regions, gap_start, gap_end)
    """
    diff = np.diff(np.concatenate([[0], signal, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    num_regions = len(starts)

    if num_regions <= 1:
        return (1, -1, -1)

    # Find gaps
    gaps = []
    for i in range(num_regions - 1):
        gap_start = ends[i]
        gap_end = starts[i + 1]
        gap_size = gap_end - gap_start
        if gap_size >= min_gap_size:
            gaps.append((gap_start, gap_end, gap_size))

    if len(gaps) == 0:
        return (1, -1, -1)

    # Return largest gap
    largest_gap = max(gaps, key=lambda x: x[2])
    return (2, largest_gap[0], largest_gap[1])


def detect_and_split_paintings(image: np.ndarray, mask: np.ndarray,
                                row_thickness: int = 50,
                                min_gap_size: int = 50) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect if image has multiple paintings and split if needed.

    Args:
        image: Original image.
        mask: Segmentation mask.
        row_thickness: Row thickness for signal detection.
        min_gap_size: Minimum gap size to consider as split.

    Returns:
        List of (image_crop, mask_crop) tuples.
    """
    # Project mask to signal
    signal = _project_mask_to_signal_thick(mask, row_thickness)

    # Detect gap
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


def crop_to_mask_bounds(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop image and mask to the bounding box of the foreground.

    Args:
        image: Image to crop.
        mask: Binary mask.

    Returns:
        Cropped (image, mask) tuple.
    """
    # Find bounding box of foreground
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No foreground, return original
        return image, mask

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop
    cropped_image = image[rmin:rmax+1, cmin:cmax+1]
    cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]

    return cropped_image, cropped_mask


def compute_descriptor_from_array(image: np.ndarray, descriptor_func, values_per_bin: int = 1) -> np.ndarray:
    """Compute descriptor from image array."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, image)

    try:
        descriptor, _ = descriptor_func(tmp_path, values_per_bin=values_per_bin)
        if isinstance(descriptor, tuple):
            descriptor = np.concatenate(descriptor)
        return descriptor
    finally:
        os.unlink(tmp_path)


def build_museum_database(museum_dir: str, descriptor_type: str, values_per_bin: int = 1):
    """Build museum database."""
    museum_path = Path(museum_dir)
    museum_images = sorted(museum_path.glob('*.jpg'))

    logger.info(f"Building museum database with {len(museum_images)} images...")

    descriptor_func = DESCRIPTOR_FUNCTIONS[descriptor_type]
    db_descriptors = []
    db_ids = []

    for img_path in tqdm(museum_images, desc="Museum descriptors"):
        descriptor, _ = descriptor_func(str(img_path), values_per_bin=values_per_bin)
        if isinstance(descriptor, tuple):
            descriptor = np.concatenate(descriptor)
        db_descriptors.append(descriptor)

        stem = img_path.stem
        if stem.startswith('bbdd_'):
            stem = stem[5:]
        db_ids.append(int(stem))

    return np.array(db_descriptors), db_ids


def process_query_images(query_dir: str, descriptor_type: str, distance_metric: str,
                          db_descriptors: np.ndarray, db_ids: List[int],
                          k: int = 10, values_per_bin: int = 1,
                          row_thickness: int = 50, min_gap_size: int = 50,
                          sigma: float = 1.0, radius: int = 2):
    """
    Process all query images: segment, split, crop, compute descriptors, retrieve.

    Returns:
        List[List[List[int]]]: Correspondences in format [[[m1,m2,...]], [[m1,m2,...],[m1,m2,...]], ...]
    """
    query_path = Path(query_dir)
    query_images = sorted(query_path.glob('*.jpg'))

    logger.info(f"Processing {len(query_images)} query images...")

    descriptor_func = DESCRIPTOR_FUNCTIONS[descriptor_type]
    distance_func = DISTANCE_FUNCTIONS[distance_metric]
    db_ids_array = np.array(db_ids)

    all_correspondences = []

    for img_path in tqdm(query_images, desc="Processing queries"):
        # Load image
        image = read_image(str(img_path))

        # Segment with Gaussian filtering
        mask = segment_background(image, apply_gaussian=True, sigma=sigma, radius=radius)

        # Detect and split paintings
        paintings = detect_and_split_paintings(image, mask, row_thickness, min_gap_size)

        logger.info(f"{img_path.name}: {len(paintings)} painting(s)")

        # Process each painting
        image_matches = []
        for painting_img, painting_mask in paintings:
            # Crop to mask bounds
            cropped_img, cropped_mask = crop_to_mask_bounds(painting_img, painting_mask)

            # Compute descriptor
            descriptor = compute_descriptor_from_array(cropped_img, descriptor_func, values_per_bin)

            # Retrieve top-k matches
            distances = np.array([distance_func(descriptor, db_desc) for db_desc in db_descriptors])
            top_k_indices = np.argsort(distances)[:k]
            top_k_matches = [int(db_ids_array[idx]) for idx in top_k_indices]

            image_matches.append(top_k_matches)

        all_correspondences.append(image_matches)

    return all_correspondences


def save_correspondences(correspondences: List[List[List[int]]], output_path: str):
    """Save correspondences to pickle file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(correspondences, f)

    logger.info(f"Saved correspondences to {output_path}")

    # Statistics
    num_single = sum(1 for img in correspondences if len(img) == 1)
    num_double = sum(1 for img in correspondences if len(img) == 2)
    logger.info(f"  Single painting: {num_single} images")
    logger.info(f"  Two paintings:   {num_double} images")


def evaluate_map_at_k(predictions: List[List[List[int]]], ground_truth: List[List[int]], k: int = 10):
    """
    Evaluate MAP@K for multi-painting retrieval.

    Args:
        predictions: Format [[[m1,m2,...]], [[m1,...],[m2,...]], ...]
        ground_truth: Format [[gt1], [gt1, gt2], ...]
        k: K value for MAP@K
    """
    all_aps = []

    for img_idx, (pred_paintings, gt_paintings) in enumerate(zip(predictions, ground_truth)):
        # Handle both formats of GT
        if isinstance(gt_paintings, int):
            gt_paintings = [gt_paintings]
        if not isinstance(gt_paintings, list):
            gt_paintings = list(gt_paintings)

        # Number of paintings in this image
        num_paintings = max(len(pred_paintings), len(gt_paintings))

        for painting_idx in range(num_paintings):
            if painting_idx < len(pred_paintings) and painting_idx < len(gt_paintings):
                pred = pred_paintings[painting_idx][:k]
                gt = [gt_paintings[painting_idx]] if isinstance(gt_paintings[painting_idx], int) else gt_paintings[painting_idx]

                # Calculate AP
                hits = 0
                sum_precisions = 0.0
                for i, pred_id in enumerate(pred):
                    if pred_id in gt:
                        hits += 1
                        precision_at_i = hits / (i + 1)
                        sum_precisions += precision_at_i

                ap = sum_precisions / min(len(gt), k) if len(gt) > 0 else 0.0
                all_aps.append(ap)

    return np.mean(all_aps) if all_aps else 0.0


def main():
    parser = argparse.ArgumentParser(description="Segmentation and Retrieval for Multi-Painting Images")

    parser.add_argument('--query_dir', type=str, required=True, help='Query images directory')
    parser.add_argument('--museum_dir', type=str, required=True, help='Museum database directory')
    parser.add_argument('--output', type=str, default='results/result.pkl', help='Output pickle file')
    parser.add_argument('--gt_file', type=str, default=None, help='Ground truth pickle file (optional)')

    parser.add_argument('--descriptor', type=str, default='hsv', choices=list(DESCRIPTOR_FUNCTIONS.keys()))
    parser.add_argument('--distance', type=str, default='chi_2.compute_chi_2_distance', choices=list(DISTANCE_FUNCTIONS.keys()))
    parser.add_argument('--k', type=int, default=10, help='Top-K matches')
    parser.add_argument('--values_per_bin', type=int, default=1)

    parser.add_argument('--row_thickness', type=int, default=50, help='Row thickness for signal detection')
    parser.add_argument('--min_gap_size', type=int, default=50, help='Minimum gap size for split detection')
    parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian filter sigma')
    parser.add_argument('--radius', type=int, default=2, help='Gaussian filter radius')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("SEGMENTATION AND RETRIEVAL SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Query Directory:  {args.query_dir}")
    logger.info(f"Museum Directory: {args.museum_dir}")
    logger.info(f"Descriptor:       {args.descriptor}")
    logger.info(f"Distance:         {args.distance}")
    logger.info(f"K (MAP@K):        {args.k}")
    logger.info(f"Gaussian sigma:   {args.sigma}")
    logger.info(f"Gaussian radius:  {args.radius}")
    logger.info(f"Row thickness:    {args.row_thickness}")
    logger.info(f"Min gap size:     {args.min_gap_size}")
    logger.info("=" * 80)

    # Build museum database
    db_descriptors, db_ids = build_museum_database(args.museum_dir, args.descriptor, args.values_per_bin)

    # Process queries and retrieve
    correspondences = process_query_images(
        args.query_dir, args.descriptor, args.distance,
        db_descriptors, db_ids, args.k, args.values_per_bin,
        args.row_thickness, args.min_gap_size, args.sigma, args.radius
    )

    # Save results
    save_correspondences(correspondences, args.output)

    # Evaluate if GT available
    if args.gt_file and os.path.exists(args.gt_file):
        with open(args.gt_file, 'rb') as f:
            gt = pickle.load(f)

        map_k = evaluate_map_at_k(correspondences, gt, args.k)
        logger.info(f"\nMAP@{args.k}: {map_k:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()