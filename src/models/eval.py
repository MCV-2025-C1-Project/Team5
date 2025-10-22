from pathlib import Path
from typing import List
import pickle
from tqdm import tqdm

from src.metrics.precision import mapk
from src.models.main import ComputeImageHistogram
from src.tools.startup import logger


# Define all available descriptors and distance metrics
ALL_DESCRIPTORS = [
    'rgb',
    'hsv',
    'ycbcr',
    'lab',
    'grayscale',
    '3d_rgb',
    '3d_hsv',
    '3d_lab',
    '2d_ycbcr',
    '2d_lab',
    '2d_hsv',
    'spatial_pyramid_lab',
    'spatial_pyramid_hsv_lvl2',
    'spatial_pyramid_hsv_lvl3',
    'spatial_pyramid_hsv_lvl4',
    'spatial_pyramid_hsv_lvl5',
    'spatial_pyramid_2d_lab',
    'spatial_pyramid_2d_hsv_lvl2',
    'spatial_pyramid_2d_hsv_lvl3',
    'spatial_pyramid_2d_hsv_lvl4',
    'spatial_pyramid_2d_hsv_lvl5',
    'spatial_pyramid_3d_lab',
    'spatial_pyramid_3d_hsv_lvl2',
    'spatial_pyramid_3d_hsv_lvl3',
    'spatial_pyramid_3d_hsv_lvl4',
    'spatial_pyramid_3d_hsv_lvl5',
    'block_histogram_lab',
    'block_histogram_hsv_2x2',
    'block_histogram_hsv_4x4',
    'block_histogram_hsv_8x8',
    'block_histogram_hsv_16x16',
    'block_histogram_2d_lab',
    'block_histogram_2d_hsv_2x2',
    'block_histogram_2d_hsv_4x4',
    'block_histogram_2d_hsv_8x8',
    'block_histogram_2d_hsv_16x16',
    'block_histogram_3d_lab',
    'block_histogram_3d_hsv_2x2',
    'block_histogram_3d_hsv_4x4',
    'block_histogram_3d_hsv_8x8',
    'block_histogram_3d_hsv_16x16',
    'dct_grayscale_4x4_8coeffs',
    'dct_grayscale_4x4_16coeffs',
    'dct_grayscale_4x4_32coeffs',
    'dct_grayscale_8x8_8coeffs',
    'dct_grayscale_8x8_16coeffs',
    'dct_grayscale_8x8_32coeffs',
    'dct_hsv_4x4_8coeffs',
    'dct_hsv_4x4_16coeffs',
    'dct_hsv_4x4_32coeffs',
    'dct_hsv_8x8_8coeffs',
    'dct_hsv_8x8_16coeffs',
    'dct_hsv_8x8_32coeffs',
    'dct_lab_4x4_8coeffs',
    'dct_lab_4x4_16coeffs',
    'dct_lab_4x4_32coeffs',
    'dct_lab_8x8_8coeffs',
    'dct_lab_8x8_16coeffs',
    'dct_lab_8x8_32coeffs',
    'lbp_gray_s1_4x4',
    'lbp_gray_ms2_4x4',
    'lbp_gray_ms2_8x8',
    'lbp_lab_ms2_4x4',
    'lbp_lab_ms2_8x8',
    'lbp_hsv_ms2_4x4',
    'lbp_hsv_ms2_8x8',
    'haar_grayscale_lvl1',
    'haar_grayscale_lvl2',
    'haar_grayscale_lvl3',
    'haar_grayscale_lvl4',
    'haar_hsv_lvl1',
    'haar_hsv_lvl2',
    'haar_hsv_lvl3',
    'haar_hsv_lvl4',
    'bior44_grayscale_lvl1',
    'bior44_grayscale_lvl2',
    'bior44_grayscale_lvl3',
    'bior44_grayscale_lvl4',
    'bior44_hsv_lvl1',
    'bior44_hsv_lvl2',
    'bior44_hsv_lvl3',
    'bior44_hsv_lvl4',
    'block_haar_grayscale_4x4_lvl1',
    'block_haar_grayscale_8x8_lvl1',
    'block_haar_grayscale_4x4_lvl2',
    'block_haar_grayscale_8x8_lvl2',
    'block_haar_grayscale_4x4_lvl3',
    'block_haar_grayscale_8x8_lvl3',
    'block_haar_hsv_4x4_lvl1',
    'block_haar_hsv_8x8_lvl1',
    'block_haar_hsv_4x4_lvl2',
    'block_haar_hsv_8x8_lvl2',
    'block_haar_hsv_4x4_lvl3',
    'block_haar_hsv_8x8_lvl3',
    'block_bior44_grayscale_4x4_lvl1',
    'block_bior44_grayscale_8x8_lvl1',
    'block_bior44_grayscale_4x4_lvl2',
    'block_bior44_grayscale_8x8_lvl2',
    'block_bior44_grayscale_4x4_lvl3',
    'block_bior44_grayscale_8x8_lvl3',
    'block_bior44_hsv_4x4_lvl1',
    'block_bior44_hsv_8x8_lvl1',
    'block_bior44_hsv_4x4_lvl2',
    'block_bior44_hsv_8x8_lvl2',
    'block_bior44_hsv_4x4_lvl3',
    'block_bior44_hsv_8x8_lvl3'
]

ALL_DISTANCE_METRICS = [
    'euclidean.euclidean_distance',
    'l1.compute_l1_distance',
    'chi_2.compute_chi_2_distance',
    'histogram_intersection.compute_histogram_intersection',
    'hellinger.hellinger_kernel',
    'cosine.compute_cosine_similarity',
    'canberra.canberra_distance',
    'bhattacharyya.bhattacharyya_distance',
    'jensen_shannon.jeffrey_divergence',
    'correlation.correlation_distance'
]

DESCRIPTOR_NAMES = {
    'rgb': 'RGB',
    'hsv': 'HSV',
    'ycbcr': 'YCbCr',
    'lab': 'LAB',
    'grayscale': 'Grayscale',
    '3d_rgb': '3D_RGB',
    '3d_hsv': '3D_HSV',
    '3d_lab': '3D_LAB',
    '2d_ycbcr': '2D_YCbCr',
    '2d_lab': '2D_LAB',
    '2d_hsv': '2D_HSV',
    'spatial_pyramid_lab': 'Spatial_Pyramid_LAB',
    'spatial_pyramid_hsv_lvl2': 'Spatial_Pyramid_HSV_LVL2',
    'spatial_pyramid_hsv_lvl3': 'Spatial_Pyramid_HSV_LVL3',
    'spatial_pyramid_hsv_lvl4': 'Spatial_Pyramid_HSV_LVL4',
    'spatial_pyramid_hsv_lvl5': 'Spatial_Pyramid_HSV_LVL5',
    'spatial_pyramid_2d_lab': 'Spatial_Pyramid_2D_LAB',
    'spatial_pyramid_2d_hsv_lvl2': 'Spatial_Pyramid_2D_HSV_LVL2',
    'spatial_pyramid_2d_hsv_lvl3': 'Spatial_Pyramid_2D_HSV_LVL3',
    'spatial_pyramid_2d_hsv_lvl4': 'Spatial_Pyramid_2D_HSV_LVL4',
    'spatial_pyramid_2d_hsv_lvl5': 'Spatial_Pyramid_2D_HSV_LVL5',
    'spatial_pyramid_3d_lab': 'Spatial_Pyramid_3D_LAB',
    'spatial_pyramid_3d_hsv_lvl2': 'Spatial_Pyramid_3D_HSV_LVL2',
    'spatial_pyramid_3d_hsv_lvl3': 'Spatial_Pyramid_3D_HSV_LVL3',
    'spatial_pyramid_3d_hsv_lvl4': 'Spatial_Pyramid_3D_HSV_LVL4',
    'spatial_pyramid_3d_hsv_lvl5': 'Spatial_Pyramid_3D_HSV_LVL5',
    'block_histogram_lab': 'Block_Histogram_LAB',
    'block_histogram_hsv_2x2': 'Block_Histogram_HSV_2X2',
    'block_histogram_hsv_4x4': 'Block_Histogram_HSV_4X4',
    'block_histogram_hsv_8x8': 'Block_Histogram_HSV_8X8',
    'block_histogram_hsv_16x16': 'Block_Histogram_HSV_16X16',
    'block_histogram_2d_lab': 'Block_Histogram_2D_LAB',
    'block_histogram_2d_hsv_2x2': 'Block_Histogram_2D_HSV_2X2',
    'block_histogram_2d_hsv_4x4': 'Block_Histogram_2D_HSV_4X4',
    'block_histogram_2d_hsv_8x8': 'Block_Histogram_2D_HSV_8X8',
    'block_histogram_2d_hsv_16x16': 'Block_Histogram_2D_HSV_16X16',
    'block_histogram_3d_lab': 'Block_Histogram_3D_LAB',
    'block_histogram_3d_hsv_2x2': 'Block_Histogram_3D_HSV_2X2',
    'block_histogram_3d_hsv_4x4': 'Block_Histogram_3D_HSV_4X4',
    'block_histogram_3d_hsv_8x8': 'Block_Histogram_3D_HSV_8X8',
    'block_histogram_3d_hsv_16x16': 'Block_Histogram_3D_HSV_16X16',
    'dct_grayscale_4x4_8coeffs': 'DCT_Grayscale_4x4_8Coeffs',
    'dct_grayscale_4x4_16coeffs': 'DCT_Grayscale_4x4_16Coeffs',
    'dct_grayscale_4x4_32coeffs': 'DCT_Grayscale_4x4_32Coeffs',
    'dct_grayscale_8x8_8coeffs': 'DCT_Grayscale_8x8_8Coeffs',
    'dct_grayscale_8x8_16coeffs': 'DCT_Grayscale_8x8_16Coeffs',
    'dct_grayscale_8x8_32coeffs': 'DCT_Grayscale_8x8_32Coeffs',
    'dct_hsv_4x4_8coeffs': 'DCT_HSV_4x4_8Coeffs',
    'dct_hsv_4x4_16coeffs': 'DCT_HSV_4x4_16Coeffs',
    'dct_hsv_4x4_32coeffs': 'DCT_HSV_4x4_32Coeffs',
    'dct_hsv_8x8_8coeffs': 'DCT_HSV_8x8_8Coeffs',
    'dct_hsv_8x8_16coeffs': 'DCT_HSV_8x8_16Coeffs',
    'dct_hsv_8x8_32coeffs': 'DCT_HSV_8x8_32Coeffs',
    'dct_lab_4x4_8coeffs': 'DCT_LAB_4x4_8Coeffs',
    'dct_lab_4x4_16coeffs': 'DCT_LAB_4x4_16Coeffs',
    'dct_lab_4x4_32coeffs': 'DCT_LAB_4x4_32Coeffs',
    'dct_lab_8x8_8coeffs': 'DCT_LAB_8x8_8Coeffs',
    'dct_lab_8x8_16coeffs': 'DCT_LAB_8x8_16Coeffs',
    'dct_lab_8x8_32coeffs': 'DCT_LAB_8x8_32Coeffs',
    'lbp_gray_s1_4x4':   'LBP_Gray_S1_4x4',
    'lbp_gray_ms2_4x4':  'LBP_Gray_MS2_4x4',
    'lbp_gray_ms2_8x8':  'LBP_Gray_MS2_8x8',
    'lbp_lab_ms2_4x4':   'LBP_LAB_MS2_4x4',
    'lbp_lab_ms2_8x8':   'LBP_LAB_MS2_8x8',
    'lbp_hsv_ms2_4x4':   'LBP_HSV_MS2_4x4',
    'lbp_hsv_ms2_8x8':   'LBP_HSV_MS2_8x8',
    'haar_grayscale_lvl1': 'Haar_Grayscale_LVL1',
    'haar_grayscale_lvl2': 'Haar_Grayscale_LVL2',
    'haar_grayscale_lvl3': 'Haar_Grayscale_LVL3',
    'haar_hsv_lvl1': 'Haar_HSV_LVL1',
    'haar_hsv_lvl2': 'Haar_HSV_LVL2',
    'haar_hsv_lvl3': 'Haar_HSV_LVL3',
    'bior44_grayscale_lvl1': 'Bior44_Grayscale_LVL1',
    'bior44_grayscale_lvl2': 'Bior44_Grayscale_LVL2',
    'bior44_grayscale_lvl3': 'Bior44_Grayscale_LVL3',
    'bior44_hsv_lvl1': 'Bior44_HSV_LVL1',
    'bior44_hsv_lvl2': 'Bior44_HSV_LVL2',
    'bior44_hsv_lvl3': 'Bior44_HSV_LVL3',
    'block_haar_grayscale_4x4_lvl1': 'Block_Haar_Grayscale_4x4_LVL1',
    'block_haar_grayscale_8x8_lvl1': 'Block_Haar_Grayscale_8x8_LVL1',
    'block_haar_grayscale_4x4_lvl2': 'Block_Haar_Grayscale_4x4_LVL2',
    'block_haar_grayscale_8x8_lvl2': 'Block_Haar_Grayscale_8x8_LVL2',
    'block_haar_grayscale_4x4_lvl3': 'Block_Haar_Grayscale_4x4_LVL3',
    'block_haar_grayscale_8x8_lvl3': 'Block_Haar_Grayscale_8x8_LVL3',
    'block_haar_hsv_4x4_lvl1': 'Block_Haar_HSV_4x4_LVL1',
    'block_haar_hsv_8x8_lvl1': 'Block_Haar_HSV_8x8_LVL1',
    'block_haar_hsv_4x4_lvl2': 'Block_Haar_HSV_4x4_LVL2',
    'block_haar_hsv_8x8_lvl2': 'Block_Haar_HSV_8x8_LVL2',
    'block_haar_hsv_4x4_lvl3': 'Block_Haar_HSV_4x4_LVL3',
    'block_haar_hsv_8x8_lvl3': 'Block_Haar_HSV_8x8_LVL3',
    'block_bior44_grayscale_4x4_lvl1': 'Block_Bior44_Grayscale_4x4_LVL1',
    'block_bior44_grayscale_8x8_lvl1': 'Block_Bior44_Grayscale_8x8_LVL1',
    'block_bior44_grayscale_4x4_lvl2': 'Block_Bior44_Grayscale_4x4_LVL2',
    'block_bior44_grayscale_8x8_lvl2': 'Block_Bior44_Grayscale_8x8_LVL2',
    'block_bior44_grayscale_4x4_lvl3': 'Block_Bior44_Grayscale_4x4_LVL3',
    'block_bior44_grayscale_8x8_lvl3': 'Block_Bior44_Grayscale_8x8_LVL3',
    'block_bior44_hsv_4x4_lvl1': 'Block_Bior44_HSV_4x4_LVL1',
    'block_bior44_hsv_8x8_lvl1': 'Block_Bior44_HSV_8x8_LVL1',
    'block_bior44_hsv_4x4_lvl2': 'Block_Bior44_HSV_4x4_LVL2',
    'block_bior44_hsv_8x8_lvl2': 'Block_Bior44_HSV_8x8_LVL2',
    'block_bior44_hsv_4x4_lvl3': 'Block_Bior44_HSV_4x4_LVL3',
    'block_bior44_hsv_8x8_lvl3': 'Block_Bior44_HSV_8x8_LVL3'
}

DISTANCE_NAMES = {
    'euclidean.euclidean_distance': 'Euclidean',
    'l1.compute_l1_distance': 'L1',
    'chi_2.compute_chi_2_distance': 'Chi-Square',
    'histogram_intersection.compute_histogram_intersection': 'Hist. Intersection',
    'hellinger.hellinger_kernel': 'Hellinger',
    'cosine.compute_cosine_similarity': 'Cosine',
    'canberra.canberra_distance': 'Canberra',
    'bhattacharyya.bhattacharyya_distance': 'Bhattacharyya',
    'jensen_shannon.jeffrey_divergence': 'Jeffrey Div.',
    'correlation.correlation_distance': 'Correlation'
}


def evaluate_all_descriptors_and_distances(
    qsd1_dir: str, museum_dir: str,
        ground_truth_pickle: str, values_per_bin: int = 1,
        k_values: List[int] = [1, 5],
        descriptors: List[str] = None,
        distance_metrics: List[str] = None):
    """
    Evaluate all combinations of descriptors and distance metrics.

    Args:
        qsd1_dir: Query images directory
        museum_dir: Museum database directory
        ground_truth_pickle: Path to ground truth pickle
        values_per_bin: Number of values per histogram bin
        k_values: List of k values for evaluation
        descriptors: List of descriptors to evaluate (None = all)
        distance_metrics: List of distance metrics to evaluate (None = all)
    """
    qsd1_path = Path(qsd1_dir)
    query_images = sorted(qsd1_path.glob('*.jpg'))

    # Load ground truth
    with open(ground_truth_pickle, 'rb') as f:
        ground_truth = pickle.load(f)

    logger.info(f"Loaded ground truth: {len(ground_truth)} entries")

    # Use provided lists or default to all
    if descriptors is None:
        descriptors = ALL_DESCRIPTORS
    else:
        # Validate provided descriptors
        for desc in descriptors:
            if desc not in ALL_DESCRIPTORS:
                raise ValueError(f"Invalid descriptor: {desc}. "
                                 f"Valid options: {ALL_DESCRIPTORS}")

    if distance_metrics is None:
        distance_metrics = ALL_DISTANCE_METRICS
    else:
        # Validate provided distance metrics
        for dist in distance_metrics:
            if dist not in ALL_DISTANCE_METRICS:
                raise ValueError(f"Invalid distance metric: {dist}. "
                                 f"Valid options: {ALL_DISTANCE_METRICS}")

    # Results dictionary - organized by descriptor then distance
    all_results = {}
    total_combinations = len(descriptors) * len(distance_metrics)
    combination_count = 0

    logger.info(f"Descriptors: {len(descriptors)}, "
                f"Distance Metrics: {len(distance_metrics)}")
    logger.info(f"Total Combinations: {total_combinations}")
    logger.info(f"Values per bin: {values_per_bin}")

    for desc_idx, descriptor in enumerate(descriptors):
        descriptor_results = {}

        logger.info(f"DESCRIPTOR: {DESCRIPTOR_NAMES[descriptor]} "
                    f"({desc_idx + 1}/{len(descriptors)})")

        # Create system once per descriptor with caching enabled
        # The first distance metric will build the database, the rest will reuse it
        system = None

        for dist_metric in distance_metrics:
            combination_count += 1
            logger.info(
                f"[{combination_count}/{total_combinations}] "
                f"{DESCRIPTOR_NAMES[descriptor]} + {DISTANCE_NAMES[dist_metric]}")

            # Initialize system with cache (reuses database if descriptor unchanged)
            if system is None:
                system = ComputeImageHistogram(
                    museum_dir, dist_metric, descriptor, values_per_bin)
            else:
                # Reuse existing system, only update distance metric
                system.distance_metric = dist_metric
                logger.info(f"Reusing cached database for {DESCRIPTOR_NAMES[descriptor]}")

            # Retrieve for all queries
            all_predicted = []
            all_actual = []

            for query_idx, query_path in enumerate(
                    tqdm(query_images, desc="Queries", leave=False)
            ):
                if query_idx < len(ground_truth):
                    retrieved = system.retrieve(
                        str(query_path), k=max(k_values))
                    predicted_ids = [img_id for img_id, _ in retrieved]

                    # Get ground truth for this query
                    gt = ground_truth[query_idx]
                    actual_ids = gt if isinstance(gt, list) else [gt]

                    all_predicted.append(predicted_ids)
                    all_actual.append(actual_ids)

            # Compute mAP@1 and mAP@5 using the new metrics
            map_1 = mapk(all_actual, all_predicted, k=1)
            map_5 = mapk(all_actual, all_predicted, k=5)

            descriptor_results[DISTANCE_NAMES[dist_metric]] = {
                'mAP@1': map_1,
                'mAP@5': map_5,
                'predicted': all_predicted,
                'actual': all_actual
            }

            logger.info(f"   mAP@1: {map_1:.4f}, mAP@5: {map_5:.4f}")

        all_results[DESCRIPTOR_NAMES[descriptor]] = descriptor_results

    return all_results
