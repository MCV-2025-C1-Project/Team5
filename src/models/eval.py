from pathlib import Path
from typing import List
import pickle
from tqdm import tqdm
from src.distances import (bhattacharyya,
                           canberra,
                           chi_2,
                           correlation,
                           cosine,
                           euclidean,
                           hellinger,
                           histogram_intersection,
                           jensen_shannon,
                           l1)
from src.metrics.precision import mapk
from src.models.main import ComputeImageFeatures
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
    'bior44_grayscale_lvl1',
    'bior44_grayscale_lvl2',
    'bior44_grayscale_lvl3',
    'bior44_hsv_lvl1',
    'bior44_hsv_lvl2',
    'bior44_hsv_lvl3',
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
    'block_bior44_hsv_8x8_lvl3',
    'sift_dog_default',
    'sift_harris_default',
    'sift_harris_laplacian_default',
    'hog_dog_default',
    'daisy_dog_default'
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
    'block_bior44_hsv_8x8_lvl3': 'Block_Bior44_HSV_8x8_LVL3',
    'sift_dog_default': 'SIFT_DOG_DEFAULT',
    'sift_harris_default': 'SIFT_HARRIS_DEFAULT',
    'sift_harris_laplacian_default': 'SIFT_HARRIS_LAPLACIAN_DEFAULT',
    'hog_dog_default': 'HOG_DOG_DEFAULT',
    'daisy_dog_default': 'DAISY_DOG_DEFAULT'
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
    distance_metrics: List[str] = None,
    mode: str = "global"):
    """
    Evaluate all combinations of descriptors and distance metrics.
    Handles multi-painting images properly by automatically using segmented
    paintings when available instead of the original multi-painting image.

    Top-k evaluation:
      • GT == [-1]: counting as correct iff the system predicts -1 within top-k.
      • Multi-painting: image-level Top-k checks ANY subpainting vs the GT set.
      • mAP@k is computed over flattened (subpainting-level) lists and includes -1 pairs.
    """
    qsd1_path = Path(qsd1_dir)
    all_query_images = sorted(qsd1_path.glob('*.jpg'))

    import re

    # Map images that have segmented versions
    has_segments = {}
    for img in all_query_images:
        match = re.match(r'(\d+)_(\d+)\.jpg$', img.name)
        if match:
            parent_id = int(match.group(1))
            has_segments[parent_id] = True

    # Filter: use segments, skip originals that have segments
    query_images = []
    for img in all_query_images:
        if re.search(r'_\d+\.jpg$', img.name):
            query_images.append(img)
        else:
            parent_id = int(img.stem)
            if parent_id not in has_segments:
                query_images.append(img)
            else:
                logger.debug(f"Skipping {img.name} - using segmented versions instead")

    logger.info(f"Filtered {len(all_query_images)} -> {len(query_images)} images (skipped originals with segments)")

    # Load GT
    with open(ground_truth_pickle, 'rb') as f:
        ground_truth = pickle.load(f)
    logger.info(f"Loaded ground truth: {len(ground_truth)} entries")

    # Group queries by parent (e.g., {2: ['00002_0.jpg','00002_1.jpg']})
    grouped_queries = {}
    for query_path in query_images:
        stem = query_path.stem
        parent_id = int(stem.split('_')[0])
        if parent_id not in grouped_queries:
            grouped_queries[parent_id] = []
        grouped_queries[parent_id].append(query_path)

    for parent_id in grouped_queries:
        grouped_queries[parent_id].sort()

    # Validate descriptor / distance lists
    if descriptors is None:
        descriptors = ALL_DESCRIPTORS
    else:
        for desc in descriptors:
            if desc not in ALL_DESCRIPTORS:
                raise ValueError(f"Invalid descriptor: {desc}")

    if distance_metrics is None:
        distance_metrics = ALL_DISTANCE_METRICS
    else:
        for dist in distance_metrics:
            if dist not in ALL_DISTANCE_METRICS:
                raise ValueError(f"Invalid distance metric: {dist}")

    all_results = {}
    total_combinations = len(descriptors) * len(distance_metrics)
    logger.info(f"Descriptors: {len(descriptors)}, Distance Metrics: {len(distance_metrics)}")
    logger.info(f"Total Combinations: {total_combinations}")
    logger.info(f"Values per bin: {values_per_bin}")

    combo_idx = 0
    for desc_idx, descriptor in enumerate(descriptors):
        descriptor_results = {}
        logger.info(f"DESCRIPTOR: {DESCRIPTOR_NAMES[descriptor]} ({desc_idx + 1}/{len(descriptors)})")
        system = None

        for dist_metric in distance_metrics:
            combo_idx += 1
            logger.info(f"[{combo_idx}/{total_combinations}] "
                        f"{DESCRIPTOR_NAMES[descriptor]} + {DISTANCE_NAMES[dist_metric]}")

            if system is None:
                system = ComputeImageFeatures(
                    museum_dir, dist_metric, descriptor, mode, values_per_bin,
                    ground_truth=ground_truth_pickle
                )
            else:
                system.distance_metric = dist_metric
                logger.info(f"Reusing cached database for {DESCRIPTOR_NAMES[descriptor]}")

            # Flattened lists for mAP
            all_predicted_flat = []
            all_actual_flat = []

            # Image-level Top-k counters (including -1)
            img_valid = 0
            img_top1_hits = 0
            img_top5_hits = 0

            for parent_id in sorted(grouped_queries.keys()):
                query_paths = grouped_queries[parent_id]
                gt = ground_truth[parent_id]
                gt_list = gt if isinstance(gt, list) else [gt]

                # Retrieve predictions for all sub-paintings
                predictions_for_image = []
                for query_path in tqdm(query_paths, desc=f"Parent {parent_id}", leave=False):
                    retrieval_result = system.retrieve(str(query_path), k=max(k_values))

                    if mode == "local":
                        retrieved, _ = retrieval_result
                        if len(retrieved) > 0 and retrieved[0][0] == -1:
                            predicted_ids = [-1]
                        else:
                            predicted_ids = [img_id for img_id, _, _ in retrieved]
                    else:
                        retrieved = retrieval_result
                        if len(retrieved) > 0 and retrieved[0] == -1:
                            predicted_ids = [-1]
                        else:
                            predicted_ids = [img_id for img_id, _ in retrieved]

                    predictions_for_image.append(predicted_ids)

                # Use ground truth in original order (no swapping)
                actual = [[gt] for gt in gt_list]
                predicted = predictions_for_image

                # ---- Flatten for mAP (INCLUDING -1 policy) ----
                # We keep -1 pairs so that mAP@k rewards ranking -1 within top-k
                all_predicted_flat.extend(predicted)
                all_actual_flat.extend(actual)

                # ---- Image-level Top-k (including -1 as valid) ----
                img_valid += 1

                if gt_list == [-1]:
                    # Count a hit if any sub-painting predicts -1 within top-k
                    top1_hit = any((pred and pred[0] == -1) for pred in predicted)
                    top5_hit = any((-1 in (pred[:5] if pred else [])) for pred in predicted)
                    if top1_hit:
                        img_top1_hits += 1
                    if top5_hit:
                        img_top5_hits += 1
                else:
                    gt_set = set(gt_list)

                    # Union of first predictions across sub-paintings (allow -1; it just won't match gt_set)
                    top1_preds = {pred[0] for pred in predicted if pred}

                    # Union of top-5 predictions across sub-paintings
                    top5_preds = set()
                    for pred in predicted:
                        if pred:
                            top5_preds.update(pred[:5])

                    if gt_set & top1_preds:
                        img_top1_hits += 1
                    if gt_set & top5_preds:
                        img_top5_hits += 1

            # Compute mAP@1 and mAP@5 over flattened pairs
            map_1 = mapk(all_actual_flat, all_predicted_flat, k=1)
            map_5 = mapk(all_actual_flat, all_predicted_flat, k=5)

            # Image-level accuracies
            top1_acc = img_top1_hits / img_valid if img_valid else 0.0
            top5_acc = img_top5_hits / img_valid if img_valid else 0.0

            descriptor_results[DISTANCE_NAMES[dist_metric]] = {
                'mAP@1': map_1,
                'mAP@5': map_5,
                'img_top1_acc': top1_acc,
                'img_top5_acc': top5_acc,
                'img_valid': img_valid,
                'predicted': all_predicted_flat,
                'actual': all_actual_flat,
                'grouped_queries': grouped_queries
            }

            logger.info(f"   mAP@1: {map_1:.4f}, mAP@5: {map_5:.4f}")
            logger.info("   Image-level accuracy (including GT=-1 matches):")
            logger.info(f"     Valid images: {img_valid}")
            logger.info(f"     Top-1 accuracy: {img_top1_hits}/{img_valid} = {top1_acc:.4f}")
            logger.info(f"     Top-5 accuracy: {img_top5_hits}/{img_valid} = {top5_acc:.4f}")

        all_results[DESCRIPTOR_NAMES[descriptor]] = descriptor_results

    return all_results
