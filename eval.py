from pathlib import Path
from typing import List
from tqdm import tqdm
import pickle
from src.metrics.precision import mapk
from main import ComputeImageHistogram

from src.tools.startup import logger


def evaluate_all_descriptors_and_distances(qsd1_dir: str, museum_dir: str,
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

    # Define all available descriptors and distance metrics
    all_descriptors = ['rgb', 'hsv', 'ycbcr', 'lab', 'grayscale']
    all_distance_metrics = ['euclidean.euclidean_distance', 'l1.compute_l1_distance', 'chi_2.compute_chi_2_distance',
                            'histogram_intersection.compute_histogram_intersection', 'hellinger.hellinger_kernel', 'cosine.compute_cosine_similarity',
                            'canberra.canberra_distance', 'bhattacharyya.bhattacharyya_distance', 'jensen_shannon.jeffrey_divergence',
                            'correlation.correlation_distance']

    # Use provided lists or default to all
    if descriptors is None:
        descriptors = all_descriptors
    else:
        # Validate provided descriptors
        for desc in descriptors:
            if desc not in all_descriptors:
                raise ValueError(
                    f"Invalid descriptor: {desc}. Valid options: {all_descriptors}")

    if distance_metrics is None:
        distance_metrics = all_distance_metrics
    else:
        # Validate provided distance metrics
        for dist in distance_metrics:
            if dist not in all_distance_metrics:
                raise ValueError(
                    f"Invalid distance metric: {dist}. Valid options: {all_distance_metrics}")

    descriptor_names = {
        'rgb': 'RGB',
        'hsv': 'HSV',
        'ycbcr': 'YCbCr',
        'lab': 'LAB',
        'grayscale': 'Grayscale'
    }

    distance_names = {
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

    # Results dictionary - organized by descriptor then distance
    all_results = {}
    total_combinations = len(descriptors) * len(distance_metrics)
    combination_count = 0

    logger.info(
        f"Descriptors: {len(descriptors)}, Distance Metrics: {len(distance_metrics)}")
    logger.info(f"Total Combinations: {total_combinations}")
    logger.info(f"Values per bin: {values_per_bin}")

    for desc_idx, descriptor in enumerate(descriptors):
        descriptor_results = {}

        logger.info(
            f"DESCRIPTOR: {descriptor_names[descriptor]} ({desc_idx + 1}/{len(descriptors)})")

        for dist_metric in distance_metrics:
            combination_count += 1
            logger.info(f"[{combination_count}/{total_combinations}] "
                        f"{descriptor_names[descriptor]} + {distance_names[dist_metric]}")

            # Initialize system
            system = ComputeImageHistogram(
                museum_dir, dist_metric, descriptor, values_per_bin)

            # Retrieve for all queries
            all_predicted = []
            all_actual = []

            for query_idx, query_path in enumerate(tqdm(query_images, desc="Queries", leave=False)):
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

            descriptor_results[distance_names[dist_metric]] = {
                'mAP@1': map_1,
                'mAP@5': map_5,
                'predicted': all_predicted,
                'actual': all_actual
            }

            logger.info(f"   mAP@1: {map_1:.4f}, mAP@5: {map_5:.4f}")

        all_results[descriptor_names[descriptor]] = descriptor_results

    return all_results
