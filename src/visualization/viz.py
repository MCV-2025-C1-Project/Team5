import numpy as np
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import pickle
from PIL import Image

from src.models.eval import evaluate_all_descriptors_and_distances
from src.models.main import ComputeImageHistogram
from src.tools.startup import logger


# Create system with best mAP@5 combination for sample visualizations
DESC_NAME_TO_FUNC = {
    'RGB': 'rgb',
    'HSV': 'hsv',
    'YCbCr': 'ycbcr',
    'LAB': 'lab',
    'Grayscale': 'grayscale',
    '3D_RGB': '3d_rgb',
    '3D_HSV': '3d_hsv',
    '3D_LAB': '3d_lab',
    '2D_YCbCr': '2d_ycbcr',
    'Spatial_Pyramid': 'spatial_pyramid',
    'Block_Histogram': 'block_histogram'
}

DIST_NAME_TO_FUNC = {
    'Euclidean': 'euclidean.euclidean_distance',
    'L1': 'l1.compute_l1_distance',
    'Chi-Square': 'chi_2.compute_chi_2_distance',
    'Hist. Intersection': 'histogram_intersection.compute_histogram_intersection',
    'Hellinger': 'hellinger.hellinger_kernel',
    'Cosine': 'cosine.compute_cosine_similarity',
    'Canberra': 'canberra.canberra_distance',
    'Bhattacharyya': 'bhattacharyya.bhattacharyya_distance',
    'Jeffrey Div.': 'jensen_shannon.jeffrey_divergence',
    'Correlation': 'correlation.correlation_distance'
}


def plot_top_k_results(query_image_path, retrieved_results,
                       museum_dir, save_path, k: int = 5):
    """Plot top-k retrieval results for a query image."""
    _, axes = plt.subplots(1, k+1, figsize=(3*(k+1), 4))

    # Plot query image
    query_img = Image.open(query_image_path)
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image', fontweight='bold')
    axes[0].axis('off')

    # Plot top-k retrieved images
    for i, (museum_id, distance) in enumerate(retrieved_results[:k]):
        museum_path = Path(museum_dir) / f"bbdd_{museum_id:05d}.jpg"
        if museum_path.exists():
            retrieved_img = Image.open(museum_path)
            axes[i+1].imshow(retrieved_img)
            axes[i +
                 1].set_title(f'#{i+1}\nID: {museum_id}\nDist: {distance:.3f}')
            axes[i+1].axis('off')
        else:
            axes[i+1].text(0.5, 0.5, f'Image {museum_id}\nnot found',
                           ha='center', va='center', transform=axes[i+1].transAxes)
            axes[i+1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of showing


def generate_comprehensive_analysis(
    qsd1_dir: str,
    museum_dir: str,
    ground_truth_pickle: str,
    values_per_bin: int = 1,
    output_dir: str = "data/results",
    k: int = 5,
    descriptors: List[str] = None,
    distance_metrics: List[str] = None
):
    """Generate complete analysis for all descriptors and distance metrics.

    Args:
        qsd1_dir: Query images directory
        museum_dir: Museum database directory
        ground_truth_pickle: Path to ground truth pickle
        values_per_bin: Number of values per histogram bin
        output_dir: Directory to save results
        k: Number of top-k retrievals
        descriptors: List of descriptors to evaluate (None = all)
        distance_metrics: List of distance metrics to evaluate (None = all)
    """

    import os
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Values per bin: {values_per_bin}")
    logger.info(f"Results will be saved to: {output_dir}")

    if descriptors:
        logger.info(f"Evaluating descriptors: {descriptors}")
    else:
        logger.info(f"Evaluating all descriptors")

    if distance_metrics:
        logger.info(f"Evaluating distance metrics: {distance_metrics}")
    else:
        logger.info(f"Evaluating all distance metrics")

    # Evaluate all descriptor-distance combinations
    all_results = evaluate_all_descriptors_and_distances(
        qsd1_dir, museum_dir, ground_truth_pickle, values_per_bin,
        descriptors=descriptors, distance_metrics=distance_metrics
    )

    # Find overall best combination
    best_overall_map1 = 0
    best_overall_map5 = 0
    best_desc_map1 = ""
    best_dist_map1 = ""
    best_desc_map5 = ""
    best_dist_map5 = ""

    for descriptor, distance_results in all_results.items():
        for distance, metrics in distance_results.items():
            if metrics['mAP@1'] > best_overall_map1:
                best_overall_map1 = metrics['mAP@1']
                best_desc_map1 = descriptor
                best_dist_map1 = distance
            if metrics['mAP@5'] > best_overall_map5:
                best_overall_map5 = metrics['mAP@5']
                best_desc_map5 = descriptor
                best_dist_map5 = distance

    logger.info(f"BEST OVERALL COMBINATIONS:")
    logger.info(f"   Best mAP@1: {best_desc_map1} + {best_dist_map1} "
                f"= {best_overall_map1:.4f}")
    logger.info(f"   Best mAP@5: {best_desc_map5} + {best_dist_map5} "
                f"= {best_overall_map5:.4f}")

    # Create comprehensive heatmap for all combinations
    descriptor_names = list(all_results.keys())
    distance_names = list(next(iter(all_results.values())).keys())

    # Create matrices for mAP@1 and mAP@5
    map1_matrix = np.zeros((len(descriptor_names), len(distance_names)))
    map5_matrix = np.zeros((len(descriptor_names), len(distance_names)))

    for i, descriptor in enumerate(descriptor_names):
        for j, distance in enumerate(distance_names):
            map1_matrix[i, j] = all_results[descriptor][distance]['mAP@1']
            map5_matrix[i, j] = all_results[descriptor][distance]['mAP@5']

    # Create comprehensive heatmaps
    # mAP@1 heatmap
    _, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(map1_matrix, cmap='YlOrRd', aspect='auto')
    ax.figure.colorbar(im, ax=ax, label='mAP@1 Score')

    ax.set_xticks(range(len(distance_names)))
    ax.set_yticks(range(len(descriptor_names)))
    ax.set_xticklabels(distance_names, rotation=45, ha='right')
    ax.set_yticklabels(descriptor_names)

    # Add text annotations
    for i in range(len(descriptor_names)):
        for j in range(len(distance_names)):
            ax.text(j, i, f'{map1_matrix[i, j]:.3f}',
                    ha="center", va="center",
                    color="black", fontweight='bold', fontsize=8)

    ax.set_title('mAP@1 Performance: All Descriptor-Distance Combinations')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_heatmap_map1.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # mAP@5 heatmap
    _, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(map5_matrix, cmap='YlOrRd', aspect='auto')
    ax.figure.colorbar(im, ax=ax, label='mAP@5 Score')

    ax.set_xticks(range(len(distance_names)))
    ax.set_yticks(range(len(descriptor_names)))
    ax.set_xticklabels(distance_names, rotation=45, ha='right')
    ax.set_yticklabels(descriptor_names)

    # Add text annotations
    for i in range(len(descriptor_names)):
        for j in range(len(distance_names)):
            ax.text(j, i, f'{map5_matrix[i, j]:.3f}',
                    ha="center", va="center",
                    color="black", fontweight='bold', fontsize=12)

    ax.set_title('mAP@5 Performance: All Descriptor-Distance Combinations')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_heatmap_map5.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    best_desc_func = DESC_NAME_TO_FUNC[best_desc_map5]
    best_dist_func = DIST_NAME_TO_FUNC[best_dist_map5]
    best_system = ComputeImageHistogram(
        museum_dir, best_dist_func, best_desc_func, values_per_bin)

    # Load ground truth
    with open(ground_truth_pickle, 'rb') as f:
        ground_truth = pickle.load(f)

    # Generate sample retrieval visualizations
    qsd1_path = Path(qsd1_dir)
    query_images = sorted(qsd1_path.glob('*.jpg'))

    # Select sample queries
    sample_indices = np.random.choice(len(query_images),
                                      min(10, len(query_images)),
                                      replace=False)

    for i, idx in enumerate(sample_indices):
        query_path = query_images[idx]
        retrieved = best_system.retrieve(str(query_path), k=k)

        save_path = f'{output_dir}/sample_{i+1}_query_{idx:05d}_top{k}.png'
        plot_top_k_results(str(query_path), retrieved,
                           museum_dir, k=k, save_path=save_path)

        # Print ground truth info
        if idx < len(ground_truth):
            gt = ground_truth[idx]
            gt_id = gt[0] if isinstance(gt, list) else gt
            logger.info(f"Query {idx}: GT={gt_id}, "
                        f"Retrieved={[id for id, _ in retrieved]}, "
                        f"Top-1 Correct={retrieved[0][0] == gt_id}")

    # Generate results for all queries and save as pickle
    logger.info("Generating results for all queries...")
    predictions = []
    gt_labels = []
    result_list = []  # List of lists for pickle output

    for query_idx, query_path in enumerate(query_images):
        if query_idx < len(ground_truth):
            retrieved = best_system.retrieve(str(query_path), k=k)
            top_k_ids = [img_id for img_id, _ in retrieved]
            result_list.append(top_k_ids)
            predictions.append(retrieved[0][0])  # Top-1 prediction

            gt = ground_truth[query_idx]
            gt_id = gt[0] if isinstance(gt, list) else gt
            gt_labels.append(gt_id)

    # Save result_list as pickle
    result_pickle_path = f'{output_dir}/result.pkl'
    with open(result_pickle_path, 'wb') as f:
        pickle.dump(result_list, f)
    logger.info(f"Saved top-{k} results to: {result_pickle_path}")

    # Create and save confusion matrices
    logger.info("Creating confusion matrices...")
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Get unique labels for confusion matrix
    unique_labels = sorted(list(set(gt_labels + predictions)))

    # Confusion matrix for top-1 predictions
    cm = confusion_matrix(gt_labels, predictions, labels=unique_labels)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f"Confusion Matrix - Top-1 Predictions\n"
              f"{best_desc_map5} + {best_dist_map5}")
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_top1.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Top-5 accuracy confusion matrix (binary: correct if GT in top-5)
    top_k_correct = []
    for idx, (gt, result) in enumerate(zip(gt_labels, result_list)):
        top_k_correct.append(1 if gt in result else 0)

    # Create a simple accuracy matrix
    logger.info("Creating accuracy matrix...")
    plt.figure(figsize=(10, 8))
    accuracy_data = np.array(
        [[sum(top_k_correct), len(top_k_correct) - sum(top_k_correct)]])
    sns.heatmap(accuracy_data, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Incorrect', 'Correct'], yticklabels=[f'Top-{k} Accuracy'])
    plt.title(f'Top-{k} Accuracy Matrix\n{best_desc_map5} + {best_dist_map5}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_top{k}.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Save summary statistics
    summary = {
        'best_descriptor': best_desc_map5,
        'best_distance': best_dist_map5,
        'best_mAP@1': best_overall_map1,
        'best_mAP@5': best_overall_map5,
        'top_1_accuracy': sum([1 for gt, pred in zip(gt_labels, predictions) if gt == pred]) / len(gt_labels),
        f'top_{k}_accuracy': sum(top_k_correct) / len(top_k_correct),
        'values_per_bin': values_per_bin,
        'k': k
    }

    summary_path = f'{output_dir}/summary.pkl'
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    logger.info(f"Saved summary to: {summary_path}")

    logger.info(f"All results saved to: {output_dir}")
    logger.info("  - Heatmaps: comprehensive_heatmap_map1.png, "
                "comprehensive_heatmap_map5.png")
    logger.info(f"  - Sample visualizations: sample_*.png")
    logger.info(f"  - Top-{k} results: result.pkl")
    logger.info(f"  - Confusion matrices: confusion_matrix_top1.png, "
                f"confusion_matrix_top{k}.png")
    logger.info(f"  - Summary: summary.pkl")

    return all_results, best_desc_map5, best_dist_map5, best_overall_map5, result_list
