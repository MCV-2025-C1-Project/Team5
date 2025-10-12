from pathlib import Path
import pickle
from typing import Dict, List
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    '2D_LAB': '2d_lab',
    '2D_HSV': '2d_hsv',
    'Spatial_Pyramid_LAB': 'spatial_pyramid_lab',
    'Spatial_Pyramid_HSV': 'spatial_pyramid_hsv',
    'Spatial_Pyramid_2D_LAB': 'spatial_pyramid_2d_lab',
    'Spatial_Pyramid_2D_HSV': 'spatial_pyramid_2d_hsv',
    'Spatial_Pyramid_3D_LAB': 'spatial_pyramid_3d_lab',
    'Spatial_Pyramid_3D_HSV': 'spatial_pyramid_3d_hsv',
    'Block_Histogram_LAB': 'block_histogram_lab',
    'Block_Histogram_HSV': 'block_histogram_hsv',
    'Block_Histogram_2D_LAB': 'block_histogram_2d_lab',
    'Block_Histogram_2D_HSV': 'block_histogram_2d_hsv',
    'Block_Histogram_3D_LAB': 'block_histogram_3d_lab',
    'Block_Histogram_3D_HSV': 'block_histogram_3d_hsv'
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


def apply_mask_to_image_mask(image: np.ndarray, mask: np.ndarray, crop: bool = True) -> np.ndarray:
    """Apply mask to image to get segmented foreground, optionally cropped to bounding box."""
    mask = (mask > 0)

    if crop:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return np.ones((100, 100, 3), dtype=np.uint8) * 255

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        cropped_image = image[rmin:rmax+1, cmin:cmax+1]
        cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]

        segmented = np.ones_like(cropped_image) * 255
        segmented[cropped_mask] = cropped_image[cropped_mask]

        return segmented.astype(np.uint8)
    else:
        segmented = np.ones_like(image) * 255
        segmented[mask] = image[mask]
        return segmented.astype(np.uint8)


def create_visualization_mask(original: np.ndarray,
                        mask: np.ndarray,
                        segmented: np.ndarray,
                        gt_mask: np.ndarray = None,
                        metrics: Dict = None,
                        title: str = "") -> plt.Figure:
    """Create comprehensive visualization with original, mask, segmented, and comparison."""

    if gt_mask is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes = axes.flatten()

    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Predicted mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Predicted Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Segmented output
    axes[2].imshow(segmented)
    axes[2].set_title('Segmented Output', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    if gt_mask is not None:
        # Ground truth mask
        axes[3].imshow(gt_mask, cmap='gray')
        axes[3].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
        axes[3].axis('off')

        # Mask comparison (TP=green, FP=red, FN=blue, TN=black)
        pred_bool = mask.astype(bool)
        gt_bool = gt_mask.astype(bool)

        tp_count = np.sum(pred_bool & gt_bool)
        fp_count = np.sum(pred_bool & ~gt_bool)
        fn_count = np.sum(~pred_bool & gt_bool)
        tn_count = np.sum(~pred_bool & ~gt_bool)

        comparison = np.zeros((*mask.shape, 3), dtype=np.uint8)
        comparison[pred_bool & gt_bool] = [0, 255, 0]   # TP
        comparison[pred_bool & ~gt_bool] = [255, 0, 0]  # FP
        comparison[~pred_bool & gt_bool] = [0, 0, 255]  # FN

        axes[4].imshow(comparison)
        title_text = (f'Mask Comparison\n'
                      f'Green=TP({tp_count:,}), Red=FP({fp_count:,}), '
                      f'Blue=FN({fn_count:,}), Black=TN({tn_count:,})')
        axes[4].set_title(title_text, fontsize=12, fontweight='bold')
        axes[4].axis('off')

        # Metrics text
        if metrics:
            axes[5].axis('off')
            pred_fg_pixels = metrics['tp'] + metrics['fp']
            gt_fg_pixels = metrics['tp'] + metrics['fn']
            metrics_text = (
                f"METRICS:\n"
                f"{'='*40}\n"
                f"Precision: {metrics['precision']:.4f}\n"
                f"  = TP / (TP + FP)\n"
                f"  = {metrics['tp']:,} / {pred_fg_pixels:,}\n\n"
                f"Recall: {metrics['recall']:.4f}\n"
                f"  = TP / (TP + FN)\n"
                f"  = {metrics['tp']:,} / {gt_fg_pixels:,}\n\n"
                f"F1 Score: {metrics['f1']:.4f}\n"
                f"{'='*40}\n"
                f"PIXEL COUNTS:\n"
                f"{'='*40}\n"
                f"True Positives:  {metrics['tp']:>10,}\n"
                f"False Positives: {metrics['fp']:>10,}\n"
                f"False Negatives: {metrics['fn']:>10,}\n"
                f"True Negatives:  {metrics['tn']:>10,}\n"
            )
            axes[5].text(0.05, 0.5, metrics_text,
                         fontsize=10, verticalalignment='center',
                         family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig


def visualize_color_pipeline_steps_mask(
    image: np.ndarray,
    *,
    border: int = 20,
    dist_percentile: float = 92.0,
    dist_margin: float = 0.5,
    opening_size: int = 5,
    closing_size: int = 7,
    min_area: int = 1500,
    wL: float = 0.6, wA: float = 1.0, wB: float = 1.0,

    sat_min: float = 30.0,
    hue_percentile: float = 92.0,
    hue_margin_deg: float = 6.0,
    save_path: str = None
) -> plt.Figure:
    """
    Visualize per-step outputs of the color-only pipeline (NO ARROWS).

    This function requires color pipeline helper functions to be imported.
    It creates a 2x5 grid visualization showing:
      [0] Original
      [1] LAB distance map
      [2] LAB mask (thresholded)
      [3] Hue distance map
      [4] Hue mask (thresholded + saturation check)
      [5] Combined mask (LAB OR Hue)
      [6] After Opening
      [7] After Closing
      [8] After Hole Fill
      [9] After Area-like Opening (Final)

    Note: This function requires additional imports and helper functions from the
    segmentation pipeline that are not included in this module.
    """
    import cv2
    from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes

    # Helper functions for color processing
    def _lab(img: np.ndarray) -> np.ndarray:
        """RGB -> LAB (float32) without any equalization."""
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

    def _hsv(img: np.ndarray) -> np.ndarray:
        """RGB -> HSV (float32) with H in [0,180) like OpenCV, S,V in [0,255]."""
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

    def _estimate_bg_from_borders(image_lab: np.ndarray, border_size: int = 20) -> Dict[str, np.ndarray]:
        """Estimate background color statistics from the image borders in LAB space."""
        H, W, _ = image_lab.shape
        b = max(1, min(border_size, min(H, W)//4))

        top    = image_lab[:b, :, :]
        bottom = image_lab[-b:, :, :]
        left   = image_lab[:, :b, :]
        right  = image_lab[:, -b:, :]

        borders = np.concatenate(
            [top.reshape(-1,3), bottom.reshape(-1,3), left.reshape(-1,3), right.reshape(-1,3)],
            axis=0
        )

        med = np.median(borders, axis=0)
        mad = np.median(np.abs(borders - med), axis=0) + 1e-6
        return {"median": med.astype(np.float32), "mad": mad.astype(np.float32)}

    def _lab_robust_distance_weighted(L: np.ndarray, A: np.ndarray, B: np.ndarray,
                                      med: np.ndarray, mad: np.ndarray,
                                      wL_: float = 0.6, wA_: float = 1.0, wB_: float = 1.0) -> np.ndarray:
        """Robust weighted distance to BG center in LAB, normalized by MAD per channel."""
        dL = (L - med[0]) / mad[0]
        dA = (A - med[1]) / mad[1]
        dB = (B - med[2]) / mad[2]
        dist = np.sqrt((wL_*dL)**2 + (wA_*dA)**2 + (wB_*dB)**2)
        return dist

    def _hue_circular_distance(h: np.ndarray, h0: float) -> np.ndarray:
        """Circular distance on hue (OpenCV H in [0,180))."""
        dh = np.abs(h - h0)
        dh = np.minimum(dh, 180.0 - dh)
        return dh

    def _morphological_area_like_opening(msk: np.ndarray, min_area_: int) -> np.ndarray:
        """Area-like filtering using only morphology (no connected components)."""
        if min_area_ <= 0:
            return msk

        r = max(1, int(np.sqrt(float(min_area_) / np.pi)))
        se = np.ones((2*r+1, 2*r+1), dtype=bool)

        msk = binary_opening(msk, structure=se)

        inv = ~msk
        inv = binary_opening(inv, structure=se)
        msk = ~inv

        return msk

    # --- Spaces & channels ---
    lab = _lab(image)
    hsv = _hsv(image)

    L = lab[..., 0]; A = lab[..., 1]; B = lab[..., 2]
    Hh = hsv[..., 0]; Ss = hsv[..., 1]

    H, W = L.shape
    b = max(1, min(border, min(H, W)//4))
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:b, :] = True; border_mask[-b:, :] = True
    border_mask[:, :b] = True; border_mask[:, -b:] = True

    # --- Background statistics from borders (LAB) ---
    stats = _estimate_bg_from_borders(lab, border_size=border)
    med = stats["median"]; mad = stats["mad"]


    # --- Step 1: LAB robust weighted distance ---
    dist_lab = _lab_robust_distance_weighted(L, A, B, med, mad, wL_=wL, wA_=wA, wB_=wB)

    # Threshold from *border* distribution (+ margin)
    border_dists = dist_lab[border_mask]
    if border_dists.size == 0:
        thr_lab = np.percentile(dist_lab, dist_percentile) + dist_margin
    else:
        thr_lab = np.percentile(border_dists, dist_percentile) + dist_margin
    mask_lab = dist_lab > thr_lab

    # --- Step 2: Hue fallback (HSV) ---
    border_hues = Hh[border_mask]
    h0 = np.median(border_hues) if border_hues.size > 0 else np.median(Hh)
    hue_dist = _hue_circular_distance(Hh, h0)
    border_hued = hue_dist[border_mask]
    thr_hue = (np.percentile(hue_dist, hue_percentile) + hue_margin_deg
               if border_hued.size == 0
               else np.percentile(border_hued, hue_percentile) + hue_margin_deg)
    mask_hue = (hue_dist > thr_hue) & (Ss >= sat_min)

    # --- Step 3: Combine purely color masks ---
    mask_combined = mask_lab | mask_hue

    # --- Step 4: Morph-only cleanup steps ---
    mask_open = binary_opening(mask_combined, structure=np.ones((opening_size, opening_size))) if opening_size > 0 else mask_combined.copy()
    mask_close = binary_closing(mask_open, structure=np.ones((closing_size, closing_size))) if closing_size > 0 else mask_open.copy()
    mask_filled = binary_fill_holes(mask_close)
    mask_final = _morphological_area_like_opening(mask_filled, min_area_=min_area)

    # --- Build figure (NO arrows) ---
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    # Normalize distance maps for display
    def norm_img(x):
        x = x.astype(np.float32)
        vmax = np.percentile(x, 99.5) if np.isfinite(x).all() else x.max()
        vmax = max(vmax, 1e-6)
        return np.clip(x / vmax, 0, 1)

    # Row 1
    axes[0].imshow(image); axes[0].set_title("Original", fontweight='bold'); axes[0].axis('off')
    axes[1].imshow(norm_img(dist_lab), cmap='gray'); axes[1].set_title(f"LAB distance\nthr={thr_lab:.2f}", fontweight='bold'); axes[1].axis('off')
    axes[2].imshow(mask_lab, cmap='gray'); axes[2].set_title("Mask (LAB > thr)", fontweight='bold'); axes[2].axis('off')
    axes[3].imshow(norm_img(hue_dist), cmap='gray'); axes[3].set_title(f"Hue distance\nh0≈{h0:.1f}°, thr≈{thr_hue:.1f}°", fontweight='bold'); axes[3].axis('off')
    axes[4].imshow(mask_hue, cmap='gray'); axes[4].set_title(f"Mask (Hue > thr & S≥{int(sat_min)})", fontweight='bold'); axes[4].axis('off')

    # Row 2
    axes[5].imshow(mask_combined, cmap='gray'); axes[5].set_title("Combined Mask (LAB ∪ Hue)", fontweight='bold'); axes[5].axis('off')
    axes[6].imshow(mask_open, cmap='gray'); axes[6].set_title(f"Opening {opening_size}×{opening_size}", fontweight='bold'); axes[6].axis('off')
    axes[7].imshow(mask_close, cmap='gray'); axes[7].set_title(f"Closing {closing_size}×{closing_size}", fontweight='bold'); axes[7].axis('off')
    axes[8].imshow(mask_filled, cmap='gray'); axes[8].set_title("Hole Fill", fontweight='bold'); axes[8].axis('off')
    axes[9].imshow(mask_final, cmap='gray'); axes[9].set_title(f"Area-like Opening (~≥{min_area}px)\nFINAL", fontweight='bold'); axes[9].axis('off')

    fig.suptitle("Per-step Color-only Segmentation Pipeline", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
    return fig


def create_summary_plots_mask(results: List[Dict], output_folder: str):
    """Create summary plots showing metrics across all mask segmentation experiments."""
    if not results:
        return

    # Extract metrics
    exp_names = [r['experiment_name'] for r in results]
    precisions = [r['avg_precision'] for r in results]
    recalls = [r['avg_recall'] for r in results]
    f1_scores = [r['avg_f1'] for r in results]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Bar plot of all metrics
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(exp_names))
    width = 0.25
    ax1.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax1.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax1.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
    ax1.axhline(y=0.90, color='r', linestyle='--', linewidth=2, label='Target (0.90)')
    ax1.set_xlabel('Experiment', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

    # 2. Precision vs Recall scatter
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(recalls, precisions, s=200, alpha=0.6, edgecolors='black', linewidth=2)
    for i, name in enumerate(exp_names):
        ax2.annotate(name, (recalls[i], precisions[i]), fontsize=8, ha='center')
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])

    # 3. Metric Values Plot
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(x, precisions, marker='o', label='Precision', linewidth=2)
    ax3.plot(x, recalls, marker='s', label='Recall', linewidth=2)
    ax3.plot(x, f1_scores, marker='^', label='F1', linewidth=2)
    ax3.set_xlabel('Experiment', fontweight='bold')
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('Metric Trends', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(exp_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1])

    # 4. F1 Score ranking
    ax4 = plt.subplot(2, 2, 4)
    sorted_indices = np.argsort(f1_scores)[::-1]
    sorted_f1 = [f1_scores[i] for i in sorted_indices]
    sorted_names = [exp_names[i] for i in sorted_indices]
    ax4.barh(range(len(sorted_names)), sorted_f1, alpha=0.8)
    ax4.set_yticks(range(len(sorted_names)))
    ax4.set_yticklabels(sorted_names)
    ax4.set_xlabel('F1 Score', fontweight='bold')
    ax4.set_title('F1 Score Ranking', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    ax4.set_xlim([0, 1])

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_folder, 'summary_metrics.png')
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary plots saved to: {plot_path}")


def save_results_mask(results: List[Dict], output_folder: str):
    """Save mask segmentation experiment results to files."""
    import datetime

    os.makedirs(output_folder, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    summary_file = os.path.join(output_folder, f'segmentation_results_{timestamp}.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BACKGROUND SEGMENTATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Experiments: {len(results)}\n\n")

        sorted_results = sorted(results, key=lambda x: x['avg_f1'], reverse=True)

        f.write("\n" + "="*80 + "\n")
        f.write("TOP EXPERIMENTS (by F1 Score)\n")
        f.write("="*80 + "\n\n")

        for i, result in enumerate(sorted_results[:10], 1):
            f.write(f"{i}. {result['experiment_name']}\n")
            f.write(f"   Precision: {result['avg_precision']:.4f}\n")
            f.write(f"   Recall:    {result['avg_recall']:.4f}\n")
            f.write(f"   F1:        {result['avg_f1']:.4f}\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("ALL EXPERIMENTS (sorted by F1)\n")
        f.write("="*80 + "\n\n")

        for result in sorted_results:
            f.write(f"{result['experiment_name']}\n")
            f.write(f"  P: {result['avg_precision']:.4f}, R: {result['avg_recall']:.4f}, ")
            f.write(f"F1: {result['avg_f1']:.4f}\n")

        # Per-image details
        f.write("\n\n" + "="*80 + "\n")
        f.write("PER-IMAGE RESULTS\n")
        f.write("="*80 + "\n\n")

        for result in sorted_results:
            f.write(f"\n{result['experiment_name']}\n")
            f.write("-" * 80 + "\n")
            for img_result in result['image_results']:
                f.write(f"  {img_result['image']}: ")
                f.write(f"P={img_result['precision']:.3f}, R={img_result['recall']:.3f}, ")
                f.write(f"F1={img_result['f1']:.3f}\n")

    print(f"\nResults saved to: {summary_file}")

    # Create summary plots
    create_summary_plots_mask(results, output_folder)
