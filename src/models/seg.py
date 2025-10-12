import numpy as np
import os
import cv2
from scipy.ndimage import (
    binary_opening, binary_closing, binary_fill_holes
)
from typing import Dict, List
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
                min_area: int = 1500) -> np.ndarray:
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


def segment_background(image: np.ndarray,
                                border: int = 20,
                                dist_percentile: float = 92.0,
                                dist_margin: float = 0.5,
                                min_area: int = 1500,
                                opening_size: int = 5,
                                closing_size: int = 7,
                                wL: float = 0.6, wA: float = 1.0, wB: float = 1.0,
                                sat_min: float = 30.0,
                                hue_percentile: float = 92.0,
                                hue_margin_deg: float = 6.0) -> np.ndarray:
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

    Returns:
        np.ndarray: Binary mask where foreground is 1 and background is 0.
    """
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
    border_mask[:b, :] = True; border_mask[-b:, :] = True
    border_mask[:, :b] = True; border_mask[:, -b:] = True

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

            # Run experiment (COLOR-ONLY)
            pred_mask = exp_config['func'](image, **exp_config['params'])

            # Per-image step visualization (NO ARROWS)
            if save_outputs:
                viz_steps_path = os.path.join(exp_output_folder, 'visualizations',
                                              os.path.splitext(img_file)[0] + '_steps.png')
                # Convert BGR to RGB for visualization
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                visualize_color_pipeline_steps_mask(
                    image_rgb,
                    border=exp_config['params'].get('border', 20),
                    dist_percentile=exp_config['params'].get('dist_percentile', 92.0),
                    dist_margin=exp_config['params'].get('dist_margin', 0.5),
                    opening_size=exp_config['params'].get('opening_size', 5),
                    closing_size=exp_config['params'].get('closing_size', 7),
                    min_area=exp_config['params'].get('min_area', 1500),
                    wL=exp_config['params'].get('wL', 0.6),
                    wA=exp_config['params'].get('wA', 1.0),
                    wB=exp_config['params'].get('wB', 1.0),
                    sat_min=exp_config['params'].get('sat_min', 30.0),
                    hue_percentile=exp_config['params'].get('hue_percentile', 92.0),
                    hue_margin_deg=exp_config['params'].get('hue_margin_deg', 6.0),
                    save_path=viz_steps_path
                )

            # Calculate metrics if GT available
            metrics = None
            if gt_mask is not None:
                tp, fp, tn, fn = compute_confusion_counts_mask(pred_mask, gt_mask)

                precision = precision_mask(tp, fp)
                recall = recall_mask(tp, fn)

                f1 = 2*((precision*recall)/(precision+recall))
                print(f"Precision: {precision} | Recal;: {recall} | F1: {f1}")

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
                # Save mask
                mask_filename = os.path.splitext(img_file)[0] + '.png'
                mask_path = os.path.join(exp_output_folder, 'masks', mask_filename)
                cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))

                # Save segmented output (already in BGR format, perfect for cv2.imwrite)
                seg_path = os.path.join(exp_output_folder, 'segmented', img_file)
                cv2.imwrite(seg_path, segmented)

                # Create and save visualization (overall)
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
        description="Color-only background removal and segmentation"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="data/raw/qsd2_w2/",
        help="Path to folder containing input images (default: data/raw/qsd2_w2/)"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results/seg/",
        help="Path to folder for output results (default: results/method_color_only_final/)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: None, process all)"
    )

    args = parser.parse_args()

    print(f"Image folder: {args.image_folder}")
    print(f"Output folder: {args.output_folder}")

    # Configuration
    IMAGE_FOLDER = args.image_folder
    OUTPUT_FOLDER = args.output_folder
    MAX_IMAGES = args.max_images

    # Experiment configuration
    experiments = [{
        'name': '',
        'func': segment_background,
        'params': {
            'border': 20,
            'dist_percentile': 92.0,
            'dist_margin': 0.5,
            'min_area': 1500,
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
