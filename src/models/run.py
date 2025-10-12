import argparse
from pathlib import Path

from src.visualization.viz import generate_comprehensive_analysis
from src.tools.startup import logger


def main():
    parser = argparse.ArgumentParser(
        description='Image Retrieval System - Comprehensive Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--query_dir',
        type=str,
        required=True,
        help='Path to query images directory (e.g., qsd1_w1)'
    )

    parser.add_argument(
        '--museum_dir',
        type=str,
        required=True,
        help='Path to museum database directory (e.g., BBDD)'
    )

    parser.add_argument(
        '--ground_truth',
        type=str,
        required=True,
        help='Path to ground truth pickle file (e.g., gt_corresps.pkl)'
    )

    # Optional arguments
    parser.add_argument(
        '--values_per_bin',
        type=int,
        default=1,
        help='Number of intensity values per histogram bin (1=256 bins, 2=128 bins, etc.)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of top-k retrievals to save'
    )

    parser.add_argument(
        '--descriptors',
        type=str,
        nargs='+',
        default=None,
        choices=['rgb', 'hsv', 'ycbcr', 'lab', 'grayscale', 'hsv', 'lab',
                 '3d_rgb', '3d_hsv', '3d_lab', '2d_ycbcr', '2d_lab', '2d_hsv',
                 'spatial_pyramid_lab', 'spatial_pyramid_hsv_lvl2', 'spatial_pyramid_hsv_lvl3',
                 'spatial_pyramid_hsv_lvl4', 'spatial_pyramid_hsv_lvl5',
                 'spatial_pyramid_2d_lab', 'spatial_pyramid_2d_hsv_lvl2',
                 'spatial_pyramid_2d_hsv_lvl3', 'spatial_pyramid_2d_hsv_lvl4',
                 'spatial_pyramid_2d_hsv_lvl5', 'spatial_pyramid_3d_lab', 'spatial_pyramid_3d_hsv',
                 'block_histogram_lab', 'block_histogram_hsv_2x2', 'block_histogram_hsv_4x4',
                 'block_histogram_hsv_8x8', 'block_histogram_hsv_16x16',
                 'block_histogram_2d_lab', 'block_histogram_2d_hsv_2x2',
                 'block_histogram_2d_hsv_4x4', 'block_histogram_2d_hsv_8x8',
                 'block_histogram_2d_hsv_16x16', 'block_histogram_3d_lab', 'block_histogram_3d_hsv'],
        help='Specific descriptors to evaluate (default: all). Multiple allowed.'
    )

    parser.add_argument(
        '--distances',
        type=str,
        nargs='+',
        default=None,
        choices=['euclidean.euclidean_distance', 'l1.compute_l1_distance', 'chi_2.compute_chi_2_distance',
                 'histogram_intersection.compute_histogram_intersection', 'hellinger.hellinger_kernel',
                 'cosine.compute_cosine_similarity', 'canberra.canberra_distance',
                 'bhattacharyya.bhattacharyya_distance', 'jensen_shannon.jeffrey_divergence',
                 'correlation.correlation_distance'],
        help='Specific distance metrics to evaluate (default: all). Multiple allowed.'
    )

    args = parser.parse_args()

    # Validate paths
    query_path = Path(args.query_dir)
    museum_path = Path(args.museum_dir)
    gt_path = Path(args.ground_truth)

    if not query_path.exists():
        raise FileNotFoundError(f"Query directory not found: {args.query_dir}")
    if not museum_path.exists():
        raise FileNotFoundError(
            f"Museum directory not found: {args.museum_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {args.ground_truth}")

    logger.info("=" * 80)
    logger.info("IMAGE RETRIEVAL SYSTEM - COMPREHENSIVE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Query Directory:     {args.query_dir}")
    logger.info(f"Museum Directory:    {args.museum_dir}")
    logger.info(f"Ground Truth:        {args.ground_truth}")
    logger.info(f"Values per bin:      {args.values_per_bin}")
    logger.info(f"Output Directory:    {args.output_dir}")
    logger.info(f"Top-k:               {args.k}")

    if args.descriptors:
        logger.info(f"Descriptors:         {', '.join(args.descriptors)}")
    else:
        logger.info(
            f"Descriptors:         All (rgb, hsv, ycbcr, lab, grayscale)")

    if args.distances:
        logger.info(f"Distance Metrics:    {len(args.distances)} selected")
    else:
        logger.info(f"Distance Metrics:    All (10 metrics)")

    # Calculate total combinations
    num_descriptors = len(args.descriptors) if args.descriptors else 5
    num_distances = len(args.distances) if args.distances else 10
    total_combinations = num_descriptors * num_distances
    logger.info(f"Total Combinations:  {total_combinations}")
    logger.info("=" * 80)

    # Run comprehensive analysis
    all_results, best_desc, best_dist, best_score, result_list = generate_comprehensive_analysis(
        args.query_dir,
        args.museum_dir,
        args.ground_truth,
        values_per_bin=args.values_per_bin,
        output_dir=args.output_dir,
        k=args.k,
        descriptors=args.descriptors,
        distance_metrics=args.distances
    )

    logger.info(f"Best Descriptor: {best_desc}")
    logger.info(f"Best Distance:   {best_dist}")
    logger.info(f"Best mAP@5:      {best_score:.4f}")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
