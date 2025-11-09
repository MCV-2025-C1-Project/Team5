from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
import pickle

from src.descriptors import (grayscale,
                             hsv,
                             lab,
                             rgb,
                             ycbcr,
                             dim2,
                             dim3,
                             spatial_pyramid,
                             block_histogram,
                             dct,
                             lbp,
                             wavelet,
                             daisy,
                             hog)
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
from src.tools.startup import logger
from src.descriptors.keypoints import dog, harris
from src.descriptors import sift


# Distance functions GLOBAL
DISTANCE_FUNCTIONS_GLOBAL = {
    'euclidean.euclidean_distance': euclidean.compute_euclidean_distance,
    'l1.compute_l1_distance': l1.compute_l1_distance,
    'chi_2.compute_chi_2_distance': chi_2.compute_chi_2_distance,
    'histogram_intersection.compute_histogram_intersection': histogram_intersection.compute_histogram_intersection_distance,
    'hellinger.hellinger_kernel': hellinger.compute_hellinger_distance,
    'cosine.compute_cosine_similarity': cosine.compute_cosine_distance,
    'canberra.canberra_distance': canberra.compute_canberra_distance,
    'bhattacharyya.bhattacharyya_distance': bhattacharyya.bhattacharyya_distance,
    'jensen_shannon.jeffrey_divergence': jensen_shannon.compute_js_divergence,
    'correlation.correlation_distance': correlation.correlation_distance
}

# Distance functions LOCAL (matrix versions)
DISTANCE_FUNCTIONS_LOCAL = {
    'euclidean.euclidean_distance': euclidean.compute_euclidean_distance_matrix,
    'l1.compute_l1_distance': l1.compute_l1_distance_matrix,
    'chi_2.compute_chi_2_distance': chi_2.compute_chi_2_distance_matrix,
    'histogram_intersection.compute_histogram_intersection': histogram_intersection.compute_histogram_intersection_matrix,
    'hellinger.hellinger_kernel': hellinger.compute_hellinger_distance_matrix,
    'cosine.compute_cosine_similarity': cosine.compute_cosine_distance_matrix,
    'canberra.canberra_distance': canberra.compute_canberra_distance_matrix,
    'bhattacharyya.bhattacharyya_distance': bhattacharyya.compute_bhattacharyya_distance_matrix,
    'jensen_shannon.jeffrey_divergence': jensen_shannon.compute_js_divergence_matrix,
    'correlation.correlation_distance': correlation.compute_correlation_distance_matrix
}

# DETECTORS & DESCRIPTORS
KEYPOINT_DETECTORS = {
    'dog_default': lambda img_bgr, **kwargs: dog.detect_dog_keypoints_from_array(
        img_bgr,
        num_scales=5,
        sigma_base=1.6,
        contrast_threshold=0.01,
        edge_threshold=10.0,
        **kwargs
    ),
    'harris_default': lambda img_bgr, **kwargs: harris.detect_harris_keypoints_from_array(
        img_bgr, **kwargs
    ),
    'harris_laplacian_default': lambda img_bgr, **kwargs: harris.detect_harris_laplacian_keypoints_from_array(
        img_bgr, **kwargs
    )
}

LOCAL_DESCRIPTORS_FUNCTIONS = {
    'sift_dog_default': lambda img_path, **kwargs: sift.compute_sift_descriptor(
        img_path,
        keypoint_detector=KEYPOINT_DETECTORS['dog_default'],
        nfeatures=250,
        **kwargs
    ),
    'sift_harris_default': lambda img_path, **kwargs: sift.compute_sift_descriptor(
        img_path,
        keypoint_detector=KEYPOINT_DETECTORS['harris_default'],
        nfeatures=250,
        **kwargs
    ),
    'sift_harris_laplacian_default': lambda img_path, **kwargs: sift.compute_sift_descriptor(
        img_path,
        keypoint_detector=KEYPOINT_DETECTORS['harris_laplacian_default'],
        nfeatures=250,
        **kwargs
    ),
    'hog_dog_default': lambda img_path, **kwargs: hog.compute_hog_descriptor(
        img_path,
        keypoint_detector=KEYPOINT_DETECTORS['dog_default'],
        nfeatures=250,
        **kwargs
    ),
    'daisy_dog_default': lambda img_path, **kwargs: daisy.compute_daisy_descriptor(
        img_path,
        keypoint_detector=KEYPOINT_DETECTORS['dog_default'],
        top_n=250,
        adjust_to_size=False,
        normalization='l2',
        orientations=8,
        visualize=False,
        **kwargs
    ),
}

# Histogram descriptor functions
GLOBAL_DESCRIPTOR_FUNCTIONS = {
    'rgb': rgb.compute_rgb_histogram,
    'hsv': hsv.compute_hsv_histogram,
    'ycbcr': ycbcr.compute_ycbcr_histogram,
    'lab': lab.compute_lab_histogram,
    'grayscale': grayscale.compute_grayscale_histogram,
    '3d_rgb': dim3.compute_3d_histogram_rgb,
    '3d_hsv': dim3.compute_3d_histogram_hsv,
    '3d_lab': dim3.compute_3d_histogram_lab,
    '2d_ycbcr': dim2.compute_2d_histogram_ycbcr,
    '2d_lab': dim2.compute_2d_histogram_lab,
    '2d_hsv': dim2.compute_2d_histogram_hsv,
    'spatial_pyramid_lab': spatial_pyramid.spatial_pyramid_histogram_lab,
    'spatial_pyramid_hsv_lvl2': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_hsv(img_path, levels=2, **kwargs),
    'spatial_pyramid_hsv_lvl3': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_hsv(img_path, levels=3, **kwargs),
    'spatial_pyramid_hsv_lvl4': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_hsv(img_path, levels=4, **kwargs),
    'spatial_pyramid_hsv_lvl5': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_hsv(img_path, levels=5, **kwargs),
    'spatial_pyramid_2d_lab': spatial_pyramid.spatial_pyramid_histogram_2d_lab,
    'spatial_pyramid_2d_hsv_lvl2': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_2d_hsv(img_path, levels=2, **kwargs),
    'spatial_pyramid_2d_hsv_lvl3': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_2d_hsv(img_path, levels=3, **kwargs),
    'spatial_pyramid_2d_hsv_lvl4': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_2d_hsv(img_path, levels=4, **kwargs),
    'spatial_pyramid_2d_hsv_lvl5': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_2d_hsv(img_path, levels=5, **kwargs),
    'spatial_pyramid_3d_lab': spatial_pyramid.spatial_pyramid_histogram_3d_lab,
    'spatial_pyramid_3d_hsv_lvl2': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_3d_hsv(img_path, levels=2, **kwargs),
    'spatial_pyramid_3d_hsv_lvl3': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_3d_hsv(img_path, levels=3, **kwargs),
    'spatial_pyramid_3d_hsv_lvl4': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_3d_hsv(img_path, levels=4, **kwargs),
    'spatial_pyramid_3d_hsv_lvl5': lambda img_path, **kwargs: spatial_pyramid.spatial_pyramid_histogram_3d_hsv(img_path, levels=5, **kwargs),
    'block_histogram_lab': block_histogram.block_based_histogram_lab,
    'block_histogram_hsv_2x2': lambda img_path, **kwargs: block_histogram.block_based_histogram_hsv(img_path, grid_size=(2, 2), **kwargs),
    'block_histogram_hsv_4x4': lambda img_path, **kwargs: block_histogram.block_based_histogram_hsv(img_path, grid_size=(4, 4), **kwargs),
    'block_histogram_hsv_8x8': lambda img_path, **kwargs: block_histogram.block_based_histogram_hsv(img_path, grid_size=(8, 8), **kwargs),
    'block_histogram_hsv_16x16': lambda img_path, **kwargs: block_histogram.block_based_histogram_hsv(img_path, grid_size=(16, 16), **kwargs),
    'block_histogram_2d_lab': block_histogram.block_based_histogram_2d_lab,
    'block_histogram_2d_hsv_2x2': lambda img_path, **kwargs: block_histogram.block_based_histogram_2d_hsv(img_path, grid_size=(2, 2), **kwargs),
    'block_histogram_2d_hsv_4x4': lambda img_path, **kwargs: block_histogram.block_based_histogram_2d_hsv(img_path, grid_size=(4, 4), **kwargs),
    'block_histogram_2d_hsv_8x8': lambda img_path, **kwargs: block_histogram.block_based_histogram_2d_hsv(img_path, grid_size=(8, 8), **kwargs),
    'block_histogram_2d_hsv_16x16': lambda img_path, **kwargs: block_histogram.block_based_histogram_2d_hsv(img_path, grid_size=(16, 16), **kwargs),
    'block_histogram_3d_lab': block_histogram.block_based_histogram_3d_lab,
    'block_histogram_3d_hsv_2x2': lambda img_path, **kwargs: block_histogram.block_based_histogram_3d_hsv(img_path, grid_size=(2, 2), **kwargs),
    'block_histogram_3d_hsv_4x4': lambda img_path, **kwargs: block_histogram.block_based_histogram_3d_hsv(img_path, grid_size=(4, 4), **kwargs),
    'block_histogram_3d_hsv_8x8': lambda img_path, **kwargs: block_histogram.block_based_histogram_3d_hsv(img_path, grid_size=(8, 8), **kwargs),
    'block_histogram_3d_hsv_16x16': lambda img_path, **kwargs: block_histogram.block_based_histogram_3d_hsv(img_path, grid_size=(16, 16), **kwargs),
    'dct_grayscale_4x4_8coeffs':    lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='grayscale', grid_size=(4, 4), zigzag_coeffs=8), None),
    'dct_grayscale_4x4_16coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='grayscale', grid_size=(4, 4), zigzag_coeffs=16), None),
    'dct_grayscale_4x4_32coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='grayscale', grid_size=(4, 4), zigzag_coeffs=32), None),
    'dct_grayscale_8x8_8coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='grayscale', grid_size=(8, 8), zigzag_coeffs=8), None),
    'dct_grayscale_8x8_16coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='grayscale', grid_size=(8, 8), zigzag_coeffs=16), None),
    'dct_grayscale_8x8_32coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='grayscale', grid_size=(8, 8), zigzag_coeffs=32), None),
    'dct_hsv_4x4_8coeffs':    lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(4, 4), zigzag_coeffs=8), None),
    'dct_hsv_4x4_16coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(4, 4), zigzag_coeffs=16), None),
    'dct_hsv_4x4_32coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(4, 4), zigzag_coeffs=32), None),
    'dct_hsv_8x8_8coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(8, 8), zigzag_coeffs=8), None),
    'dct_hsv_8x8_16coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(8, 8), zigzag_coeffs=16), None),
    'dct_hsv_8x8_32coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(8, 8), zigzag_coeffs=32), None),
    'dct_lab_4x4_8coeffs':    lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='lab', grid_size=(4, 4), zigzag_coeffs=8), None),
    'dct_lab_4x4_16coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='lab', grid_size=(4, 4), zigzag_coeffs=16), None),
    'dct_lab_4x4_32coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='lab', grid_size=(4, 4), zigzag_coeffs=32), None),
    'dct_lab_8x8_8coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='lab', grid_size=(8, 8), zigzag_coeffs=8), None),
    'dct_lab_8x8_16coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='lab', grid_size=(8, 8), zigzag_coeffs=16), None),
    'dct_lab_8x8_32coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='lab', grid_size=(8, 8), zigzag_coeffs=32), None),
    'lbp_gray_s1_4x4':   lambda img_path, **kwargs: (lbp.compute_lbp_descriptor(img_path, color_space='grayscale', grid_size=(4, 4), scales=[(1, 8)]), None),
    'lbp_gray_ms2_4x4':  lambda img_path, **kwargs: (lbp.compute_lbp_descriptor(img_path, color_space='grayscale', grid_size=(4, 4), scales=[(1, 8), (3, 24)]), None),
    'lbp_gray_ms2_8x8':  lambda img_path, **kwargs: (lbp.compute_lbp_descriptor(img_path, color_space='grayscale', grid_size=(8, 8), scales=[(1, 8), (3, 24)]), None),
    'lbp_lab_ms2_4x4':   lambda img_path, **kwargs: (lbp.compute_lbp_descriptor(img_path, color_space='lab', grid_size=(4, 4), scales=[(1, 8), (3, 24)]), None),
    'lbp_lab_ms2_8x8':   lambda img_path, **kwargs: (lbp.compute_lbp_descriptor(img_path, color_space='lab', grid_size=(8, 8), scales=[(1, 8), (3, 24)]), None),
    'lbp_hsv_ms2_4x4':   lambda img_path, **kwargs: (lbp.compute_lbp_descriptor(img_path, color_space='hsv', grid_size=(4, 4), scales=[(1, 8), (3, 24)]), None),
    'lbp_hsv_ms2_8x8':   lambda img_path, **kwargs: (lbp.compute_lbp_descriptor(img_path, color_space='hsv', grid_size=(8, 8), scales=[(1, 8), (3, 24)]), None),
    'haar_grayscale_lvl1':   lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='grayscale', levels=1), None),
    'haar_grayscale_lvl2':   lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='grayscale', levels=2), None),
    'haar_grayscale_lvl3':   lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='grayscale', levels=3), None),
    'haar_hsv_lvl1':         lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='hsv', levels=1), None),
    'haar_hsv_lvl2':         lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='hsv', levels=2), None),
    'haar_hsv_lvl3':         lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='hsv', levels=3), None),
    'bior44_grayscale_lvl1':   lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='grayscale', wavelet='bior4.4', levels=1), None),
    'bior44_grayscale_lvl2':   lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='grayscale', wavelet='bior4.4', levels=2), None),
    'bior44_grayscale_lvl3':   lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='grayscale', wavelet='bior4.4', levels=3), None),
    'bior44_hsv_lvl1':         lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='hsv', wavelet='bior4.4', levels=1), None),
    'bior44_hsv_lvl2':         lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='hsv', wavelet='bior4.4', levels=2), None),
    'bior44_hsv_lvl3':         lambda img_path, **kwargs: (wavelet.compute_dwt_descriptor(img_path, color_space='hsv', wavelet='bior4.4', levels=3), None),
    'block_haar_grayscale_4x4_lvl1':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', grid_size=(4, 4), levels=1), None),
    'block_haar_grayscale_8x8_lvl1':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', grid_size=(8, 8), levels=1), None),
    'block_haar_grayscale_4x4_lvl2':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', grid_size=(4, 4), levels=2), None),
    'block_haar_grayscale_8x8_lvl2':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', grid_size=(8, 8), levels=2), None),
    'block_haar_grayscale_4x4_lvl3':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', grid_size=(4, 4), levels=3), None),
    'block_haar_grayscale_8x8_lvl3':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', grid_size=(8, 8), levels=3), None),
    'block_haar_hsv_4x4_lvl1':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', grid_size=(4, 4), levels=1), None),
    'block_haar_hsv_8x8_lvl1':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', grid_size=(8, 8), levels=1), None),
    'block_haar_hsv_4x4_lvl2':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', grid_size=(4, 4), levels=2), None),
    'block_haar_hsv_8x8_lvl2':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', grid_size=(8, 8), levels=2), None),
    'block_haar_hsv_4x4_lvl3':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', grid_size=(4, 4), levels=3), None),
    'block_haar_hsv_8x8_lvl3':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', grid_size=(8, 8), levels=3), None),
    'block_bior44_grayscale_4x4_lvl1':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', wavelet='bior4.4', grid_size=(4, 4), levels=1), None),
    'block_bior44_grayscale_8x8_lvl1':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', wavelet='bior4.4', grid_size=(8, 8), levels=1), None),
    'block_bior44_grayscale_4x4_lvl2':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', wavelet='bior4.4', grid_size=(4, 4), levels=2), None),
    'block_bior44_grayscale_8x8_lvl2':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', wavelet='bior4.4', grid_size=(8, 8), levels=2), None),
    'block_bior44_grayscale_4x4_lvl3':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', wavelet='bior4.4', grid_size=(4, 4), levels=3), None),
    'block_bior44_grayscale_8x8_lvl3':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='grayscale', wavelet='bior4.4', grid_size=(8, 8), levels=3), None),
    'block_bior44_hsv_4x4_lvl1':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', wavelet='bior4.4', grid_size=(4, 4), levels=1), None),
    'block_bior44_hsv_8x8_lvl1':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', wavelet='bior4.4', grid_size=(8, 8), levels=1), None),
    'block_bior44_hsv_4x4_lvl2':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', wavelet='bior4.4', grid_size=(4, 4), levels=2), None),
    'block_bior44_hsv_8x8_lvl2':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', wavelet='bior4.4', grid_size=(8, 8), levels=2), None),
    'block_bior44_hsv_4x4_lvl3':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', wavelet='bior4.4', grid_size=(4, 4), levels=3), None),
    'block_bior44_hsv_8x8_lvl3':    lambda img_path, **kwargs: (wavelet.compute_block_dwt_descriptor(img_path, color_space='hsv', wavelet='bior4.4', grid_size=(8, 8), levels=3), None)
}


class ComputeImageFeatures:
    """
    Unified image retrieval system for global and local features, including unknown detection.

    Unknown detection is performed by:
    - Bidirectional (cross-checked) matching after ratio test
    - Threshold on number of bidirectional matches
    - Ambiguity check between top-2 candidates
    - RANSAC geometric verification to count inliers
    """

    def __init__(self, museum_dir, distance_metric, descriptor_name,
                 mode='global', values_per_bin=1,
                 min_matches_threshold=10, ratio_threshold=0.75,
                 ransac_thresh=10.0, ambiguity_ratio=1.3,
                 ground_truth=None):
        self.museum_dir = Path(museum_dir)
        self.distance_metric = distance_metric
        self.descriptor_name = descriptor_name
        self.mode = mode
        self.values_per_bin = values_per_bin

        # Unknown detection / verification thresholds
        self.min_matches_threshold = min_matches_threshold
        self.ratio_threshold = ratio_threshold
        self.ransac_thresh = ransac_thresh
        self.ambiguity_ratio = ambiguity_ratio

        self.query_keypoints = None

        # Optional ground-truth loading (not required)
        self.ground_truth = None
        if ground_truth:
            try:
                with open(ground_truth, 'rb') as f:
                    self.ground_truth = pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load ground_truth: {e}")

        self.museum_features = self._build_database()

    # ----------------------- PUBLIC API -----------------------

    def retrieve(self, query_path, k=5):
        """
        Returns:
            known: (results[:k], match_data_for_top_k)
            unknown: ([(-1, 0, 0)], [])
        """
        if self.mode == 'global':
            descriptor_func = GLOBAL_DESCRIPTOR_FUNCTIONS[self.descriptor_name]
            distance_func = DISTANCE_FUNCTIONS_GLOBAL[self.distance_metric]
            results = self._retrieve_global(query_path, k, descriptor_func, distance_func)
            # global branch does not run unknown detection heuristics (hist distance only)
            return results, []
        else:
            descriptor_func = LOCAL_DESCRIPTORS_FUNCTIONS[self.descriptor_name]
            distance_func = DISTANCE_FUNCTIONS_LOCAL[self.distance_metric]
            return self._retrieve_local(query_path, descriptor_func, distance_func, k)

    # ----------------------- GLOBAL RETRIEVAL -----------------------

    def _retrieve_global(self, query_image_path: str, k: int, descriptor_func, distance_func):
        qdest, _ = descriptor_func(query_image_path, values_per_bin=self.values_per_bin)
        if isinstance(qdest, tuple):
            qdest = np.concatenate(qdest)

        distances = []
        for museum_id, mdesc in self.museum_features.items():
            if isinstance(mdesc, tuple):
                mdesc = np.concatenate(mdesc)
            dist = distance_func(qdest, mdesc)
            distances.append((museum_id, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    # ----------------------- LOCAL RETRIEVAL -----------------------

    def _retrieve_local(self, query_path, descriptor_func, distance_func, k=5, batch_size=5000):
        """
        Perform local descriptor matching with:
          - Euclidean/L1: BFMatcher (fast) forward & backward
          - Others: manual distance matrices to build KNN forward & backward
          - Ratio test (self.ratio_threshold)
          - Bidirectional cross-check
          - RANSAC geometric verification (count inliers)
          - Unknown detection (min matches, ambiguity, inliers)
        """
        kps_q, desc_q = descriptor_func(query_path)
        self.query_keypoints = kps_q
        if desc_q is None or len(desc_q) == 0:
            logger.warning(f"No descriptors for query: {query_path}")
            return ([(-1, 0, 0)], [])

        # Can we use BFMatcher?
        metric_key = self.distance_metric
        use_bfmatcher = metric_key in ("euclidean.euclidean_distance", "l1.compute_l1_distance")

        if use_bfmatcher:
            norm_type = cv2.NORM_L2 if metric_key == "euclidean.euclidean_distance" else cv2.NORM_L1
            bf = cv2.BFMatcher(norm_type, crossCheck=False)

        results = []
        match_data = []

        for img_id, data in self.museum_features.items():
            desc_m = data["descriptors"]
            kps_m = data["keypoints"]
            if desc_m is None or len(desc_m) == 0:
                continue

            # ---------- KNN MATCHES (FORWARD & BACKWARD) ----------
            if use_bfmatcher:
                try:
                    matches_forward = bf.knnMatch(desc_q, desc_m, k=2)  # q -> m
                    matches_backward = bf.knnMatch(desc_m, desc_q, k=2) # m -> q
                except cv2.error:
                    continue
            else:
                # Manual forward distances: (n_q x n_m)
                dist_qm = distance_func(desc_q, desc_m, batch_size=batch_size)
                # Manual backward distances: (n_m x n_q)
                dist_mq = distance_func(desc_m, desc_q, batch_size=batch_size)

                # Build K=2 neighbors for each row
                def manual_knn(dist_matrix):
                    # idx sorted by distance
                    k_neighbors = min(2, dist_matrix.shape[1])
                    sorted_idx = np.argsort(dist_matrix, axis=1)[:, :k_neighbors]
                    sorted_dists = np.take_along_axis(dist_matrix, sorted_idx, axis=1)
                    # Build OpenCV-like KNN match pairs list
                    pairs = []
                    for qi in range(sorted_idx.shape[0]):
                        if k_neighbors == 1:
                            # can't apply ratio; just put one
                            m = cv2.DMatch(_queryIdx=qi,
                                           _trainIdx=int(sorted_idx[qi, 0]),
                                           _imgIdx=0,
                                           _distance=float(sorted_dists[qi, 0]))
                            pairs.append([m])  # single neighbor
                        else:
                            m0 = cv2.DMatch(_queryIdx=qi,
                                            _trainIdx=int(sorted_idx[qi, 0]),
                                            _imgIdx=0,
                                            _distance=float(sorted_dists[qi, 0]))
                            m1 = cv2.DMatch(_queryIdx=qi,
                                            _trainIdx=int(sorted_idx[qi, 1]),
                                            _imgIdx=0,
                                            _distance=float(sorted_dists[qi, 1]))
                            pairs.append([m0, m1])
                    return pairs

                matches_forward = manual_knn(dist_qm)  # q -> m
                matches_backward = manual_knn(dist_mq) # m -> q

            # ---------- RATIO TEST ----------
            good_forward = {}   # query_idx -> (museum_idx, dist)
            for pair in matches_forward:
                if len(pair) == 0:
                    continue
                if len(pair) == 1:
                    # keep single neighbor (no ratio possible) — conservative: skip
                    continue
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_forward[m.queryIdx] = (m.trainIdx, m.distance)

            good_backward = {}  # museum_idx -> (query_idx, dist)
            for pair in matches_backward:
                if len(pair) == 0:
                    continue
                if len(pair) == 1:
                    continue
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_backward[m.queryIdx] = (m.trainIdx, m.distance)

            # ---------- BIDIRECTIONAL CROSS-CHECK ----------
            bidir_matches = []
            for q_idx, (m_idx, dval) in good_forward.items():
                if m_idx in good_backward:
                    back_q_idx, _ = good_backward[m_idx]
                    if back_q_idx == q_idx:
                        bidir_matches.append(cv2.DMatch(
                            _queryIdx=q_idx, _trainIdx=m_idx, _imgIdx=0, _distance=float(dval)
                        ))

            num_bidir = len(bidir_matches)
            if num_bidir == 0:
                continue

            # ---------- RANSAC GEOMETRIC VERIFICATION ----------
            num_bidir = len(bidir_matches)

            if num_bidir < 4:
                # Not enough correspondences to estimate a homography; treat as zero inliers
                inlier_matches = []
                num_inliers = 0
            else:
                src_pts = np.float32([kps_q[m.queryIdx].pt for m in bidir_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kps_m[m.trainIdx].pt for m in bidir_matches]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
                inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros((num_bidir,), dtype=bool)
                inlier_matches = [m for m, keep in zip(bidir_matches, inlier_mask) if keep]
                num_inliers = len(inlier_matches)

            avg_distance = float(np.mean([m.distance for m in bidir_matches])) if bidir_matches else float('inf')

            # Accumulate (sort by inliers first, then bidir count)
            results.append((img_id, num_inliers, num_bidir))
            match_data.append({
                "index": img_id,
                "good_matches": bidir_matches,
                "inlier_matches": inlier_matches,
                "avg_distance": avg_distance
            })


        # ---------- UNKNOWN DECISION ----------
        if len(results) == 0:
            return ([(-1, 0, 0)], [])

        # sort primarily by inliers desc, then by bidirectional count desc
        results.sort(key=lambda x: (x[1], x[2]), reverse=True)

        best_inliers = results[0][1]
        best_bidir  = results[0][2]

        # Threshold A: enough bidirectional matches
        if best_bidir < self.min_matches_threshold:
            return ([(-1, 0, 0)], [])

        # Threshold B: ambiguity between top-2 (based on inliers)
        if len(results) >= 2:
            second_inliers = results[1][1]
            ratio = (best_inliers / max(second_inliers, 1))
            if ratio < self.ambiguity_ratio:
                return ([(-1, 0, 0)], [])

        # Threshold C: enough inliers after RANSAC (use same threshold as min_matches or stricter)
        if best_inliers < max(4, int(0.4 * self.min_matches_threshold)):
            # require at least 4 for homography, and a fraction of min_matches
            return ([(-1, 0, 0)], [])

        # ---------- PACK TOP-K ----------
        topk = results[:k]
        sorted_match_data = [
            next(md for md in match_data if md["index"] == img_id)
            for (img_id, _, _) in topk
        ]
        return topk, sorted_match_data

    # ----------------------- DB BUILD & CACHE -----------------------

    def _build_database(self):
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{self.mode}_{self.descriptor_name}.pkl"

        if cache_path.exists():
            logger.info(f"Loading cached features from {cache_path}")
            with open(cache_path, "rb") as f:
                features = pickle.load(f)
            if self.mode == "local":
                for museum_id, data in features.items():
                    kp_tuples = data["keypoints"]
                    data["keypoints"] = [
                        cv2.KeyPoint(
                            x=pt[0][0],
                            y=pt[0][1],
                            size=pt[1],
                            angle=pt[2],
                            response=pt[3],
                            octave=pt[4],
                            class_id=pt[5]
                        )
                        for pt in kp_tuples
                    ]
            return features

        features = {}
        for img_path in tqdm(sorted(self.museum_dir.glob("*.jpg"))):
            museum_id = self._extract_museum_id(img_path.name)

            if self.mode == 'global':
                descriptor_func = GLOBAL_DESCRIPTOR_FUNCTIONS[self.descriptor_name]
                desc, _ = descriptor_func(str(img_path), values_per_bin=self.values_per_bin)
                if isinstance(desc, tuple):
                    desc = np.concatenate(desc)
                features[museum_id] = desc
            else:
                descriptor_func = LOCAL_DESCRIPTORS_FUNCTIONS[self.descriptor_name]
                keypoints, desc = descriptor_func(str(img_path))
                if desc is not None and len(desc) > 0:
                    features[museum_id] = {"keypoints": keypoints, "descriptors": desc}

        # Serialize keypoints for local mode
        if self.mode == "local":
            features_serializable = {}
            for museum_id, data in features.items():
                kp_serializable = [
                    (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                    for kp in data["keypoints"]
                ]
                features_serializable[museum_id] = {
                    "keypoints": kp_serializable,
                    "descriptors": data["descriptors"]
                }
        else:
            features_serializable = features

        with open(cache_path, "wb") as f:
            pickle.dump(features_serializable, f)

        return features

    def _extract_museum_id(self, filename: str) -> int:
        stem = Path(filename).stem
        if stem.startswith('bbdd_'):
            stem = stem[5:]
        return int(stem)
