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

# Distance functions LOCAL
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
    'dct_grayscale_4x4_32coeffs':    lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='grayscale', grid_size=(4, 4), zigzag_coeffs=32), None),
    'dct_grayscale_8x8_8coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='grayscale', grid_size=(8, 8), zigzag_coeffs=8), None),
    'dct_grayscale_8x8_16coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='grayscale', grid_size=(8, 8), zigzag_coeffs=16), None),
    'dct_grayscale_8x8_32coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='grayscale', grid_size=(8, 8), zigzag_coeffs=32), None),
    'dct_hsv_4x4_8coeffs':    lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(4, 4), zigzag_coeffs=8), None),
    'dct_hsv_4x4_16coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(4, 4), zigzag_coeffs=16), None),
    'dct_hsv_4x4_32coeffs':    lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(4, 4), zigzag_coeffs=32), None),
    'dct_hsv_8x8_8coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(8, 8), zigzag_coeffs=8), None),
    'dct_hsv_8x8_16coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(8, 8), zigzag_coeffs=16), None),
    'dct_hsv_8x8_32coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='hsv', grid_size=(8, 8), zigzag_coeffs=32), None),
    'dct_lab_4x4_8coeffs':    lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='lab', grid_size=(4, 4), zigzag_coeffs=8), None),
    'dct_lab_4x4_16coeffs':   lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='lab', grid_size=(4, 4), zigzag_coeffs=16), None),
    'dct_lab_4x4_32coeffs':    lambda img_path, **kwargs: (dct.compute_block_dct_descriptor(img_path, color_space='lab', grid_size=(4, 4), zigzag_coeffs=32), None),
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
    Unified image retrieval system for global and local features.
    """

    def __init__(self, museum_dir, distance_metric, descriptor_name, mode='global', values_per_bin=1):
        """Initialize the retrieval system and precompute museum features."""
        self.museum_dir = Path(museum_dir)
        self.distance_metric = distance_metric
        self.descriptor_name = descriptor_name
        self.mode = mode
        self.values_per_bin = values_per_bin
        self.query_keypoints = None

        self.museum_features = self._build_database()

    def _extract_museum_id(self, filename: str) -> int:
        """Extract integer museum ID from filename."""
        stem = Path(filename).stem
        if stem.startswith('bbdd_'):
            stem = stem[5:]
        return int(stem)

    def _build_database(self):
        """Compute and store descriptors (global or local) for all museum images."""
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = cache_dir / f"{self.mode}_{self.descriptor_name}.pkl"

        if cache_path.exists():
            logger.info(f"Loading cached features from {cache_path}")
            with open(cache_path, "rb") as f:
                features = pickle.load(f)

            # Rebuild cv2.KeyPoint objects from tuples
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
            
            # Global
            if self.mode == 'global':
                descriptor_func = GLOBAL_DESCRIPTOR_FUNCTIONS[self.descriptor_name]
                desc, _ = descriptor_func(str(img_path), values_per_bin=self.values_per_bin)
                if isinstance(desc, tuple):
                    desc = np.concatenate(desc)
                features[museum_id] = desc

            # Local
            else:
                descriptor_func = LOCAL_DESCRIPTORS_FUNCTIONS[self.descriptor_name]
                keypoints, desc = descriptor_func(str(img_path))
                if desc is not None and len(desc) > 0:
                    features[museum_id] = {"keypoints": keypoints, "descriptors": desc}

        features_serializable = {}

        if self.mode == "local":
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
    
    def retrieve(self, query_path, k=5):
        """Dispatch to the appropriate retrieval method."""
        if self.mode == 'global':
            descriptor_func = GLOBAL_DESCRIPTOR_FUNCTIONS[self.descriptor_name]
            distance_func = DISTANCE_FUNCTIONS_GLOBAL[self.distance_metric]
            return self._retrieve_global(query_path, k, descriptor_func, distance_func)

        else:
            descriptor_func = LOCAL_DESCRIPTORS_FUNCTIONS[self.descriptor_name]
            distance_func = DISTANCE_FUNCTIONS_LOCAL[self.distance_metric]
            return self._retrieve_local(query_path, descriptor_func, distance_func, k)

    def _retrieve_global(self, query_image_path: str, descriptor_func, distance_func, k: int = 5):
        """Compute global descriptor distance-based retrieval."""
        qdest, _ = descriptor_func(
            query_image_path, values_per_bin=self.values_per_bin)

        # Ensure histogram is a numpy array, not a tuple
        if isinstance(qdest, tuple):
            qdest = np.concatenate(qdest)

        distances = []
        for museum_id, mdesc in self.museum_features.items():
            # Ensure museum histogram is also a numpy array
            if isinstance(mdesc, tuple):
                mdesc = np.concatenate(mdesc)
            dist = distance_func(qdest, mdesc)
            distances.append((museum_id, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def _retrieve_local(
        self, query_path, descriptor_func, distance_func,
        k=5, ratio_thresh=0.75, ransac_thresh=10.0, batch_size=5000
    ):
        """
        Perform local descriptor matching with ratio test and RANSAC verification.
        Uses BFMatcher for Euclidean/L1 distances to speed up computation.
        Returns top-k matches by number of inliers.
        """
        kps_q, desc_q = descriptor_func(query_path)
        self.query_keypoints = kps_q
        results, match_data = [], []

        # Decide if we can use OpenCV’s fast BFMatcher
        use_bfmatcher = self.distance_metric in ["euclidean.euclidean_distance", "l1.compute_l1_distance"]

        if use_bfmatcher:
            if self.distance_metric == "euclidean":
                norm_type = cv2.NORM_L2
            else:  # L1 distance
                norm_type = cv2.NORM_L1
            bf = cv2.BFMatcher(norm_type)

        for i, (img_id, data) in enumerate(self.museum_features.items()):
            desc_m = data["descriptors"]
            kps_m = data["keypoints"]

            if desc_q is None or desc_m is None or len(desc_m) == 0:
                results.append((img_id, 0, 0))
                match_data.append({"index": img_id, "good_matches": [], "inlier_matches": []})
                continue

            # Matching 
            if use_bfmatcher:
                # Use OpenCV’s fast KNN matching
                matches_knn = bf.knnMatch(desc_q, desc_m, k=2)
            else:
                # Use our custom distance matrix computation
                dist_matrix = distance_func(desc_q, desc_m, batch_size=batch_size)
                # Handle cases where there are fewer than 2 descriptors
                num_neighbors = min(2, dist_matrix.shape[1])
                if num_neighbors < 2:
                    results.append((img_id, 0, 0))
                    match_data.append({"index": img_id, "good_matches": [], "inlier_matches": []})
                    continue

                sorted_idx = np.argsort(dist_matrix, axis=1)[:, :2]
                sorted_dists = np.take_along_axis(dist_matrix, sorted_idx, axis=1)
                matches_knn = [
                    (
                        cv2.DMatch(_queryIdx=qi, _trainIdx=int(sorted_idx[qi, 0]), _imgIdx=0, _distance=float(sorted_dists[qi, 0])),
                        cv2.DMatch(_queryIdx=qi, _trainIdx=int(sorted_idx[qi, 1]), _imgIdx=0, _distance=float(sorted_dists[qi, 1]))
                    )
                    for qi in range(len(sorted_idx))
                ]

            # Ratio test 
            # good_matches = [m for m, n in matches_knn if m.distance < ratio_thresh * n.distance]
            good_matches = []
            for pair in matches_knn:
                if len(pair) < 2:
                    continue  # skip if less than 2 neighbors found
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            num_tentative = len(good_matches)

            if num_tentative < 4:
                results.append((img_id, num_tentative, 0))
                match_data.append({"index": img_id, "good_matches": good_matches, "inlier_matches": []})
                continue

            # RANSAC geometric verification
            src_pts = np.float32([kps_q[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kps_m[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
            inlier_matches = [m for j, m in enumerate(good_matches) if mask is not None and mask.ravel()[j]]
            num_inliers = len(inlier_matches)

            results.append((img_id, num_tentative, num_inliers))
            match_data.append({
                "index": img_id,
                "good_matches": good_matches,
                "inlier_matches": inlier_matches
            })

        # Sort and return 
        results.sort(key=lambda x: x[2], reverse=True)
        sorted_match_data = [
            next(md for md in match_data if md["index"] == img_id)
            for (img_id, _, _) in results[:k]
        ]
        return results[:k], sorted_match_data

