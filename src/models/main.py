from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.descriptors import (grayscale,
                             hsv,
                             lab,
                             rgb,
                             ycbcr,
                             dim2,
                             dim3,
                             spatial_pyramid,
                             block_histogram)
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


# Distance functions
DISTANCE_FUNCTIONS = {
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

# Histogram descriptor functions
DESCRIPTOR_FUNCTIONS = {
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
    'block_histogram_3d_hsv_16x16': lambda img_path, **kwargs: block_histogram.block_based_histogram_3d_hsv(img_path, grid_size=(16, 16), **kwargs)
}


class ComputeImageHistogram:
    def __init__(
        self,
        museum_dir: str,
        distance_metric: str = 'chi_2.compute_chi_2_distance',
        descriptor_type: str = 'grayscale',
        values_per_bin: int = 1
    ):
        """
        Initialize retrieval system.

        Args:
            museum_dir: Directory with museum images
            distance_metric: Distance function name
            descriptor_type: Type of histogram descriptor ('rgb', 'hsv', 'ycbcr', 'lab', 'cielab', 'grayscale')
            values_per_bin: Number of intensity values per histogram bin
        """
        self.museum_dir = Path(museum_dir)
        self.distance_metric = distance_metric
        self.descriptor_type = descriptor_type
        self.values_per_bin = values_per_bin

        # Load museum database
        self.museum_images = sorted(self.museum_dir.glob('*.jpg'))
        self.museum_histograms = {}
        self._build_database()

    def _extract_museum_id(self, filename: str) -> int:
        """Extract integer museum ID from filename."""
        stem = Path(filename).stem
        if stem.startswith('bbdd_'):
            stem = stem[5:]
        return int(stem)

    def _build_database(self):
        """Pre-compute histograms for all museum images."""
        logger.info(
            f"Building BBDD database ({self.descriptor_type}, "
            f"{self.distance_metric}, bins_per_value={self.values_per_bin})...")

        descriptor_func = DESCRIPTOR_FUNCTIONS[self.descriptor_type]

        for img_path in tqdm(self.museum_images):
            hist, _ = descriptor_func(
                str(img_path), values_per_bin=self.values_per_bin)
            # Ensure histogram is a numpy array, not a tuple
            if isinstance(hist, tuple):
                hist = np.concatenate(hist)
            museum_id = self._extract_museum_id(img_path.name)
            self.museum_histograms[museum_id] = hist

    def retrieve(self, query_image_path: str, k: int = 5):
        """Retrieve top-k similar images."""
        descriptor_func = DESCRIPTOR_FUNCTIONS[self.descriptor_type]
        query_hist, _ = descriptor_func(
            query_image_path, values_per_bin=self.values_per_bin)

        # Ensure histogram is a numpy array, not a tuple
        if isinstance(query_hist, tuple):
            query_hist = np.concatenate(query_hist)

        distance_func = DISTANCE_FUNCTIONS[self.distance_metric]

        distances = []
        for museum_id, museum_hist in self.museum_histograms.items():
            # Ensure museum histogram is also a numpy array
            if isinstance(museum_hist, tuple):
                museum_hist = np.concatenate(museum_hist)
            dist = distance_func(query_hist, museum_hist)
            distances.append((museum_id, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]
