import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.descriptors import grayscale, hsv, lab, rgb, ycbcr
from src.distances import bhattacharyya, canberra, chi_2, correlation, cosine, euclidean, hellinger, histogram_intersection, jensen_shannon, l1
from src.tools.startup import logger


class ComputeImageHistogram:
    def __init__(self, museum_dir: str, distance_metric: str = 'chi_2.compute_chi_2_distance',
                 descriptor_type: str = 'grayscale', values_per_bin: int = 1):
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

        # Distance functions
        self.distance_functions = {
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
        self.descriptor_functions = {
            'rgb': rgb.compute_rgb_histogram,
            'hsv': hsv.compute_hsv_histogram,
            'ycbcr': ycbcr.compute_ycbcr_histogram,
            'lab': lab.compute_lab_histogram,
            'grayscale': grayscale.compute_grayscale_histogram
        }

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
            f"Building BBDD database ({self.descriptor_type}, {self.distance_metric}, bins_per_value={self.values_per_bin})...")

        descriptor_func = self.descriptor_functions[self.descriptor_type]

        for img_path in tqdm(self.museum_images):
            hist = descriptor_func(
                str(img_path), values_per_bin=self.values_per_bin)
            # Ensure histogram is a numpy array, not a tuple
            if isinstance(hist, tuple):
                hist = np.concatenate(hist)
            museum_id = self._extract_museum_id(img_path.name)
            self.museum_histograms[museum_id] = hist

    def retrieve(self, query_image_path: str, k: int = 5):
        """Retrieve top-k similar images."""
        descriptor_func = self.descriptor_functions[self.descriptor_type]
        query_hist = descriptor_func(
            query_image_path, values_per_bin=self.values_per_bin)

        # Ensure histogram is a numpy array, not a tuple
        if isinstance(query_hist, tuple):
            query_hist = np.concatenate(query_hist)

        distance_func = self.distance_functions[self.distance_metric]

        distances = []
        for museum_id, museum_hist in self.museum_histograms.items():
            # Ensure museum histogram is also a numpy array
            if isinstance(museum_hist, tuple):
                museum_hist = np.concatenate(museum_hist)
            dist = distance_func(query_hist, museum_hist)
            distances.append((museum_id, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]
