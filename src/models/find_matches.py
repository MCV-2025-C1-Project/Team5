import pickle
import argparse
import os
import numpy as np

from src.distances.bhattacharyya import bhattacharyya_distance
from src.distances.canberra import compute_canberra_distance
from src.distances.chi_2 import compute_chi_2_distance
from src.distances.l1 import compute_l1_distance
from src.distances.correlation import correlation_distance
from src.distances.jensen_shannon import compute_js_divergence
from src.distances.histogram_intersection import compute_histogram_intersection_distance
from src.distances.hellinger import compute_hellinger_distance
from src.distances.euclidean import compute_euclidean_distance
from src.distances.cosine import compute_cosine_distance

from src.tools.startup import logger

# ---------------------------------------------------------------------------
# Map from metric name -> corresponding distance/similarity function
# ---------------------------------------------------------------------------
METRICS = {
    "bhattacharyya": bhattacharyya_distance,
    "canberra": compute_canberra_distance,
    "chi_2": compute_chi_2_distance,
    "l1": compute_l1_distance,
    "correlation": correlation_distance,
    "js_divergence": compute_js_divergence,
    "intersection": compute_histogram_intersection_distance,
    "hellinger": compute_hellinger_distance,
    "euclidean": compute_euclidean_distance,
    "cosine": compute_cosine_distance,
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def load_descriptors(pkl_path):
    """
    Load precomputed descriptors and their corresponding IDs from a .pkl file.

    Parameters
    ----------
    pkl_path : str
        Path to the pickle file containing {'descriptors', 'ids'}.

    Returns
    -------
    descriptors : np.ndarray
        Array of shape (n_images, dim) with image descriptors.
    ids : list
        List of image identifiers (usually filenames or numeric IDs).
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['descriptors'], data['ids']


def find_top_k_matches(query_descs, db_descs, db_ids, metric_func, k=10):
    """
    Find the top-K most similar database images for each query descriptor.

    Parameters
    ----------
    query_descs : np.ndarray
        Descriptors of query images, shape (n_query, dim).
    db_descs : np.ndarray
        Descriptors of database images, shape (n_db, dim).
    db_ids : list
        Identifiers for each database image.
    metric_func : callable
        Function that computes the distance or similarity between two descriptors.
    k : int, optional (default=10)
        Number of best matches to retrieve for each query.

    Returns
    -------
    top_k_ids : list[list]
        List of top-K matching IDs for each query descriptor.
    """
    top_k_ids = []
    logger.info(f"Computing distances using '{metric_func.__name__}'...")
    db_ids_array = np.array(db_ids)

    for q in query_descs:
        # Compute distance/similarity between the query descriptor and all DB descriptors
        distances = np.array([metric_func(q, db) for db in db_descs])

        # Sort indices by increasing distance (or decreasing similarity)
        top_k_indices = np.argsort(distances)[:k]
        top_k_ids.append(list(db_ids_array[top_k_indices]))

    return top_k_ids


def convert_ids_to_int(top_matches):
    """
    Convert image IDs (strings) to integers if possible.

    This helps standardize IDs such as '00023.jpg' -> 23.

    Parameters
    ----------
    top_matches : list[list[str]]
        Nested list of IDs (top-K per query).

    Returns
    -------
    converted : list[list[int]]
        Same structure, but with numeric IDs instead of strings.
    """
    converted = []
    for match_list in top_matches:
        nums = []
        for m in match_list:
            num = int(''.join([c for c in str(m) if c.isdigit()]))
            nums.append(num)
        converted.append(nums)
    return converted

# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Find top-K database matches for query image descriptors."
    )
    parser.add_argument("query_pkl",
                        help="Path to query descriptors .pkl file (e.g., QST1_...pkl)")
    parser.add_argument("db_pkl",
                        help="Path to database descriptors .pkl file (e.g., BBDD_...pkl)")
    parser.add_argument("--metric", required=True, choices=METRICS.keys(),
                        help="Distance/similarity metric to use.")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of top matches to return per query.")
    parser.add_argument("--outdir", default="data/results/week1",
                        help="Base output directory for saving results.")
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Load query and database descriptors
    # --------------------------------------------------------------
    logger.info(f"Loading query descriptors from {args.query_pkl}")
    query_descs, query_ids = load_descriptors(args.query_pkl)

    logger.info(f"Loading database descriptors from {args.db_pkl}")
    db_descs, db_ids = load_descriptors(args.db_pkl)

    # --------------------------------------------------------------
    # Compute top-K matches using the selected metric
    # --------------------------------------------------------------
    metric_function = METRICS[args.metric]
    top_matches = find_top_k_matches(
        query_descs, db_descs, db_ids, metric_function, args.k)

    # Convert image IDs to integer form
    top_matches = convert_ids_to_int(top_matches)

    # --------------------------------------------------------------
    # Build output path dynamically from query file name
    # --------------------------------------------------------------
    descriptor_name = pickle.load(open(args.query_pkl, 'rb'))[
        'descriptor_name']
    method_name = f"{descriptor_name}_{args.metric}"

    # Extract the base name of the query pickle (e.g., "QSD1_descriptors.pkl" → "QSD1_descriptors")
    query_base = os.path.splitext(os.path.basename(args.query_pkl))[0]

    # Create an output directory based on the query name and method
    output_path = os.path.join(args.outdir, method_name)
    os.makedirs(output_path, exist_ok=True)

    # Define output file name based on query name
    output_file = os.path.join(output_path, f"results_{query_base}.pkl")

    # --------------------------------------------------------------
    # Save results as pickle
    # --------------------------------------------------------------
    with open(output_file, 'wb') as f:
        pickle.dump(top_matches, f)

    logger.info(
        f"Execution completed! Saved {len(top_matches)} query results to: {output_file}")
    logger.info("Example result for the first query:")
    logger.info(top_matches[0])


if __name__ == "__main__":
    main()
