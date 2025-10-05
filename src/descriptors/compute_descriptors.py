import os
import glob
import argparse
import pickle
import numpy as np

from src.data.extract import read_image

from src.descriptors.grayscale import compute_grayscale_histogram
from src.descriptors.hsv import compute_hsv_histogram
from src.descriptors.lab import compute_lab_histogram
from src.descriptors.rgb import compute_rgb_histogram
from src.descriptors.ycbcr import compute_ycbcr_histogram

from src.tools.startup import logger

# Map a short name -> (callable, default params)
DESCRIPTORS = {
    "grayscale": (compute_grayscale_histogram, {"values_per_bin": 1}),
    "hsv":       (compute_hsv_histogram,       {"values_per_bin": 1}),
    "lab":       (compute_lab_histogram,       {"values_per_bin": 1}),
    "rgb":       (compute_rgb_histogram,       {"values_per_bin": 1}),
    "ycbcr":     (compute_ycbcr_histogram,     {"values_per_bin": 1}),
}

# Only process real images (ignore .png masks & metadata) IMG_EXTS = (".jpg", ".jpeg")
IMG_EXTS = (".jpg", ".jpeg")


def list_images(folder: str):
    """Return sorted list of image paths with allowed extensions."""
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


def compute_for_folder(folder: str, func, **kwargs):
    """
    Compute descriptors for all images in a given folder.

    Parameters
    ----------
    folder : str
        Path to the folder containing images.
    func : callable
        Descriptor function. Must accept an image (BGR np.ndarray) and return
        a 1-D descriptor array.
    **kwargs :
        Extra keyword arguments passed to the descriptor function (e.g. bins).

    Returns
    -------
    descs : np.ndarray, shape (n_images, dim)
        Stack of descriptors for all images in the folder.
    ids : list
        List of image IDs (integers if filenames are numeric, otherwise strings)
        corresponding to each row of `descs`.
    """
    paths = list_images(folder)  # get all image file paths (.jpg/.jpeg)
    descs, ids = [], []

    for p in paths:

        hist, _ = func(p, **kwargs)
        descs.append(hist.astype(np.float32))  # store descriptor as float32

        # extract the filename without extension to use as image ID
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            # convert to int if possible (e.g. "00023" -> 23)
            ids.append(int(stem))
        except ValueError:
            ids.append(stem)  # otherwise keep as string

    # stack all descriptors into a single 2-D array [n_images, descriptor_dim]
    return np.vstack(descs), ids


def main():
    ap = argparse.ArgumentParser(
        description="Compute 1D image descriptors."
    )
    ap.add_argument("--descriptor", required=True, choices=DESCRIPTORS.keys(),
                    help="Which descriptor to run: grayscale | hsv | lab | rgb | ycbcr")
    ap.add_argument("--input", required=True,
                    help="Folder with images (BBDD, QSD1, QST1)")
    ap.add_argument("--outdir", default="data/descriptors",
                    help="Output folder")
    # optional overrides
    ap.add_argument("--values_per_bin", type=int, default=None,
                    help="Intensity values per bin")
    args = ap.parse_args()

    func, defaults = DESCRIPTORS[args.descriptor]
    params = defaults.copy()
    if args.values_per_bin is not None:
        params["values_per_bin"] = args.values_per_bin

    logger.info(f"Beginning computation of descriptor '{args.descriptor}' to data in folder '{args.input}'."
                f" Values per bin: {params['values_per_bin']}")

    os.makedirs(args.outdir, exist_ok=True)

    H, ids = compute_for_folder(args.input, func, **params)

    # Build a tag for the filename
    tag = f"{args.descriptor}_vpb{params['values_per_bin']}"
    dataset_name = os.path.basename(os.path.normpath(args.input))
    # Save everything into one pickle
    out_file = os.path.join(args.outdir, f"{dataset_name}_{tag}.pkl")

    data_to_save = {
        "descriptors": H,   # numpy array [n_images, dim]
        "ids": ids,         # list of IDs
        "params": params,   # optional: store the parameters used
        "descriptor_name": args.descriptor,
        "dataset_name": dataset_name,
    }

    with open(out_file, "wb") as f:
        pickle.dump(data_to_save, f)

    logger.info(
        f"Execution completed! Saved {H.shape} descriptors to {out_file}")


if __name__ == "__main__":
    main()
