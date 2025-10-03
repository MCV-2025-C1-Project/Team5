import os, glob, argparse, pickle
import numpy as np
import cv2

from data.extract import read_image

from descriptors.gray_level_histogram import compute_histogram as descriptor_gray_level
from descriptors.hsv_hue_weighted_histogram import compute_histogram as descriptor_hsv_hue_weighted
from descriptors.lab_chroma_histogram import compute_histogram as descriptor_cielab_chroma

# Map a short name -> (callable, default params)
DESCRIPTORS = {
    "gray_level": (descriptor_gray_level,        {}),
    "hsv_hueS":  (descriptor_hsv_hue_weighted, {"bins": 72, "s_min": 0.10}),
    "lab_croma": (descriptor_cielab_chroma,   {"bins": 64, "c_max": 150.0}),
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
        img = read_image(p)  # read the image in BGR (OpenCV default)
        if img is None:
            print(f"[WARN] Could not read {p}")
            continue

        # compute the descriptor for this image
        hist, bin_edges = func(img, **kwargs)
        descs.append(hist.astype(np.float32))  # store descriptor as float32

        # extract the filename without extension to use as image ID
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            ids.append(int(stem))  # convert to int if possible (e.g. "00023" -> 23)
        except ValueError:
            ids.append(stem)       # otherwise keep as string

    # stack all descriptors into a single 2-D array [n_images, descriptor_dim]
    return np.vstack(descs), ids

def main():
    ap = argparse.ArgumentParser(
        description="Compute 1D image descriptors."
    )
    ap.add_argument("--descriptor", required=True, choices=DESCRIPTORS.keys(),
                    help="Which descriptor to run: gray_level | hsv_hueS | lab_croma")
    ap.add_argument("--input", required=True,
                    help="Folder with images (BBDD, QSD1, QST1)")
    ap.add_argument("--outdir", default="data/descriptors",
                    help="Output folder")
    # optional overrides
    ap.add_argument("--bins",  type=int,   default=None)
    ap.add_argument("--s_min", type=float, default=None)
    ap.add_argument("--c_max", type=float, default=None)
    args = ap.parse_args()

    func, defaults = DESCRIPTORS[args.descriptor]
    params = defaults.copy()
    if args.bins  is not None: params["bins"]  = args.bins
    if args.s_min is not None and "s_min" in defaults: params["s_min"] = args.s_min
    if args.c_max is not None and "c_max" in defaults: params["c_max"] = args.c_max

    os.makedirs(args.outdir, exist_ok=True)

    H, ids = compute_for_folder(args.input, func, **params)

    # Build a tag for the filename
    tag = args.descriptor
    if "bins" in params:
        tag += f"_bins{params['bins']}"
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

    print(f"[OK] Saved {H.shape} descriptors to {out_file}")

if __name__ == "__main__":
    main()
