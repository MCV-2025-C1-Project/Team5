import numpy as np

from src.tools import utils
from src.data.extract import read_image


def compute_confusion_counts_mask(result: np.ndarray, annotation: np.ndarray):
    """Compute confusion counts between a prediction and an annotation mask.

    Args:
        result: Predicted mask. Can be a HxW boolean/int array or an image
            (HxWxC) where any non-zero channel indicates foreground.
        annotation: Ground-truth annotation mask with the same accepted
            formats as ``result``.

    Returns:
        A 4-tuple of integers: ``(tp, fp, tn, fn)`` where:
            - tp: True positives (A & O)
            - fp: False positives (¬A & O)
            - tn: True negatives (¬A & ¬O)
            - fn: False negatives (A & ¬O)

    Raises:
        ValueError: if the normalized masks do not have the same shape.
    """
    # Normalize masks to boolean HxW
    result_b = utils.normalize_mask(result)
    annotation_b = utils.normalize_mask(annotation)

    utils.validate_same_shape(result_b, annotation_b)

    tp = np.logical_and(result_b, annotation_b).sum()
    fp = np.logical_and(np.logical_not(annotation_b), result_b).sum()
    tn = np.logical_and(np.logical_not(annotation_b), np.logical_not(result_b)).sum()
    fn = np.logical_and(annotation_b, np.logical_not(result_b)).sum()

    return int(tp), int(fp), int(tn), int(fn)


def precision_mask(tp: int, fp: int) -> float:
    """Compute precision from confusion counts.

    Args:
        tp: Number of true positives.
        fp: Number of false positives.

    Returns:
        A float in [0.0, 1.0] representing precision.
    """
    denom = tp + fp
    if denom == 0:
        return 0.0
    return tp / denom


def recall_mask(tp: int, fn: int) -> float:
    """Compute recall (sensitivity) from confusion counts.

    Args:
        tp: Number of true positives.
        fn: Number of false negatives.

    Returns:
        A float in [0.0, 1.0] representing recall.
    """
    denom = tp + fn
    if denom == 0:
        return 0.0
    return tp / denom


def f1_mask(tp: int, fp: int, fn: int) -> float:
    """Compute the F1-score from confusion counts.

    Args:
        tp: Number of true positives.
        fp: Number of false positives.
        fn: Number of false negatives.

    Returns:
        A float in [0.0, 1.0] representing the F1-score.
    """
    prec = precision_mask(tp, fp)
    rec = recall_mask(tp, fn)
    denom = prec + rec
    if denom == 0:
        return 0.0
    return 2 * (prec * rec) / denom
