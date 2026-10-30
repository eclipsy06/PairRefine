"""Native-free primitive selection shared by training and deployment."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def select_primitive_from_pae(
    partition_ids: np.ndarray,
    residue_mask: np.ndarray,
    predicted_aligned_error: np.ndarray,
) -> Tuple[int, float, Dict[int, float]]:
    """Select the movable primitive with the largest symmetric cross-PAE."""

    partition_ids = np.asarray(partition_ids, dtype=np.int64)
    valid = np.asarray(residue_mask, dtype=bool)
    pae = np.asarray(predicted_aligned_error, dtype=np.float64)
    if partition_ids.ndim != 1 or valid.shape != partition_ids.shape:
        raise ValueError("partition_ids and residue_mask must be matching vectors")
    if pae.shape != (len(partition_ids), len(partition_ids)):
        raise ValueError("PAE must be a square matrix matching the partition length")
    if not np.all(np.isfinite(pae)) or np.any(pae < 0):
        raise ValueError("PAE must contain finite non-negative values")

    scores: Dict[int, float] = {}
    for primitive_id in sorted(set(partition_ids[valid].tolist())):
        if primitive_id < 0:
            continue
        moving = valid & (partition_ids == primitive_id)
        anchor = valid & (partition_ids != primitive_id)
        if int(moving.sum()) < 3 or not np.any(anchor):
            continue
        forward = pae[np.ix_(moving, anchor)]
        reverse = pae[np.ix_(anchor, moving)].T
        scores[int(primitive_id)] = float(0.5 * (forward.mean() + reverse.mean()))
    if not scores:
        raise ValueError("partition has no movable primitive")
    primitive_id = max(scores, key=lambda item: (scores[item], -item))
    return primitive_id, scores[primitive_id], scores
