"""Sequence-mapped C-alpha metrics for multidomain structure predictions."""

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _kabsch(
    mobile: np.ndarray,
    reference: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("Kabsch inputs must both have shape [N, 3]")
    if mobile.shape[0] < 3:
        raise ValueError("at least three points are required for a stable fit")
    if weights is None:
        weights = np.ones(mobile.shape[0], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (mobile.shape[0],) or np.any(weights < 0):
        raise ValueError("Kabsch weights must be non-negative with shape [N]")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("Kabsch weights must have positive mass")
    mobile_center = (mobile * weights[:, None]).sum(axis=0) / total_weight
    reference_center = (reference * weights[:, None]).sum(axis=0) / total_weight
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center
    covariance = (mobile_centered * weights[:, None]).T @ reference_centered
    left, _, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right))
    rotation = left @ correction @ right
    translation = reference_center - mobile_center @ rotation
    aligned = mobile @ rotation + translation
    return aligned, rotation, translation


def _rmsd(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(left - right), axis=-1))))


def _lddt(
    predicted: np.ndarray,
    native: np.ndarray,
    pair_mask: np.ndarray,
    thresholds: Iterable[float] = (0.5, 1.0, 2.0, 4.0),
) -> Optional[float]:
    native_distance = np.linalg.norm(native[:, None] - native[None, :], axis=-1)
    predicted_distance = np.linalg.norm(
        predicted[:, None] - predicted[None, :], axis=-1
    )
    eligible = pair_mask & (native_distance < 15.0)
    if not np.any(eligible):
        return None
    error = np.abs(predicted_distance - native_distance)[eligible]
    scores = [(error < threshold).mean() for threshold in thresholds]
    return float(np.mean(scores))


def _fixed_correspondence_tm_score(
    predicted: np.ndarray, native: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Optimize a TM-like score with a fixed residue correspondence."""

    length = native.shape[0]
    d0 = max(0.5, 1.24 * np.cbrt(max(length - 15, 1)) - 1.8)
    aligned, rotation, translation = _kabsch(predicted, native)
    for _ in range(20):
        distances = np.linalg.norm(aligned - native, axis=-1)
        weights = 1.0 / np.square(1.0 + np.square(distances / d0))
        updated, updated_rotation, updated_translation = _kabsch(
            predicted, native, weights=weights
        )
        if np.max(np.abs(updated - aligned)) < 1e-6:
            aligned = updated
            rotation = updated_rotation
            translation = updated_translation
            break
        aligned = updated
        rotation = updated_rotation
        translation = updated_translation
    distances = np.linalg.norm(aligned - native, axis=-1)
    score = float(np.mean(1.0 / (1.0 + np.square(distances / d0))))
    return score, aligned


def ca_structure_metrics(
    predicted_ca: np.ndarray,
    predicted_mask: np.ndarray,
    native_ca: np.ndarray,
    native_mask: np.ndarray,
    domain_ids: np.ndarray,
) -> Dict[str, object]:
    """Compute global, intra-domain, and relative-domain C-alpha metrics."""

    predicted_ca = np.asarray(predicted_ca, dtype=np.float64)
    native_ca = np.asarray(native_ca, dtype=np.float64)
    predicted_mask = np.asarray(predicted_mask, dtype=bool)
    native_mask = np.asarray(native_mask, dtype=bool)
    domain_ids = np.asarray(domain_ids, dtype=np.int64)
    length = predicted_ca.shape[0]
    if predicted_ca.shape != native_ca.shape or predicted_ca.shape != (length, 3):
        raise ValueError("predicted and native C-alpha arrays must have shape [L, 3]")
    if any(array.shape != (length,) for array in (predicted_mask, native_mask, domain_ids)):
        raise ValueError("masks and domain_ids must have shape [L]")
    valid = predicted_mask & native_mask
    if int(valid.sum()) < 3:
        raise ValueError("fewer than three mapped C-alpha atoms are valid")
    predicted = predicted_ca[valid]
    native = native_ca[valid]
    labels = domain_ids[valid]
    valid_count = predicted.shape[0]
    upper = np.triu(np.ones((valid_count, valid_count), dtype=bool), k=1)
    labeled = labels >= 0
    same_domain = labels[:, None] == labels[None, :]
    labeled_pair = labeled[:, None] & labeled[None, :]

    global_aligned, _, _ = _kabsch(predicted, native)
    tm_score, _ = _fixed_correspondence_tm_score(predicted, native)
    result: Dict[str, object] = {
        "metric_protocol": "sequence_mapped_ca_v1",
        "valid_ca": int(valid_count),
        "global_ca_rmsd": _rmsd(global_aligned, native),
        "fixed_correspondence_ca_tm_score": tm_score,
        "global_ca_lddt": _lddt(predicted, native, upper),
        "intra_domain_ca_lddt": _lddt(
            predicted, native, upper & labeled_pair & same_domain
        ),
        "inter_domain_ca_lddt": _lddt(
            predicted, native, upper & labeled_pair & ~same_domain
        ),
    }

    domains = []
    fits = {}
    for domain_id in sorted(set(labels[labeled].tolist())):
        mask = labels == domain_id
        if int(mask.sum()) < 3:
            continue
        aligned, rotation, translation = _kabsch(predicted[mask], native[mask])
        fits[int(domain_id)] = (rotation, translation)
        domain_pair = np.triu(np.ones((int(mask.sum()), int(mask.sum())), dtype=bool), k=1)
        domains.append(
            {
                "domain_id": int(domain_id),
                "valid_ca": int(mask.sum()),
                "ca_rmsd": _rmsd(aligned, native[mask]),
                "ca_lddt": _lddt(predicted[mask], native[mask], domain_pair),
            }
        )
    result["domains"] = domains

    relative_pairs = []
    domain_ids_with_fits = sorted(fits)
    for left_index, left_domain in enumerate(domain_ids_with_fits):
        left_mask = labels == left_domain
        left_rotation, left_translation = fits[left_domain]
        for right_domain in domain_ids_with_fits[left_index + 1 :]:
            right_mask = labels == right_domain
            right_rotation, _ = fits[right_domain]
            right_after_left_fit = (
                predicted[right_mask] @ left_rotation + left_translation
            )
            relative_rotation = left_rotation @ right_rotation.T
            cosine = (np.trace(relative_rotation) - 1.0) / 2.0
            angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
            predicted_centroid_distance = np.linalg.norm(
                predicted[left_mask].mean(axis=0) - predicted[right_mask].mean(axis=0)
            )
            native_centroid_distance = np.linalg.norm(
                native[left_mask].mean(axis=0) - native[right_mask].mean(axis=0)
            )
            relative_pairs.append(
                {
                    "domain_i": left_domain,
                    "domain_j": right_domain,
                    "relative_rotation_error_deg": float(angle),
                    "domain_j_rmsd_after_domain_i_fit": _rmsd(
                        right_after_left_fit, native[right_mask]
                    ),
                    "centroid_distance_absolute_error": float(
                        abs(predicted_centroid_distance - native_centroid_distance)
                    ),
                }
            )
    result["relative_domain_pairs"] = relative_pairs
    if relative_pairs:
        result["mean_relative_rotation_error_deg"] = float(
            np.mean([item["relative_rotation_error_deg"] for item in relative_pairs])
        )
        result["mean_domain_placement_rmsd"] = float(
            np.mean(
                [item["domain_j_rmsd_after_domain_i_fit"] for item in relative_pairs]
            )
        )
    return result
