"""Training utilities used only by the final IterBlob pair-attention head."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .structure_metrics import ca_structure_metrics
from .tier1_pose_geometry import apply_full_se3_action, apply_local_hinge_pose
from .tier1_residue_pair_attention import ResiduePairAttentionPoseHead


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def endpoint_loss(
    model: ResiduePairAttentionPoseHead,
    example: dict,
    device: torch.device,
    scale: float,
) -> torch.Tensor:
    """Smooth L1 loss on rigidly transformed movable-residue coordinates."""

    tokens = torch.as_tensor(example["pair_tokens"], device=device)
    pose = model(tokens)["pose"]
    points = torch.as_tensor(example["points"], device=device)
    pivot = torch.as_tensor(example["pivot"], device=device)
    basis = torch.as_tensor(example["basis"], device=device)
    target = torch.as_tensor(example["target_endpoints"], device=device)
    predicted = apply_local_hinge_pose(points, pivot, basis, pose)
    return F.smooth_l1_loss(predicted / float(scale), target / float(scale))


@torch.no_grad()
def predict_poses(
    model: ResiduePairAttentionPoseHead,
    examples: Sequence[dict],
    device: torch.device,
) -> List[np.ndarray]:
    model.eval()
    return [
        model(torch.as_tensor(example["pair_tokens"], device=device))["pose"]
        .detach()
        .cpu()
        .numpy()
        for example in examples
    ]


def structure_records(
    examples: Sequence[dict], poses: Sequence[np.ndarray]
) -> List[dict]:
    """Evaluate development metrics before and after the predicted rigid pose."""

    records = []
    for example, pose in zip(examples, poses):
        if example["primitive_id"] < 0:
            refined = example["coordinates"].copy()
            pose = np.zeros(6, dtype=np.float32)
        else:
            refined = apply_full_se3_action(
                example["coordinates"],
                example["residue_mask"],
                example["partition_ids"],
                primitive_id=example["primitive_id"],
                rotation_vector_degrees=np.asarray(pose[:3], dtype=np.float32),
                translation_vector_angstroms=np.asarray(pose[3:], dtype=np.float32),
                rotation_pivot="hinge",
                basis_mode=example["basis_mode"],
            )
        baseline = ca_structure_metrics(
            example["coordinates"],
            example["residue_mask"],
            example["native"],
            example["residue_mask"],
            example["domain_ids"],
        )
        after = ca_structure_metrics(
            refined,
            example["residue_mask"],
            example["native"],
            example["residue_mask"],
            example["domain_ids"],
        )
        record = {
            "target": example["target"],
            "difficulty_bin": example["difficulty_bin"],
            "primitive_id": example["primitive_id"],
            "pair_count": (
                len(example["pair_tokens"])
                if example["pair_tokens"] is not None
                else 0
            ),
        }
        pose_names = (
            "rotation_x_degrees",
            "rotation_y_degrees",
            "rotation_z_degrees",
            "translation_x_angstroms",
            "translation_y_angstroms",
            "translation_z_angstroms",
        )
        for index, name in enumerate(pose_names):
            record[f"predicted_{name}"] = float(pose[index])
            record[f"teacher_{name}"] = float(example["target_pose"][index])
        metric_names = (
            "fixed_correspondence_ca_tm_score",
            "global_ca_rmsd",
            "global_ca_lddt",
            "inter_domain_ca_lddt",
            "intra_domain_ca_lddt",
        )
        for metric in metric_names:
            before = baseline[metric]
            refined_value = after[metric]
            record[f"baseline_{metric}"] = before
            record[f"refined_{metric}"] = refined_value
            record[f"improvement_{metric}"] = (
                None
                if before is None or refined_value is None
                else float(
                    before - refined_value
                    if metric == "global_ca_rmsd"
                    else refined_value - before
                )
            )
        records.append(record)
    return records


def summarize_records(records: Sequence[dict]) -> Dict[str, float]:
    fields = {
        "mean_delta_tm": "improvement_fixed_correspondence_ca_tm_score",
        "mean_delta_global_rmsd": "improvement_global_ca_rmsd",
        "mean_delta_global_lddt": "improvement_global_ca_lddt",
        "mean_delta_inter_domain_lddt": "improvement_inter_domain_ca_lddt",
        "mean_delta_intra_domain_lddt": "improvement_intra_domain_ca_lddt",
    }
    result: Dict[str, float] = {"target_count": len(records)}
    for output_name, field in fields.items():
        values = [row[field] for row in records if row[field] is not None]
        result[output_name] = float(np.mean(values)) if values else None
    result["positive_tm_fraction"] = float(
        np.mean([row[fields["mean_delta_tm"]] > 0 for row in records])
    )
    return result


def write_records(path: Path, records: Sequence[dict]) -> None:
    if not records:
        raise ValueError("cannot write an empty record sequence")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
