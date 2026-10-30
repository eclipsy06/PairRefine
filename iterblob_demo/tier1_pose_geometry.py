"""Bounded pose and hinge-centered rigid-transform utilities.

These functions are deterministic geometry operators used by the final
pair-attention head. They contain no graph refiner, boundary HMM, pose-voting
head, confidence gate, native scoring, or trainable parameters.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from .tier1_primitive_geometry import (
    DEFAULT_BASIS_MODE,
    axis_angle_matrix,
    primitive_hinge_pivot,
    primitive_local_basis,
)


def cap_norm(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    """Cap a vector's Euclidean norm without changing its direction."""

    if maximum <= 0:
        raise ValueError("maximum must be positive")
    norm = vector.norm(dim=-1, keepdim=True)
    scale = torch.clamp(float(maximum) / norm.clamp_min(1e-6), max=1.0)
    return vector * scale


def rotate_vectors(
    vectors: torch.Tensor, rotation_vectors: torch.Tensor
) -> torch.Tensor:
    """Apply exponential-map rotations with a differentiable identity limit."""

    theta_squared = rotation_vectors.square().sum(dim=-1, keepdim=True)
    theta_safe = torch.sqrt(theta_squared + 1e-12)
    theta_fourth = theta_squared.square()
    small = theta_squared < 1e-8
    sine_over_theta = torch.where(
        small,
        1.0 - theta_squared / 6.0 + theta_fourth / 120.0,
        torch.sin(theta_safe) / theta_safe,
    )
    one_minus_cosine_over_theta_squared = torch.where(
        small,
        0.5 - theta_squared / 24.0 + theta_fourth / 720.0,
        (1.0 - torch.cos(theta_safe)) / (theta_squared + 1e-12),
    )
    first_cross = torch.cross(rotation_vectors, vectors, dim=-1)
    second_cross = torch.cross(rotation_vectors, first_cross, dim=-1)
    return (
        vectors
        + sine_over_theta * first_cross
        + one_minus_cosine_over_theta_squared * second_cross
    )


def apply_local_hinge_pose(
    points: torch.Tensor,
    pivot: torch.Tensor,
    basis: torch.Tensor,
    pose: torch.Tensor,
) -> torch.Tensor:
    """Apply a differentiable local 6-DoF pose around a hinge pivot."""

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if pivot.shape != (3,) or basis.shape != (3, 3) or pose.shape != (6,):
        raise ValueError("pivot, basis, and pose have inconsistent shapes")
    rotation_global_radians = (pose[:3] @ basis) * (math.pi / 180.0)
    translation_global = pose[3:] @ basis
    centered = points - pivot.unsqueeze(0)
    expanded_rotation = rotation_global_radians.unsqueeze(0).expand_as(centered)
    return (
        rotate_vectors(centered, expanded_rotation)
        + pivot.unsqueeze(0)
        + translation_global.unsqueeze(0)
    )


def _as_vector3(value: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def apply_full_se3_action(
    coordinates: np.ndarray,
    residue_mask: np.ndarray,
    partition_ids: np.ndarray,
    *,
    primitive_id: int,
    rotation_vector_degrees: Sequence[float],
    translation_vector_angstroms: Sequence[float],
    rotation_pivot: str = "hinge",
    basis_mode: str = DEFAULT_BASIS_MODE,
) -> np.ndarray:
    """Apply one local-frame rigid transform to exactly one frozen primitive."""

    coordinates = np.asarray(coordinates, dtype=np.float32)
    residue_mask = np.asarray(residue_mask, dtype=bool)
    partition_ids = np.asarray(partition_ids, dtype=np.int64)
    rotation_local = _as_vector3(
        rotation_vector_degrees, name="rotation_vector_degrees"
    )
    translation_local = _as_vector3(
        translation_vector_angstroms, name="translation_vector_angstroms"
    )
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape [L, 3]")
    if residue_mask.shape != coordinates.shape[:1] or partition_ids.shape != residue_mask.shape:
        raise ValueError("mask and partition_ids must match coordinate length")
    if rotation_pivot not in {"center", "hinge"}:
        raise ValueError("rotation_pivot must be 'center' or 'hinge'")

    selected = residue_mask & (partition_ids == int(primitive_id))
    if int(selected.sum()) < 3 or int((residue_mask & ~selected).sum()) < 1:
        raise ValueError(
            "a movable primitive requires at least three residues and one anchor"
        )
    center, basis = primitive_local_basis(
        coordinates,
        residue_mask,
        partition_ids,
        int(primitive_id),
        basis_mode=basis_mode,
    )
    pivot = (
        center
        if rotation_pivot == "center"
        else primitive_hinge_pivot(
            coordinates, residue_mask, partition_ids, int(primitive_id)
        )
    )

    rotation_magnitude = float(np.linalg.norm(rotation_local))
    if rotation_magnitude == 0.0:
        rotation = np.eye(3, dtype=np.float64)
    else:
        rotation_global = rotation_local @ basis
        rotation = axis_angle_matrix(
            rotation_global / rotation_magnitude,
            math.radians(rotation_magnitude),
        )
    translation_global = translation_local @ basis
    points = coordinates[selected].astype(np.float64)
    moved = (points - pivot) @ rotation.T + pivot
    moved = moved + translation_global[None, :]
    result = coordinates.copy()
    result[selected] = moved.astype(np.float32)
    return result
