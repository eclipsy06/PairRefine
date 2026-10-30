"""Deterministic local geometry for frozen IterBlob primitives.

This module contains geometry only. It does not discover primitives, score
native structures, generate pose candidates, or define a trainable network.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


LEGACY_BASIS_MODE = "legacy_global_sign"
EQUIVARIANT_BASIS_MODE = "equivariant_v2"
DEFAULT_BASIS_MODE = EQUIVARIANT_BASIS_MODE
BASIS_MODES = (LEGACY_BASIS_MODE, EQUIVARIANT_BASIS_MODE)


def _normalize(vector: np.ndarray, *, tolerance: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        raise ValueError("cannot normalize a degenerate vector")
    return vector / norm


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    """Legacy non-equivariant PCA sign rule retained for old artifacts only."""

    nonzero = np.flatnonzero(np.abs(vector) > 1e-8)
    if len(nonzero) and vector[nonzero[0]] < 0:
        return -vector
    return vector


def _equivariant_sign(
    direction: np.ndarray, points: np.ndarray, pivot: np.ndarray
) -> np.ndarray:
    """Resolve PCA sign ambiguity using rotation-invariant projections."""

    projections = (points - pivot) @ direction
    index = int(np.argmax(np.abs(projections)))
    if abs(float(projections[index])) <= 1e-12:
        return direction
    if float(projections[index]) < 0.0:
        return -direction
    return direction


def primitive_local_basis(
    coordinates: np.ndarray,
    residue_mask: np.ndarray,
    partition_ids: np.ndarray,
    primitive_id: int,
    *,
    basis_mode: str = DEFAULT_BASIS_MODE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a native-free pivot and orthonormal primitive-local basis.

    The first axis points from the movable primitive toward its anchors. The
    remaining axes are obtained from movable-residue PCA and use an
    input-rotation-equivariant sign rule in the default ``equivariant_v2``
    mode. No native/test coordinates are read.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    residue_mask = np.asarray(residue_mask, dtype=bool)
    partition_ids = np.asarray(partition_ids, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape [L, 3]")
    if residue_mask.shape != coordinates.shape[:1] or partition_ids.shape != residue_mask.shape:
        raise ValueError("mask and partition_ids must match coordinate length")
    if basis_mode not in BASIS_MODES:
        raise ValueError(f"basis_mode must be one of {BASIS_MODES}")

    selected = residue_mask & (partition_ids == int(primitive_id))
    if int(selected.sum()) < 3:
        raise ValueError("a movable primitive requires at least three residues")
    rest = residue_mask & ~selected
    if int(rest.sum()) < 1:
        raise ValueError("a movable primitive requires at least one anchor residue")

    points = coordinates[selected]
    pivot = points.mean(axis=0)
    direction_to_rest = coordinates[rest].mean(axis=0) - pivot
    centered = points - pivot
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    sign_rule = (
        (lambda axis: _canonical_sign(axis))
        if basis_mode == LEGACY_BASIS_MODE
        else (lambda axis: _equivariant_sign(axis, points, pivot))
    )
    principal_axes = [_normalize(sign_rule(axis)) for axis in right]

    try:
        first = _normalize(direction_to_rest)
    except ValueError:
        first = principal_axes[0]
    second = None
    fallback_axes = [np.eye(3, dtype=np.float64)[index] for index in range(3)]
    for candidate in principal_axes + fallback_axes:
        orthogonal = candidate - np.dot(candidate, first) * first
        if float(np.linalg.norm(orthogonal)) > 1e-8:
            second = _normalize(sign_rule(orthogonal))
            break
    if second is None:
        raise ValueError("could not construct a primitive-local basis")
    third = _normalize(np.cross(first, second))
    second = _normalize(np.cross(third, first))
    basis = np.stack((first, second, third), axis=0)
    return pivot.astype(np.float64), basis.astype(np.float64)


def primitive_hinge_pivot(
    coordinates: np.ndarray,
    residue_mask: np.ndarray,
    partition_ids: np.ndarray,
    primitive_id: int,
) -> np.ndarray:
    """Estimate a sequence-connection pivot without native coordinates."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    residue_mask = np.asarray(residue_mask, dtype=bool)
    partition_ids = np.asarray(partition_ids, dtype=np.int64)
    selected = residue_mask & (partition_ids == int(primitive_id))
    rest = residue_mask & ~selected
    if int(selected.sum()) < 1 or int(rest.sum()) < 1:
        raise ValueError("hinge pivot requires movable and anchor residues")

    boundary = residue_mask[:-1] & residue_mask[1:] & (selected[:-1] != selected[1:])
    boundary_indices = np.flatnonzero(boundary)
    if len(boundary_indices):
        midpoints = 0.5 * (
            coordinates[boundary_indices] + coordinates[boundary_indices + 1]
        )
        return midpoints.mean(axis=0)

    selected_indices = np.flatnonzero(selected)
    rest_indices = np.flatnonzero(rest)
    sequence_gap = np.abs(selected_indices[:, None] - rest_indices[None, :])
    left, right = np.unravel_index(np.argmin(sequence_gap), sequence_gap.shape)
    return 0.5 * (
        coordinates[selected_indices[left]] + coordinates[rest_indices[right]]
    )


def axis_angle_matrix(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for one finite axis-angle rotation."""

    axis = _normalize(np.asarray(axis, dtype=np.float64))
    x, y, z = axis
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    identity = np.eye(3, dtype=np.float64)
    outer = np.outer(axis, axis)
    return (
        math.cos(angle_radians) * identity
        + math.sin(angle_radians) * skew
        + (1.0 - math.cos(angle_radians)) * outer
    )
