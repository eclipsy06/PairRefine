"""Final IterBlob residue-pair latent-attention pose head.

The head consumes the complete movable-anchor residue-pair set for one
PAE-selected frozen primitive and predicts one bounded local 6-DoF rigid pose.
Native coordinates are not accepted by the feature builder and cannot enter
the model input.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .tier1_pose_geometry import cap_norm


RESIDUE_PAIR_FEATURE_DIM = 17
RESIDUE_PAIR_FEATURE_VERSION = "full_local_pair_geometry_v1"
FINAL_HEAD_TYPE = "ours_pair_attention"
FINAL_HEAD_SPEC = {
    "reference_year": 2026,
    "family": "complete residue-pair latent cross-attention",
    "implementation_scope": "final_project_model",
}


def build_residue_pair_tokens(
    coordinates: np.ndarray,
    residue_mask: np.ndarray,
    partition_ids: np.ndarray,
    confidence: np.ndarray,
    predicted_aligned_error: np.ndarray,
    *,
    primitive_id: int,
    pivot: np.ndarray,
    basis: np.ndarray,
    coordinate_scale_angstroms: float = 32.0,
) -> np.ndarray:
    """Build one token for every moving-to-anchor residue pair.

    The local frame is supplied by the frozen primitive protocol.  Each token
    contains moving/anchor positions relative to the hinge, their displacement,
    distance, directed PAE, pLDDT, normalized sequence positions, and whether
    the anchor residue is outside a named primitive.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    residue_mask = np.asarray(residue_mask, dtype=bool)
    partition_ids = np.asarray(partition_ids, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float64)
    pae = np.asarray(predicted_aligned_error, dtype=np.float64)
    pivot = np.asarray(pivot, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape [L, 3]")
    length = coordinates.shape[0]
    if residue_mask.shape != (length,) or partition_ids.shape != (length,):
        raise ValueError("mask and partition must match the coordinate length")
    if confidence.shape != (length,) or pae.shape != (length, length):
        raise ValueError("confidence and PAE must match the coordinate length")
    if pivot.shape != (3,) or basis.shape != (3, 3):
        raise ValueError("pivot and basis have inconsistent shapes")
    if coordinate_scale_angstroms <= 0:
        raise ValueError("coordinate scale must be positive")
    if not all(
        np.all(np.isfinite(value))
        for value in (coordinates, confidence, pae, pivot, basis)
    ):
        raise ValueError("residue-pair inputs must be finite")
    if np.any(pae < 0):
        raise ValueError("predicted aligned error cannot be negative")

    valid = residue_mask & np.all(np.isfinite(coordinates), axis=-1)
    moving = np.flatnonzero(valid & (partition_ids == int(primitive_id)))
    anchor = np.flatnonzero(valid & (partition_ids != int(primitive_id)))
    if len(moving) < 3 or len(anchor) < 1:
        raise ValueError("a movable primitive and at least one anchor are required")

    moving_local = (coordinates[moving] - pivot[None, :]) @ basis.T
    anchor_local = (coordinates[anchor] - pivot[None, :]) @ basis.T
    moving_grid = np.broadcast_to(
        moving_local[:, None, :], (len(moving), len(anchor), 3)
    )
    anchor_grid = np.broadcast_to(
        anchor_local[None, :, :], (len(moving), len(anchor), 3)
    )
    displacement = anchor_grid - moving_grid
    distance = np.linalg.norm(displacement, axis=-1, keepdims=True)
    forward_pae = pae[np.ix_(moving, anchor)][..., None]
    reverse_pae = pae[np.ix_(anchor, moving)].T[..., None]
    moving_confidence = np.broadcast_to(
        confidence[moving, None, None], (len(moving), len(anchor), 1)
    )
    anchor_confidence = np.broadcast_to(
        confidence[None, anchor, None], (len(moving), len(anchor), 1)
    )
    denominator = float(max(length - 1, 1))
    moving_position = np.broadcast_to(
        (moving.astype(np.float64) / denominator)[:, None, None],
        (len(moving), len(anchor), 1),
    )
    anchor_position = np.broadcast_to(
        (anchor.astype(np.float64) / denominator)[None, :, None],
        (len(moving), len(anchor), 1),
    )
    flexible_anchor = np.broadcast_to(
        (partition_ids[anchor] < 0).astype(np.float64)[None, :, None],
        (len(moving), len(anchor), 1),
    )
    tokens = np.concatenate(
        (
            moving_grid / float(coordinate_scale_angstroms),
            anchor_grid / float(coordinate_scale_angstroms),
            displacement / float(coordinate_scale_angstroms),
            distance / float(coordinate_scale_angstroms),
            forward_pae / 32.0,
            reverse_pae / 32.0,
            moving_confidence / 100.0,
            anchor_confidence / 100.0,
            moving_position,
            anchor_position,
            flexible_anchor,
        ),
        axis=-1,
    )
    result = tokens.reshape(-1, RESIDUE_PAIR_FEATURE_DIM).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("residue-pair tokens contain non-finite values")
    return result


class ResiduePairAttentionPoseHead(nn.Module):
    """Permutation-invariant cross-attention pose head over all residue pairs."""

    def __init__(
        self,
        *,
        hidden_dim: int = 96,
        latent_count: int = 8,
        attention_heads: int = 4,
        max_rotation_degrees: float = 60.0,
        max_translation_angstroms: float = 4.0,
        use_token_encoder: bool = True,
        use_cross_attention: bool = True,
        use_nonlinear_decoder: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or latent_count < 1:
            raise ValueError("hidden_dim and latent_count must be positive")
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if not use_token_encoder and hidden_dim < RESIDUE_PAIR_FEATURE_DIM:
            raise ValueError(
                "hidden_dim must fit raw pair features when token encoder is removed"
            )
        if max_rotation_degrees <= 0 or max_translation_angstroms <= 0:
            raise ValueError("pose limits must be positive")
        self.hidden_dim = int(hidden_dim)
        self.use_token_encoder = bool(use_token_encoder)
        self.use_cross_attention = bool(use_cross_attention)
        self.use_nonlinear_decoder = bool(use_nonlinear_decoder)
        self.max_rotation_degrees = float(max_rotation_degrees)
        self.max_translation_angstroms = float(max_translation_angstroms)
        self.token_encoder = (
            nn.Sequential(
                nn.Linear(RESIDUE_PAIR_FEATURE_DIM, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            if self.use_token_encoder
            else None
        )
        if self.use_cross_attention:
            self.latents = nn.Parameter(torch.randn(latent_count, hidden_dim) * 0.02)
            self.cross_attention = nn.MultiheadAttention(
                hidden_dim, attention_heads, batch_first=True
            )
            self.latent_norm = nn.LayerNorm(hidden_dim)
        else:
            self.register_parameter("latents", None)
            self.cross_attention = None
            self.latent_norm = None
        if self.use_nonlinear_decoder:
            self.pose_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 6),
            )
            output_layer = self.pose_head[-1]
        else:
            self.pose_head = nn.Linear(2 * hidden_dim, 6)
            output_layer = self.pose_head
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, pair_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pair_tokens.ndim != 2 or pair_tokens.shape[-1] != RESIDUE_PAIR_FEATURE_DIM:
            raise ValueError(
                "pair_tokens must have shape [pair_count, RESIDUE_PAIR_FEATURE_DIM]"
            )
        if pair_tokens.shape[0] < 1:
            raise ValueError("at least one residue-pair token is required")
        tokens = pair_tokens.float()
        if self.token_encoder is None:
            encoded = F.pad(tokens, (0, self.hidden_dim - tokens.shape[-1]))
        else:
            encoded = self.token_encoder(tokens)
        encoded = encoded.unsqueeze(0)
        if self.cross_attention is None:
            summary_source = encoded
        else:
            queries = self.latents.unsqueeze(0)
            attended, _ = self.cross_attention(
                queries, encoded, encoded, need_weights=False
            )
            summary_source = self.latent_norm(attended + queries)
        summary = torch.cat(
            (summary_source.mean(dim=1), summary_source.amax(dim=1)), dim=-1
        )
        raw_pose = torch.tanh(self.pose_head(summary).squeeze(0))
        rotation = cap_norm(
            raw_pose[:3] * self.max_rotation_degrees,
            self.max_rotation_degrees,
        )
        translation = cap_norm(
            raw_pose[3:] * self.max_translation_angstroms,
            self.max_translation_angstroms,
        )
        pose = torch.cat((rotation, translation), dim=-1)
        return {"pose": pose}


def create_pair_attention_pose_head(
    *,
    hidden_dim: int = 96,
    latent_count: int = 8,
    attention_heads: int = 4,
    max_rotation_degrees: float = 60.0,
    max_translation_angstroms: float = 4.0,
) -> ResiduePairAttentionPoseHead:
    """Build the package's only trainable network architecture."""

    return ResiduePairAttentionPoseHead(
        hidden_dim=hidden_dim,
        latent_count=latent_count,
        attention_heads=attention_heads,
        max_rotation_degrees=max_rotation_degrees,
        max_translation_angstroms=max_translation_angstroms,
    )
