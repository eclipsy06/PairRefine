"""IterBlob final residue-pair latent-attention pose-head package.

Only the ``ours_pair_attention`` architecture is part of this distribution.
Historical alternatives are intentionally excluded.
"""

from .tier1_residue_pair_attention import (
    FINAL_HEAD_SPEC,
    FINAL_HEAD_TYPE,
    ResiduePairAttentionPoseHead,
    create_pair_attention_pose_head,
)


FINAL_TRAINING_ENTRYPOINT = "train_tier1_residue_pair_attention.py"
PRETRAINING_USED = False
CONFIDENCE_GATING_USED = False

__all__ = [
    "FINAL_HEAD_SPEC",
    "FINAL_HEAD_TYPE",
    "FINAL_TRAINING_ENTRYPOINT",
    "PRETRAINING_USED",
    "CONFIDENCE_GATING_USED",
    "ResiduePairAttentionPoseHead",
    "create_pair_attention_pose_head",
]
