"""Train the final IterBlob residue-pair latent-attention pose head.

This is the package's only training entry point. It trains
``ours_pair_attention`` from scratch on caller-provided manifests, uses frozen
PAE deployment primitive selection, and selects checkpoints with the official
Zhang-lab TMscore executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import List, Optional, Sequence

import numpy as np
import torch


DEMO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_ROOT))

from iterblob_demo.tier1_pair_attention_training import (  # noqa: E402
    endpoint_loss,
    predict_poses,
    read_csv,
    structure_records,
    summarize_records,
    write_records,
)
from iterblob_demo.tier1_primitive_geometry import (  # noqa: E402
    BASIS_MODES,
    DEFAULT_BASIS_MODE,
    primitive_hinge_pivot,
    primitive_local_basis,
)
from iterblob_demo.tier1_primitive_selection import (  # noqa: E402
    select_primitive_from_pae,
)
from iterblob_demo.tier1_residue_pair_attention import (  # noqa: E402
    FINAL_HEAD_SPEC,
    FINAL_HEAD_TYPE,
    RESIDUE_PAIR_FEATURE_VERSION,
    build_residue_pair_tokens,
    create_pair_attention_pose_head,
)
from iterblob_demo.tier1_pose_geometry import apply_full_se3_action  # noqa: E402
from iterblob_demo.official_validation_selection import (  # noqa: E402
    OfficialValidationEvaluator,
)


SCHEMA_VERSION = "iterblob_pair_attention_from_scratch_v4"
PRETRAINING_USED = False
CONFIDENCE_GATING_USED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--train-partition-manifest", type=Path, required=True)
    parser.add_argument("--validation-partition-manifest", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--head-type",
        choices=(FINAL_HEAD_TYPE,),
        default=FINAL_HEAD_TYPE,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--evaluation-interval", type=int, default=1)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("tm_lddt", "tm_rmsd", "tm_rmsd_lddt_2of3"),
        default="tm_lddt",
        help="Validation safety metrics used when selecting the best checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--latent-count", type=int, default=8)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--endpoint-scale-angstroms", type=float, default=4.0)
    parser.add_argument("--basis-mode", choices=BASIS_MODES, default=DEFAULT_BASIS_MODE)
    parser.add_argument(
        "--primitive-selection",
        choices=("pae_deployment",),
        default="pae_deployment",
        help="Frozen native-free PAE selection used by the final deployment protocol.",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic PyTorch/CUDA kernels (enabled by default).",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument(
        "--save-epochs",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Persist these exact training epochs independently of validation "
            "selection. Intended for fixed-epoch paper ablations."
        ),
    )
    parser.add_argument(
        "--official-tmscore-binary",
        type=Path,
        required=True,
        help=(
            "Use the official TMscore executable on the validation split and "
            "select best_model.pt by official TM-score with an RMSD safety gate."
        ),
    )
    parser.add_argument(
        "--official-tmscore-workers",
        type=int,
        default=1,
        help="Parallel TMscore.exe workers; metric values and ordering are unchanged.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split_examples(
    manifest: Path,
    partition_manifest: Path,
    labels_csv: Path,
    *,
    split: str,
    basis_mode: str,
    primitive_selection: str,
    limit: Optional[int] = None,
) -> List[dict]:
    rows = [row for row in read_csv(manifest) if row.get("split") == split]
    if limit is not None:
        rows = rows[:limit]
    partitions = {
        row["target"].casefold(): row
        for row in read_csv(partition_manifest)
        if row.get("split") == split
    }
    labels = {row["target"].casefold(): row for row in read_csv(labels_csv)}
    examples = []
    for row in rows:
        target = row["target"]
        key = target.casefold()
        if key not in partitions or key not in labels:
            raise ValueError(f"missing partition or supervision label for {target}")
        with np.load(row["cache_npz"], allow_pickle=False) as arrays:
            sequence = str(arrays["sequence"].item())
            coordinates = arrays["coordinates"].astype(np.float32)
            native = arrays["native_coordinates"].astype(np.float32)
            residue_mask = arrays["residue_mask"].astype(bool)
            confidence = arrays["plddt"].astype(np.float32)
            pae = arrays["pae"].astype(np.float32)
            domain_ids = arrays["domain_ids"].astype(np.int64)
        with np.load(partitions[key]["partition_npz"], allow_pickle=False) as arrays:
            partition_ids = arrays["partition_ids"].astype(np.int64)
        label = labels[key]
        primitive_id = int(label["best_primitive_id"])
        if primitive_selection != "pae_deployment":
            raise ValueError("the final package requires pae_deployment selection")
        label_policy = label.get("primitive_selection", "")
        if label_policy != "highest_symmetric_cross_partition_pae":
            raise ValueError(
                f"{target} does not have a PAE-consistent supervision label"
            )
        try:
            expected_primitive, _, _ = select_primitive_from_pae(
                partition_ids, residue_mask, pae
            )
        except ValueError:
            expected_primitive = -1
        if primitive_id != expected_primitive:
            raise ValueError(
                f"PAE primitive mismatch for {target}: "
                f"label={primitive_id}, deployment={expected_primitive}"
            )
        target_pose = np.asarray(
            [
                float(label["best_rotation_x_degrees"]),
                float(label["best_rotation_y_degrees"]),
                float(label["best_rotation_z_degrees"]),
                float(label["best_translation_x_angstroms"]),
                float(label["best_translation_y_angstroms"]),
                float(label["best_translation_z_angstroms"]),
            ],
            dtype=np.float32,
        )
        example = {
            "target": target,
            "sequence": sequence,
            "confidence": confidence,
            "difficulty_bin": row.get("difficulty_bin", "unscored"),
            "coordinates": coordinates,
            "native": native,
            "residue_mask": residue_mask,
            "domain_ids": domain_ids,
            "partition_ids": partition_ids,
            "primitive_id": primitive_id,
            "basis_mode": basis_mode,
            "target_pose": target_pose,
            "pair_tokens": None,
            "points": None,
            "pivot": None,
            "basis": None,
            "target_endpoints": None,
        }
        if primitive_id >= 0:
            selected = residue_mask & (partition_ids == primitive_id)
            if int(selected.sum()) < 3 or int((residue_mask & ~selected).sum()) < 1:
                raise ValueError(f"selected primitive is not movable for {target}")
            _, basis = primitive_local_basis(
                coordinates,
                residue_mask,
                partition_ids,
                primitive_id,
                basis_mode=basis_mode,
            )
            pivot = primitive_hinge_pivot(
                coordinates, residue_mask, partition_ids, primitive_id
            )
            target_coordinates = apply_full_se3_action(
                coordinates,
                residue_mask,
                partition_ids,
                primitive_id=primitive_id,
                rotation_vector_degrees=target_pose[:3],
                translation_vector_angstroms=target_pose[3:],
                rotation_pivot="hinge",
                basis_mode=basis_mode,
            )
            example.update(
                {
                    "pair_tokens": build_residue_pair_tokens(
                        coordinates,
                        residue_mask,
                        partition_ids,
                        confidence,
                        pae,
                        primitive_id=primitive_id,
                        pivot=pivot,
                        basis=basis,
                    ),
                    "points": coordinates[selected].astype(np.float32),
                    "pivot": np.asarray(pivot, dtype=np.float32),
                    "basis": np.asarray(basis, dtype=np.float32),
                    "target_endpoints": target_coordinates[selected].astype(np.float32),
                }
            )
        examples.append(example)
    if not examples:
        raise ValueError(f"no {split} examples")
    return examples


@torch.no_grad()
def predict_split_poses(
    model: torch.nn.Module,
    examples: Sequence[dict],
    device: torch.device,
) -> List[np.ndarray]:
    moving_examples = [example for example in examples if example["primitive_id"] >= 0]
    moving_predictions = iter(predict_poses(model, moving_examples, device))
    return [
        next(moving_predictions)
        if example["primitive_id"] >= 0
        else np.zeros(6, dtype=np.float32)
        for example in examples
    ]


def evaluate_split(model, examples, device):
    records = structure_records(
        examples, predict_split_poses(model, examples, device)
    )
    return summarize_records(records), records


def save_checkpoint(path: Path, payload: dict) -> None:
    def portable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, tuple):
            return tuple(portable(item) for item in value)
        return value

    with path.open("wb") as handle:
        torch.save(portable(payload), handle)


def checkpoint_feasible(metrics: dict, policy: str) -> bool:
    if policy == "tm_rmsd_lddt_2of3":
        return metric_improvement_count(metrics) >= 2
    if policy == "tm_rmsd":
        return metrics["mean_delta_global_rmsd"] >= 0.0
    return (
        metrics["mean_delta_global_lddt"] >= 0.0
        and metrics["mean_delta_inter_domain_lddt"] >= 0.0
    )


def metric_improvement_count(metrics: dict) -> int:
    values = (
        metrics["mean_delta_tm"],
        metrics["mean_delta_global_rmsd"],
        metrics["mean_delta_global_lddt"],
    )
    return sum(float(value) > 0.0 for value in values)


def checkpoint_score(metrics: dict, policy: str):
    if policy == "tm_rmsd_lddt_2of3":
        return metric_improvement_count(metrics), metrics["mean_delta_tm"]
    return metrics["mean_delta_tm"]


def main() -> None:
    args = parse_args()
    requested_save_epochs = sorted(set(args.save_epochs))
    if any(epoch < 1 or epoch > args.epochs for epoch in requested_save_epochs):
        raise ValueError("--save-epochs values must be within [1, --epochs]")
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.evaluation_interval < 1
        or args.official_tmscore_workers < 1
    ):
        raise ValueError("epochs, batch size, and evaluation interval must be positive")
    if args.learning_rate <= 0 or args.endpoint_scale_angstroms <= 0:
        raise ValueError("learning rate and endpoint scale must be positive")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    started = time.perf_counter()
    train_examples = load_split_examples(
        args.train_manifest,
        args.train_partition_manifest,
        args.train_labels,
        split="train",
        basis_mode=args.basis_mode,
        primitive_selection=args.primitive_selection,
        limit=args.limit_train,
    )
    validation_examples = load_split_examples(
        args.validation_manifest,
        args.validation_partition_manifest,
        args.validation_labels,
        split="validation",
        basis_mode=args.basis_mode,
        primitive_selection=args.primitive_selection,
        limit=args.limit_validation,
    )
    model_config = {
        "hidden_dim": args.hidden_dim,
        "latent_count": args.latent_count,
        "attention_heads": args.attention_heads,
    }
    identity_model = create_pair_attention_pose_head(
        hidden_dim=args.hidden_dim,
        latent_count=args.latent_count,
        attention_heads=args.attention_heads,
    )
    identity_state = {
        key: value.detach().cpu().clone()
        for key, value in identity_model.state_dict().items()
    }
    model = create_pair_attention_pose_head(
        hidden_dim=args.hidden_dim,
        latent_count=args.latent_count,
        attention_heads=args.attention_heads,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    identity_model = identity_model.to(device)
    identity_metrics, _ = evaluate_split(identity_model, validation_examples, device)
    if any(abs(float(value)) > 1e-12 for key, value in identity_metrics.items() if key.startswith("mean_delta_")):
        raise RuntimeError("fresh modern head is not an exact identity checkpoint")
    initial_metrics, _ = evaluate_split(model, validation_examples, device)
    official_evaluator = None
    best_official_epoch = 0
    best_official_metrics = None
    best_official_state = identity_state
    official_feasible_checkpoints = []
    fixed_epoch_checkpoints = []
    if args.official_tmscore_binary is not None:
        official_evaluator = OfficialValidationEvaluator(
            validation_examples,
            args.official_tmscore_binary,
            args.output_dir / "official_validation",
            workers=args.official_tmscore_workers,
        )
        initial_poses = predict_split_poses(model, validation_examples, device)
        best_official_metrics, _ = official_evaluator.evaluate(initial_poses, 0)
    best_score = (
        (0, 0.0)
        if args.checkpoint_policy == "tm_rmsd_lddt_2of3"
        else 0.0
    )
    best_epoch = 0
    best_state = identity_state
    best_tm_epoch = 0
    best_tm_score = 0.0
    best_tm_state = identity_state
    best_rmsd_epoch = 0
    best_rmsd_score = 0.0
    best_rmsd_state = identity_state
    initial_feasible = checkpoint_feasible(
        initial_metrics, args.checkpoint_policy
    )
    initial_score = checkpoint_score(initial_metrics, args.checkpoint_policy)
    if initial_feasible and initial_score > best_score:
        best_score = initial_score
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    if initial_metrics["mean_delta_tm"] > best_tm_score:
        best_tm_score = initial_metrics["mean_delta_tm"]
        best_tm_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    if initial_metrics["mean_delta_global_rmsd"] > best_rmsd_score:
        best_rmsd_score = initial_metrics["mean_delta_global_rmsd"]
        best_rmsd_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    del identity_model
    history = [
        {
            "epoch": 0,
            "stage": "identity_safety_baseline",
            "train_endpoint_loss": None,
            "validation": identity_metrics,
        },
        {
            "epoch": 0,
            "stage": "from_scratch_initialization",
            "train_endpoint_loss": None,
            "validation": initial_metrics,
        },
    ]
    generator = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = generator.permutation(len(train_examples))
        loss_sum = 0.0
        moving_count = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            moving = [
                int(index)
                for index in indices
                if train_examples[int(index)]["primitive_id"] >= 0
            ]
            if not moving:
                continue
            optimizer.zero_grad(set_to_none=True)
            for index in moving:
                loss = endpoint_loss(
                    model,
                    train_examples[index],
                    device,
                    args.endpoint_scale_angstroms,
                )
                (loss / len(moving)).backward()
                loss_sum += float(loss.detach().cpu())
                moving_count += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_loss = loss_sum / moving_count
        if epoch in requested_save_epochs:
            fixed_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            fixed_dir = args.output_dir / "fixed_epoch_checkpoints"
            fixed_dir.mkdir(parents=True, exist_ok=True)
            fixed_checkpoint = fixed_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                fixed_checkpoint,
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_role": "paper_ablation_fixed_epoch_weights",
                    "selection_metric": "predeclared_fixed_epoch",
                    "epoch": epoch,
                    "head_type": args.head_type,
                    "head_spec": FINAL_HEAD_SPEC,
                    "model_config": model_config,
                    "args": vars(args),
                    "model_state_dict": fixed_state,
                },
            )
            fixed_epoch_checkpoints.append(
                {
                    "epoch": epoch,
                    "checkpoint": str(fixed_checkpoint.resolve()),
                    "checkpoint_sha256": sha256(fixed_checkpoint),
                }
            )
        if epoch % args.evaluation_interval != 0 and epoch != args.epochs:
            history.append(
                {"epoch": epoch, "train_endpoint_loss": train_loss, "validation": None}
            )
            continue
        validation_metrics, _ = evaluate_split(model, validation_examples, device)
        official_metrics = None
        if official_evaluator is not None:
            validation_poses = predict_split_poses(model, validation_examples, device)
            official_metrics, _ = official_evaluator.evaluate(validation_poses, epoch)
            official_feasible = (
                official_metrics["mean_delta_official_tm_score"] > 0.0
                and official_metrics["mean_official_rmsd_reduction"] > 0.0
            )
            best_official_feasible = (
                best_official_metrics["mean_delta_official_tm_score"] > 0.0
                and best_official_metrics["mean_official_rmsd_reduction"] > 0.0
            )
            official_score = (
                official_metrics["mean_delta_official_tm_score"],
                official_metrics["mean_official_rmsd_reduction"],
            )
            best_official_score = (
                best_official_metrics["mean_delta_official_tm_score"],
                best_official_metrics["mean_official_rmsd_reduction"],
            )
            if official_feasible:
                feasible_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                feasible_dir = args.output_dir / "official_feasible_checkpoints"
                feasible_dir.mkdir(parents=True, exist_ok=True)
                feasible_checkpoint = feasible_dir / f"epoch_{epoch:03d}.pt"
                save_checkpoint(
                    feasible_checkpoint,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_role": "official_validation_feasible_epoch_weights",
                        "selection_metric": "official_tm_score_and_rmsd_both_positive",
                        "epoch": epoch,
                        "official_validation": official_metrics,
                        "head_type": args.head_type,
                        "model_config": model_config,
                        "args": vars(args),
                        "model_state_dict": feasible_state,
                    },
                )
                official_feasible_checkpoints.append(
                    {
                        "epoch": epoch,
                        "official_validation": official_metrics,
                        "checkpoint": str(feasible_checkpoint.resolve()),
                        "checkpoint_sha256": sha256(feasible_checkpoint),
                    }
                )
            if official_feasible and (
                not best_official_feasible or official_score > best_official_score
            ):
                best_official_epoch = epoch
                best_official_metrics = official_metrics
                best_official_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                # Persist immediately so an interrupted long run cannot lose the
                # best official-validation checkpoint found so far.
                save_checkpoint(
                    args.output_dir / "best_model_official_in_progress.pt",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_role": "best_official_validation_weights_in_progress",
                        "selection_metric": "official_tm_score_with_rmsd_gate",
                        "epoch": best_official_epoch,
                        "official_validation": best_official_metrics,
                        "head_type": args.head_type,
                        "model_config": model_config,
                        "args": vars(args),
                        "model_state_dict": best_official_state,
                    },
                )
        feasible = checkpoint_feasible(
            validation_metrics, args.checkpoint_policy
        )
        score = checkpoint_score(validation_metrics, args.checkpoint_policy)
        if feasible and score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if validation_metrics["mean_delta_tm"] > best_tm_score:
            best_tm_score = validation_metrics["mean_delta_tm"]
            best_tm_epoch = epoch
            best_tm_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if validation_metrics["mean_delta_global_rmsd"] > best_rmsd_score:
            best_rmsd_score = validation_metrics["mean_delta_global_rmsd"]
            best_rmsd_epoch = epoch
            best_rmsd_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        record = {
            "epoch": epoch,
            "train_endpoint_loss": train_loss,
            "validation": validation_metrics,
            "checkpoint_feasible": feasible,
            "official_validation": official_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    final_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    final_train_metrics, final_train_records = evaluate_split(model, train_examples, device)
    if official_evaluator is not None:
        best_state = best_official_state
        best_epoch = best_official_epoch
    model.load_state_dict(best_state)
    best_validation_metrics, best_validation_records = evaluate_split(
        model, validation_examples, device
    )
    if official_evaluator is not None:
        gate = {
            "mean_delta_official_tm_score_positive": (
                best_official_metrics["mean_delta_official_tm_score"] > 0.0
            ),
            "mean_official_rmsd_reduction_positive": (
                best_official_metrics["mean_official_rmsd_reduction"] > 0.0
            ),
        }
    elif args.checkpoint_policy == "tm_rmsd_lddt_2of3":
        improved_count = metric_improvement_count(best_validation_metrics)
        gate = {
            "improved_metric_count": improved_count,
            "at_least_two_of_three_improved": improved_count >= 2,
        }
    elif args.checkpoint_policy == "tm_rmsd":
        gate = {
            "mean_delta_tm_above_0_001": best_validation_metrics["mean_delta_tm"] > 0.001,
        }
        gate["mean_delta_global_rmsd_nonnegative"] = (
            best_validation_metrics["mean_delta_global_rmsd"] >= 0.0
        )
    else:
        gate = {
            "mean_delta_tm_above_0_001": best_validation_metrics["mean_delta_tm"] > 0.001,
        }
        gate["mean_delta_global_lddt_nonnegative"] = (
            best_validation_metrics["mean_delta_global_lddt"] >= 0.0
        )
        gate["mean_delta_inter_domain_lddt_nonnegative"] = (
            best_validation_metrics["mean_delta_inter_domain_lddt"] >= 0.0
        )
    gate["passed"] = all(gate.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = args.output_dir / "best_model.pt"
    best_tm_checkpoint = args.output_dir / "best_model_tm.pt"
    best_rmsd_checkpoint = args.output_dir / "best_model_rmsd.pt"
    final_checkpoint = args.output_dir / "final_model.pt"
    common = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": RESIDUE_PAIR_FEATURE_VERSION,
        "head_type": args.head_type,
        "head_spec": FINAL_HEAD_SPEC,
        "model_config": model_config,
        "args": vars(args),
    }
    save_checkpoint(
        best_checkpoint,
        {
            **common,
            "artifact_role": "best_validation_weights",
            "selection_metric": (
                "official_tm_score_with_rmsd_gate"
                if official_evaluator is not None
                else args.checkpoint_policy
            ),
            "epoch": best_epoch,
            "model_state_dict": best_state,
        },
    )
    save_checkpoint(
        best_tm_checkpoint,
        {
            **common,
            "artifact_role": "best_validation_weights",
            "selection_metric": "mean_delta_tm",
            "epoch": best_tm_epoch,
            "model_state_dict": best_tm_state,
        },
    )
    save_checkpoint(
        best_rmsd_checkpoint,
        {
            **common,
            "artifact_role": "best_validation_weights",
            "selection_metric": "mean_delta_global_rmsd",
            "epoch": best_rmsd_epoch,
            "model_state_dict": best_rmsd_state,
        },
    )
    save_checkpoint(
        final_checkpoint,
        {**common, "artifact_role": "final_training_weights", "epoch": args.epochs, "model_state_dict": final_state},
    )
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    selection_name = (
        "pae_primitive"
        if args.primitive_selection == "pae_deployment"
        else "teacher_primitive"
    )
    validation_path = args.output_dir / f"validation_{selection_name}_per_target.csv"
    train_path = args.output_dir / f"final_train_{selection_name}_per_target.csv"
    write_records(validation_path, best_validation_records)
    write_records(train_path, final_train_records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "claim_status": "development_validation_not_independent_test_evidence",
        "independent_test_touched": False,
        "frozen_backbones": ["AFDB_prediction", "Chainsaw", "Merizo_M1"],
        "training_mode": f"pose_only_{selection_name}",
        "primitive_selection": args.primitive_selection,
        "isolated_intervention": "modern_pose_head_under_fixed_residue_pair_protocol",
        "head_type": args.head_type,
        "head_spec": FINAL_HEAD_SPEC,
        "pretraining": {"used": PRETRAINING_USED},
        "confidence_gating": {"used": CONFIDENCE_GATING_USED},
        "from_scratch_initialization_validation": initial_metrics,
        "feature_version": RESIDUE_PAIR_FEATURE_VERSION,
        "train_count": len(train_examples),
        "validation_count": len(validation_examples),
        "train_nonidentity_count": sum(example["primitive_id"] >= 0 for example in train_examples),
        "validation_nonidentity_count": sum(example["primitive_id"] >= 0 for example in validation_examples),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "checkpoint_policy": args.checkpoint_policy,
        "official_checkpoint_selection": official_evaluator is not None,
        "best_official_validation": best_official_metrics,
        "official_feasible_checkpoints": official_feasible_checkpoints,
        "fixed_epoch_checkpoints": fixed_epoch_checkpoints,
        "best_epoch": best_epoch,
        "best_validation": best_validation_metrics,
        "metric_specific_checkpoints": {
            "tm": {
                "epoch": best_tm_epoch,
                "validation_mean_delta_tm": best_tm_score,
                "checkpoint": str(best_tm_checkpoint.resolve()),
                "checkpoint_sha256": sha256(best_tm_checkpoint),
            },
            "rmsd": {
                "epoch": best_rmsd_epoch,
                "validation_mean_delta_global_rmsd": best_rmsd_score,
                "checkpoint": str(best_rmsd_checkpoint.resolve()),
                "checkpoint_sha256": sha256(best_rmsd_checkpoint),
            },
        },
        "final_train": final_train_metrics,
        "go_no_go_gate": gate,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "best_checkpoint": str(best_checkpoint.resolve()),
        "best_checkpoint_sha256": sha256(best_checkpoint),
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_checkpoint_sha256": sha256(final_checkpoint),
        "validation_per_target": str(validation_path.resolve()),
        "final_train_per_target": str(train_path.resolve()),
        "inputs": {
            "train_manifest": str(args.train_manifest.resolve()),
            "validation_manifest": str(args.validation_manifest.resolve()),
            "train_partition_manifest": str(args.train_partition_manifest.resolve()),
            "validation_partition_manifest": str(args.validation_partition_manifest.resolve()),
            "train_labels": str(args.train_labels.resolve()),
            "validation_labels": str(args.validation_labels.resolve()),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
