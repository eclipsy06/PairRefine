"""Run PairRefine on cached predicted structures and frozen partitions.

Only predicted coordinates, pLDDT, PAE and the supplied partition are read by
inference. Native coordinates and supervision labels are never loaded here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from iterblob_demo.tier1_pose_geometry import apply_full_se3_action
from iterblob_demo.tier1_primitive_geometry import (
    primitive_hinge_pivot,
    primitive_local_basis,
)
from iterblob_demo.tier1_primitive_selection import select_primitive_from_pae
from iterblob_demo.tier1_residue_pair_attention import create_pair_attention_pose_head
from iterblob_demo.tier1_residue_pair_attention import build_residue_pair_tokens


POSE_FIELDS = (
    "rotation_x_degrees", "rotation_y_degrees", "rotation_z_degrees",
    "translation_x_angstroms", "translation_y_angstroms", "translation_z_angstroms",
)
BASIS_MODE = "equivariant_v2"


def read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def target_index(rows: list[dict], label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        target = row.get("target", "").strip()
        if not target or target.casefold() in result:
            raise ValueError(f"{label} contains an empty or duplicate target: {target!r}")
        result[target.casefold()] = row
    if not result:
        raise ValueError(f"{label} contains no targets")
    return result


def resolve_asset(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.resolve().parent / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinates_sha256(coordinates: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(coordinates, dtype="<f4").tobytes()).hexdigest()


def safe_target_name(target: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", target):
        raise ValueError(f"target is unsuitable for an output filename: {target!r}")
    return target


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_ca_pdb(
    path: Path, sequence: str, coordinates: np.ndarray,
    plddt: np.ndarray, mask: np.ndarray,
) -> None:
    """Use the original benchmark's mapped residue numbers and CA-only trace."""
    residues = dict(zip(
        "ARNDCQEGHILKMFPSTWYVX",
        ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
         "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
         "TYR", "VAL", "UNK"),
    ))
    length = len(sequence)
    if coordinates.shape != (length, 3) or mask.shape != (length,) or plddt.shape != (length,):
        raise ValueError("sequence, coordinates, confidence and mask lengths must match")
    if length > 9999 or not np.all(np.isfinite(coordinates[mask])):
        raise ValueError("CA PDB export requires finite coordinates and at most 9999 residues")
    lines = ["REMARK 250 IterBlob official-TMscore C-alpha trace"]
    atom_index = 1
    for residue_index, (letter, coordinate) in enumerate(zip(sequence, coordinates), start=1):
        if not mask[residue_index - 1]:
            continue
        x, y, z = coordinate
        if any(len(f"{float(value):8.3f}") > 8 for value in coordinate):
            raise ValueError("coordinate exceeds the fixed-width PDB field")
        lines.append(
            f"ATOM  {atom_index:5d}  CA  "
            f"{residues.get(letter, 'UNK'):>3s} A{residue_index:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}"
            f"{plddt[residue_index - 1]:6.2f}           C"
        )
        atom_index += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def load_predicted_target(row: dict, partition: dict, manifest: Path, partition_manifest: Path) -> dict:
    provenance = " ".join(str(partition.get(key, "")).casefold() for key in (
        "provider", "provider_version", "input_source",
    ))
    if any(word in provenance for word in ("native", "reference", "official_casp")):
        raise ValueError(f"{row['target']}: provide predicted, native-free partitions")
    cache = resolve_asset(row["cache_npz"], manifest)
    partition_path = resolve_asset(partition["partition_npz"], partition_manifest)
    with np.load(cache, allow_pickle=False) as arrays:
        values = {
            "sequence": str(arrays["sequence"].item()),
            "coordinates": arrays["coordinates"].astype(np.float32),
            "residue_mask": arrays["residue_mask"].astype(bool),
            "plddt": arrays["plddt"].astype(np.float32),
            "pae": arrays["pae"].astype(np.float32),
        }
    with np.load(partition_path, allow_pickle=False) as arrays:
        values["partition_ids"] = arrays["partition_ids"].astype(np.int64)
    length = len(values["sequence"])
    shapes = {"coordinates": (length, 3), "residue_mask": (length,),
              "plddt": (length,), "pae": (length, length), "partition_ids": (length,)}
    for key, expected in shapes.items():
        if values[key].shape != expected:
            raise ValueError(f"{row['target']}: invalid {key} shape {values[key].shape}")
    if not values["residue_mask"].any():
        raise ValueError(f"{row['target']}: residue_mask is empty")
    for key in ("coordinates", "plddt", "pae"):
        if not np.all(np.isfinite(values[key])):
            raise ValueError(f"{row['target']}: {key} contains non-finite values")
    if np.any(values["pae"] < 0):
        raise ValueError(f"{row['target']}: PAE contains negative values")
    values.update(cache_path=cache, partition_path=partition_path)
    return values


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping with model_state_dict or state_dict")
    if checkpoint.get("head_type", "ours_pair_attention") != "ours_pair_attention":
        raise ValueError("this release supports only ours_pair_attention checkpoints")
    training_args = checkpoint.get("args", {})
    if not isinstance(training_args, dict):
        training_args = vars(training_args)
    basis_mode = training_args.get("basis_mode", checkpoint.get("basis_mode", BASIS_MODE))
    if basis_mode != BASIS_MODE:
        raise ValueError("the released inference protocol requires equivariant_v2")
    if training_args.get("primitive_selection", "pae_deployment") != "pae_deployment":
        raise ValueError("checkpoint primitive selection must be pae_deployment")
    if training_args.get("pair_feature_profile", "full") != "full":
        raise ValueError("checkpoint must use the complete residue-pair feature profile")
    if training_args.get("v3_physics_gate", False):
        raise ValueError("the released inference protocol has no physics/confidence gate")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("checkpoint has neither model_state_dict nor state_dict")
    config = dict(checkpoint.get("model_config", {}))
    if not config:
        config = {"hidden_dim": int(state["token_encoder.0.weight"].shape[0]),
                  "latent_count": int(state["latents"].shape[0]), "attention_heads": 4}
    model = create_pair_attention_pose_head(**config)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split")
    parser.add_argument("--limit-targets", type=int)
    parser.add_argument("--write-pdb", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1 or (args.limit_targets is not None and args.limit_targets < 1):
        raise ValueError("threads and limit-targets must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    rows = read_csv(args.manifest)
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]
    target_index(rows, "input manifest")
    if args.limit_targets is not None:
        rows = rows[:args.limit_targets]
    partitions = target_index(read_csv(args.partition_manifest), "partition manifest")
    started = time.perf_counter()
    model, checkpoint, config = load_model(args.checkpoint, device)
    checkpoint_hash = sha256(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = args.output_dir / "pdb"
    if args.write_pdb:
        pdb_dir.mkdir(exist_ok=True)
    output_rows = []
    for index, row in enumerate(rows, start=1):
        target = safe_target_name(row["target"])
        if target.casefold() not in partitions:
            raise ValueError(f"missing predicted partition for {target}")
        item_started = time.perf_counter()
        values = load_predicted_target(row, partitions[target.casefold()], args.manifest, args.partition_manifest)
        coordinates, mask, partition = values["coordinates"], values["residue_mask"], values["partition_ids"]
        valid_ids = [int(value) for value in np.unique(partition[mask]) if value >= 0]
        pair_count = 0
        pose = np.zeros(6, dtype=np.float32)
        primitive_id, score, scores = -1, None, {}
        status = "identity_fallback_single_predicted_primitive"
        refined = coordinates.copy()
        forward_seconds = 0.0
        if len(valid_ids) >= 2:
            primitive_id, score, scores = select_primitive_from_pae(partition, mask, values["pae"])
            _, basis = primitive_local_basis(coordinates, mask, partition, primitive_id, basis_mode=BASIS_MODE)
            pivot = primitive_hinge_pivot(coordinates, mask, partition, primitive_id)
            tokens = build_residue_pair_tokens(
                coordinates, mask, partition, values["plddt"], values["pae"],
                primitive_id=primitive_id, pivot=pivot, basis=basis,
            )
            pair_count = len(tokens)
            tensor = torch.from_numpy(tokens).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_started = time.perf_counter()
            with torch.inference_mode():
                pose = model(tensor)["pose"].detach().cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - forward_started
            if not np.all(np.isfinite(pose)):
                raise ValueError(f"{target}: model produced a non-finite pose")
            refined = apply_full_se3_action(
                coordinates, mask, partition, primitive_id=primitive_id,
                rotation_vector_degrees=pose[:3], translation_vector_angstroms=pose[3:],
                rotation_pivot="hinge", basis_mode=BASIS_MODE,
            )
            status = "pose_head_applied"
        record = {
            "target": target, "split": row.get("split", ""),
            "pdb_id": row.get("pdb_id", target[:4].upper()),
            "uniprot_accession": row.get("uniprot_accession", ""),
            "length": len(values["sequence"]), "mapped_residues": int(mask.sum()),
            "action_status": status, "selected_primitive_id": primitive_id,
            "selected_primitive_pae": score,
            "primitive_pae_scores": json.dumps(scores, sort_keys=True),
            "pair_token_count": pair_count, "basis_mode": BASIS_MODE,
            "pose_scale": 1.0, "confidence_gating_used": False,
            **{key: float(value) for key, value in zip(POSE_FIELDS, pose)},
            "checkpoint_sha256": checkpoint_hash,
            "cache_sha256": sha256(values["cache_path"]),
            "partition_sha256": sha256(values["partition_path"]),
            "refined_coordinates_sha256": coordinates_sha256(refined),
            "forward_seconds": forward_seconds,
            "target_elapsed_seconds": time.perf_counter() - item_started,
        }
        if args.write_pdb:
            write_ca_pdb(pdb_dir / f"{target}_pairrefine_mapped_ca.pdb", values["sequence"], refined, values["plddt"], mask)
        output_rows.append(record)
        print(json.dumps({"completed": index, "total": len(rows), "target": target, "action_status": status}), flush=True)
    actions_path = args.output_dir / "predicted_actions.csv"
    write_csv(actions_path, output_rows)
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": "pairrefine_prediction_v1", "status": "complete",
        "target_count": len(output_rows), "head_type": "ours_pair_attention",
        "model_config": config, "epoch": checkpoint.get("epoch"),
        "artifact_role": checkpoint.get("artifact_role"), "checkpoint_sha256": checkpoint_hash,
        "basis_mode": BASIS_MODE, "primitive_selection": "pae_deployment",
        "pose_scale": 1.0, "confidence_gating_used": False,
        "native_coordinates_read": False, "device": str(device),
        "elapsed_seconds": elapsed, "seconds_per_target": elapsed / len(output_rows),
        "forward_seconds_total": sum(row["forward_seconds"] for row in output_rows),
        "manifest_sha256": sha256(args.manifest),
        "partition_manifest_sha256": sha256(args.partition_manifest),
        "predicted_actions_sha256": sha256(actions_path),
    }
    (args.output_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
