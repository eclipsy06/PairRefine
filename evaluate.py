"""Evaluate saved PairRefine actions using the official Zhang-lab TMscore.

The reference is Structure2. Both structures use the same sequence-mapped
residue_mask as the original benchmark. RMSD is the executable's RMSD of the
common residues; positive TM-score deltas and RMSD reductions indicate gains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from iterblob_demo.official_structure_metrics import run_tmscore
from iterblob_demo.tier1_pose_geometry import apply_full_se3_action
from iterblob_demo.tier1_primitive_selection import select_primitive_from_pae
from predict import (
    BASIS_MODE, POSE_FIELDS, coordinates_sha256, load_predicted_target,
    read_csv, safe_target_name, sha256, target_index, write_ca_pdb, write_csv,
)


def paired_summary(values: list[float], *, seed: int, samples: int) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("paired summary requires finite, nonempty values")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(data), size=(count, len(data)))
        means[start:start + count] = data[indices].mean(axis=1)
    return {
        "n": len(data), "mean": float(data.mean()), "median": float(np.median(data)),
        "mean_ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "positive_count": int(np.sum(data > 0)), "unchanged_count": int(np.sum(data == 0)),
        "negative_count": int(np.sum(data < 0)), "positive_fraction": float(np.mean(data > 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-results", "--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="user_supplied_cohort")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--limit-targets", type=int)
    parser.add_argument("--target")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if args.bootstrap_samples < 1 or (args.limit_targets is not None and args.limit_targets < 1):
        raise ValueError("bootstrap-samples and limit-targets must be positive")
    saved_rows = read_csv(args.saved_results)
    target_index(saved_rows, "predicted actions")
    if args.target:
        saved_rows = [row for row in saved_rows if row["target"].casefold() == args.target.casefold()]
    if args.limit_targets is not None:
        saved_rows = saved_rows[:args.limit_targets]
    if not saved_rows:
        raise ValueError("no predicted actions selected")
    manifest = target_index(read_csv(args.manifest), "input manifest")
    partitions = target_index(read_csv(args.partition_manifest), "partition manifest")
    checkpoint_hashes = {row["checkpoint_sha256"] for row in saved_rows if row.get("checkpoint_sha256")}
    if len(checkpoint_hashes) > 1:
        raise ValueError("predicted actions combine different checkpoints")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir, raw_dir = args.output_dir / "pdb", args.output_dir / "raw_tmscore"
    pdb_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    output_rows = []
    started = time.perf_counter()
    for index, saved in enumerate(saved_rows, start=1):
        target = safe_target_name(saved["target"])
        key = target.casefold()
        if key not in manifest or key not in partitions:
            raise ValueError(f"missing manifest or partition for {target}")
        values = load_predicted_target(manifest[key], partitions[key], args.manifest, args.partition_manifest)
        for field, path in (("cache_sha256", values["cache_path"]), ("partition_sha256", values["partition_path"])):
            if saved.get(field) and sha256(path) != saved[field]:
                raise ValueError(f"{target}: {field} differs from inference")
        if saved.get("basis_mode", BASIS_MODE) != BASIS_MODE or float(saved.get("pose_scale", 1.0)) != 1.0:
            raise ValueError(f"{target}: expected equivariant_v2 and pose_scale=1")
        if saved.get("confidence_gating_used", "false").lower() not in ("false", "0", ""):
            raise ValueError(f"{target}: confidence gating is outside the released protocol")
        with np.load(values["cache_path"], allow_pickle=False) as arrays:
            native = arrays["native_coordinates"].astype(np.float32)
        coordinates, mask, partition = values["coordinates"], values["residue_mask"], values["partition_ids"]
        if native.shape != coordinates.shape or not np.all(np.isfinite(native[mask])):
            raise ValueError(f"{target}: native coordinates are invalid")
        primitive_id = int(saved["selected_primitive_id"])
        pose = np.asarray([float(saved[field]) for field in POSE_FIELDS], dtype=np.float32)
        if not np.all(np.isfinite(pose)):
            raise ValueError(f"{target}: non-finite saved pose")
        valid_ids = [int(value) for value in np.unique(partition[mask]) if value >= 0]
        if len(valid_ids) < 2:
            if primitive_id != -1 or np.any(pose != 0):
                raise ValueError(f"{target}: single-primitive fallback must be the identity")
            refined = coordinates.copy()
        else:
            expected, _, _ = select_primitive_from_pae(partition, mask, values["pae"])
            if primitive_id != expected:
                raise ValueError(f"{target}: saved primitive disagrees with PAE deployment selection")
            refined = apply_full_se3_action(
                coordinates, mask, partition, primitive_id=primitive_id,
                rotation_vector_degrees=pose[:3], translation_vector_angstroms=pose[3:],
                rotation_pivot="hinge", basis_mode=BASIS_MODE,
            )
        if saved.get("refined_coordinates_sha256") and coordinates_sha256(refined) != saved["refined_coordinates_sha256"]:
            raise ValueError(f"{target}: reconstructed coordinates differ from inference")
        paths = {
            "baseline": pdb_dir / f"{target}_afdb_v6_mapped_ca.pdb",
            "iterblob": pdb_dir / f"{target}_iterblob_mapped_ca.pdb",
            "native": pdb_dir / f"{target}_native_mapped_ca.pdb",
        }
        for name, trace, confidence in (
            ("baseline", coordinates, values["plddt"]),
            ("iterblob", refined, values["plddt"]),
            ("native", native, np.full_like(values["plddt"], 100.0)),
        ):
            write_ca_pdb(paths[name], values["sequence"], trace, confidence, mask)
        scores = {}
        for name in ("baseline", "iterblob"):
            score = run_tmscore(paths[name], paths["native"], args.binary)
            if score.common_residues != int(mask.sum()) or score.structure2_length != int(mask.sum()):
                raise ValueError(f"{target}: official residue count differs from the mapped trace")
            scores[name] = score
            (raw_dir / f"{target}_{name}.json").write_text(
                json.dumps(score.to_dict(include_raw=True), indent=2) + "\n", encoding="utf-8",
            )
        baseline, model = scores["baseline"], scores["iterblob"]
        delta_tm = model.tm_score - baseline.tm_score
        rmsd_reduction = baseline.common_residue_rmsd - model.common_residue_rmsd
        output_rows.append({
            "target": target, "pdb_id": saved.get("pdb_id", ""),
            "uniprot_accession": saved.get("uniprot_accession", ""),
            "official_common_residues": baseline.common_residues,
            "official_normalization_length": baseline.structure2_length,
            "baseline_official_tm_score": baseline.tm_score,
            "iterblob_official_tm_score": model.tm_score,
            "delta_official_tm_score": delta_tm,
            "baseline_official_rmsd": baseline.common_residue_rmsd,
            "iterblob_official_rmsd": model.common_residue_rmsd,
            "delta_official_rmsd": rmsd_reduction,
            "official_rmsd_reduction": rmsd_reduction,
            "both_metrics_improved": delta_tm > 0 and rmsd_reduction > 0,
            "selected_primitive_id": primitive_id,
            "action_status": saved.get("action_status", ""),
        })
        print(json.dumps({"completed": index, "total": len(saved_rows), "target": target,
                          "delta_official_tm_score": delta_tm,
                          "official_rmsd_reduction": rmsd_reduction}), flush=True)
    per_target_path = args.output_dir / "official_tmscore_per_target.csv"
    write_csv(per_target_path, output_rows)
    tm_improvement = paired_summary([row["delta_official_tm_score"] for row in output_rows], seed=args.seed, samples=args.bootstrap_samples)
    rmsd_improvement = paired_summary([row["official_rmsd_reduction"] for row in output_rows], seed=args.seed + 1, samples=args.bootstrap_samples)
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": "iterblob_external_official_tmscore_v1", "status": "complete",
        "dataset_name": args.dataset_name, "metric_implementation": "Zhang_lab_TMscore_executable",
        "normalization": "sequence_mapped_native_CA_trace_as_structure2",
        "rmsd_definition": "TMscore RMSD of the common residues",
        "target_count": len(output_rows),
        "mean_baseline_official_tm_score": float(np.mean([row["baseline_official_tm_score"] for row in output_rows])),
        "mean_iterblob_official_tm_score": float(np.mean([row["iterblob_official_tm_score"] for row in output_rows])),
        "official_tm_improvement": tm_improvement,
        "mean_baseline_official_rmsd": float(np.mean([row["baseline_official_rmsd"] for row in output_rows])),
        "mean_iterblob_official_rmsd": float(np.mean([row["iterblob_official_rmsd"] for row in output_rows])),
        "official_rmsd_reduction": rmsd_improvement,
        "both_metrics_improved": tm_improvement["mean"] > 0 and rmsd_improvement["mean"] > 0,
        "checkpoint_sha256": next(iter(checkpoint_hashes), None),
        "elapsed_seconds": elapsed, "seconds_per_target_pair": elapsed / len(output_rows),
        "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.seed,
        "inputs": {"saved_results_sha256": sha256(args.saved_results),
                   "manifest_sha256": sha256(args.manifest),
                   "partition_manifest_sha256": sha256(args.partition_manifest),
                   "tmscore_binary_sha256": sha256(args.binary), "basis_mode": BASIS_MODE},
        "per_target_csv": per_target_path.name,
    }
    (args.output_dir / "official_tmscore_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
