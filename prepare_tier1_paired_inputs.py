"""Build real AFDB-prediction to MPDB-native training pairs for Tier 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import List

import numpy as np


DEMO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEMO_ROOT
sys.path.insert(0, str(DEMO_ROOT))

from iterblob_demo.tier1_paired import (  # noqa: E402
    align_chain_to_afdb,
    parse_afdb_pae,
    project_afdb_pae,
    project_afdb_to_chain,
    stratified_unique_accession_sample,
)
from iterblob_demo.structure_metrics import ca_structure_metrics  # noqa: E402
from prepare_tier1_afdb_inputs import parse_afdb_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        default=WORKSPACE_ROOT
        / "output"
        / "iterblob_tier1"
        / "mpdb_clustered_cohort"
        / "tier1_mpdb_clustered_manifest.csv",
    )
    parser.add_argument(
        "--afdb-manifest",
        type=Path,
        default=WORKSPACE_ROOT
        / "output"
        / "iterblob_tier1"
        / "paired_afdb_models_pilot"
        / "tier1_afdb_model_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT
        / "output"
        / "iterblob_tier1"
        / "paired_inputs_pilot",
    )
    parser.add_argument("--limit-train", type=int, default=16)
    parser.add_argument("--limit-validation", type=int, default=8)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--min-identity", type=float, default=0.95)
    parser.add_argument("--min-query-coverage", type=float, default=0.80)
    parser.add_argument("--min-labeled-coverage", type=float, default=0.80)
    parser.add_argument("--min-mapped-residues", type=int, default=60)
    parser.add_argument("--selection-seed", type=int, default=20260817)
    parser.add_argument(
        "--pae-manifest",
        type=Path,
        help="Optional PAE-only manifest; every included accession must be present.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not 0 < args.min_identity <= 1:
        raise ValueError("min identity must be in (0, 1]")
    if not 0 < args.min_query_coverage <= 1 or not 0 < args.min_labeled_coverage <= 1:
        raise ValueError("coverage thresholds must be in (0, 1]")
    if args.min_mapped_residues < 3:
        raise ValueError("min mapped residues must be at least three")
    required_paths = [args.cohort_manifest, args.afdb_manifest]
    if args.pae_manifest is not None:
        required_paths.append(args.pae_manifest)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    with args.cohort_manifest.open("r", encoding="utf-8", newline="") as handle:
        cohort_rows = list(csv.DictReader(handle))
    with args.afdb_manifest.open("r", encoding="utf-8", newline="") as handle:
        afdb_rows = list(csv.DictReader(handle))
    pae_by_accession = None
    if args.pae_manifest is not None:
        with args.pae_manifest.open("r", encoding="utf-8", newline="") as handle:
            pae_rows = list(csv.DictReader(handle))
        pae_by_accession = {row["accession"].upper(): row for row in pae_rows}
        if len(pae_by_accession) != len(pae_rows):
            raise ValueError("PAE manifest contains duplicate accessions")
    split_limits = {
        split: limit
        for split, limit in (
            ("train", args.limit_train),
            ("validation", args.limit_validation),
            ("test", args.limit_test),
        )
        if limit > 0
    }
    if not split_limits or min(args.limit_train, args.limit_validation, args.limit_test) < 0:
        raise ValueError("at least one split limit must be positive; none may be negative")
    selected = stratified_unique_accession_sample(
        cohort_rows,
        split_limits,
        seed=args.selection_seed,
    )
    afdb_by_accession = {row["accession"].upper(): row for row in afdb_rows}
    if len(afdb_by_accession) != len(afdb_rows):
        raise ValueError("AFDB manifest contains duplicate accessions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "examples"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed_models = {}
    output_rows = []
    exclusions = []
    for row in selected:
        target = row["target"]
        accession = row.get("uniprot_accession", "").upper()
        afdb_row = afdb_by_accession.get(accession)
        if afdb_row is None:
            exclusions.append({"target": target, "reason": "AFDB model unavailable"})
            continue
        try:
            with np.load(row["cache_npz"], allow_pickle=False) as arrays:
                sequence = str(arrays["sequence"])
                aatype = arrays["aatype"].astype(np.int64)
                native_coordinates = arrays["coordinates"].astype(np.float32)
                domain_ids = arrays["domain_ids"].astype(np.int64)
                motion_ids = arrays["motion_ids"].astype(np.int64)
                residue_numbers = arrays["residue_numbers"].astype(np.int32)
                insertion_codes = arrays["insertion_codes"].astype("U1")
            length = len(sequence)
            expected_shape = (length,)
            if native_coordinates.shape != (length, 3) or any(
                value.shape != expected_shape
                for value in (aatype, domain_ids, motion_ids, residue_numbers, insertion_codes)
            ):
                raise ValueError("native cache arrays have inconsistent shapes")

            model_path = Path(afdb_row["model_cif"])
            model_key = str(model_path.resolve())
            if model_key not in parsed_models:
                parsed_models[model_key] = parse_afdb_model(model_path)
            afdb_coordinates, afdb_mask, afdb_plddt, afdb_sequence, _ = parsed_models[
                model_key
            ]
            mapping = align_chain_to_afdb(sequence, afdb_sequence)
            predicted_coordinates, predicted_mask, plddt = project_afdb_to_chain(
                mapping, afdb_coordinates, afdb_mask, afdb_plddt
            )
            native_mask = np.isfinite(native_coordinates).all(axis=-1)
            residue_mask = predicted_mask & native_mask
            labeled = domain_ids >= 0
            mapped_count = int(residue_mask.sum())
            labeled_count = int(labeled.sum())
            labeled_coverage = float((residue_mask & labeled).sum() / max(labeled_count, 1))
            if mapping.identity < args.min_identity:
                raise ValueError(
                    f"alignment identity {mapping.identity:.4f} < {args.min_identity:.4f}"
                )
            if mapping.query_coverage < args.min_query_coverage:
                raise ValueError(
                    f"query coverage {mapping.query_coverage:.4f} < {args.min_query_coverage:.4f}"
                )
            if labeled_coverage < args.min_labeled_coverage:
                raise ValueError(
                    f"labeled coverage {labeled_coverage:.4f} < {args.min_labeled_coverage:.4f}"
                )
            if mapped_count < args.min_mapped_residues:
                raise ValueError(
                    f"mapped residues {mapped_count} < {args.min_mapped_residues}"
                )

            baseline_metrics = ca_structure_metrics(
                predicted_coordinates,
                residue_mask,
                native_coordinates,
                residue_mask,
                domain_ids,
            )
            baseline_inter_lddt = baseline_metrics.get("inter_domain_ca_lddt")
            if baseline_inter_lddt is None:
                difficulty_bin = "unscored"
            elif baseline_inter_lddt < 0.80:
                difficulty_bin = "hard"
            elif baseline_inter_lddt < 0.90:
                difficulty_bin = "medium"
            else:
                difficulty_bin = "easy"

            pae = None
            pae_maximum = None
            pae_row = None
            if pae_by_accession is not None:
                pae_row = pae_by_accession.get(accession)
                if pae_row is None:
                    raise ValueError("AFDB PAE unavailable")
                afdb_pae, pae_maximum = parse_afdb_pae(Path(pae_row["pae_json"]))
                if afdb_pae.shape[0] != len(afdb_sequence):
                    raise ValueError(
                        f"PAE length {afdb_pae.shape[0]} != AFDB sequence "
                        f"{len(afdb_sequence)}"
                    )
                pae = project_afdb_pae(
                    mapping, afdb_pae, fill_value=float(pae_maximum)
                )

            cache_path = cache_dir / f"{target}.paired.npz"
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            cache_payload = {
                "target": np.asarray(target),
                "sequence": np.asarray(sequence),
                "aatype": aatype,
                "coordinates": predicted_coordinates,
                "native_coordinates": native_coordinates,
                "residue_mask": residue_mask,
                "predicted_mask": predicted_mask,
                "native_mask": native_mask,
                "plddt": plddt,
                "domain_ids": domain_ids,
                "motion_ids": motion_ids,
                "residue_numbers": residue_numbers,
                "insertion_codes": insertion_codes,
                "afdb_index_for_chain": mapping.query_to_reference,
            }
            if pae is not None:
                cache_payload["pae"] = pae
                cache_payload["pae_maximum"] = np.asarray(
                    pae_maximum, dtype=np.float32
                )
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **cache_payload)
            temporary.replace(cache_path)
            output_row = dict(row)
            output_row.update(
                {
                    "cache_npz": str(cache_path.resolve()),
                    "native_cache_npz": str(Path(row["cache_npz"]).resolve()),
                    "afdb_model_cif": str(model_path.resolve()),
                    "afdb_model_sha256": afdb_row["model_sha256"],
                    "alignment_identity": f"{mapping.identity:.8f}",
                    "query_coverage": f"{mapping.query_coverage:.8f}",
                    "labeled_coverage": f"{labeled_coverage:.8f}",
                    "mapped_residues": mapped_count,
                    "mean_mapped_plddt": f"{float(plddt[residue_mask].mean()):.6f}",
                    "baseline_tm_score": f"{float(baseline_metrics['fixed_correspondence_ca_tm_score']):.8f}",
                    "baseline_global_lddt": f"{float(baseline_metrics['global_ca_lddt']):.8f}",
                    "baseline_inter_domain_lddt": (
                        ""
                        if baseline_inter_lddt is None
                        else f"{float(baseline_inter_lddt):.8f}"
                    ),
                    "baseline_domain_placement_rmsd": (
                        ""
                        if baseline_metrics.get("mean_domain_placement_rmsd") is None
                        else f"{float(baseline_metrics['mean_domain_placement_rmsd']):.8f}"
                    ),
                    "difficulty_bin": difficulty_bin,
                    "pae_json": "" if pae_row is None else pae_row["pae_json"],
                    "pae_sha256": "" if pae_row is None else pae_row["pae_sha256"],
                    "mean_mapped_pae": (
                        ""
                        if pae is None
                        else f"{float(pae[np.ix_(residue_mask, residue_mask)].mean()):.6f}"
                    ),
                }
            )
            output_rows.append(output_row)
        except Exception as error:
            exclusions.append(
                {"target": target, "reason": f"{type(error).__name__}: {error}"}
            )

    manifest_path = args.output_dir / "tier1_paired_manifest.csv"
    write_csv(manifest_path, output_rows)
    summary = {
        "schema_version": "iterblob_tier1_paired_inputs_v1",
        "status": "complete" if len(output_rows) == len(selected) else "partial",
        "input_mode": "paired_afdb_prediction_to_mpdb_native",
        "cohort_manifest": str(args.cohort_manifest.resolve()),
        "cohort_manifest_sha256": sha256(args.cohort_manifest),
        "afdb_manifest": str(args.afdb_manifest.resolve()),
        "afdb_manifest_sha256": sha256(args.afdb_manifest),
        "pae_manifest": (
            None if args.pae_manifest is None else str(args.pae_manifest.resolve())
        ),
        "pae_manifest_sha256": (
            None if args.pae_manifest is None else sha256(args.pae_manifest)
        ),
        "requested": len(selected),
        "selection": {
            "strategy": "length_domain_stratified_unique_accession_v1",
            "seed": args.selection_seed,
        },
        "included": len(output_rows),
        "split_counts": {
            split: sum(row["split"] == split for row in output_rows)
            for split in split_limits
        },
        "difficulty_counts": {
            split: {
                difficulty: sum(
                    row["split"] == split and row["difficulty_bin"] == difficulty
                    for row in output_rows
                )
                for difficulty in ("hard", "medium", "easy", "unscored")
            }
            for split in split_limits
        },
        "thresholds": {
            "min_identity": args.min_identity,
            "min_query_coverage": args.min_query_coverage,
            "min_labeled_coverage": args.min_labeled_coverage,
            "min_mapped_residues": args.min_mapped_residues,
        },
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
        "exclusions": exclusions,
        "test_set_touched": "test" in split_limits,
    }
    (args.output_dir / "tier1_paired_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if output_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
