"""Freeze a small, local MPDB cohort for Tier-1 IterBlob training."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

import numpy as np


DEMO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_ROOT))

from iterblob_demo.tier1_data import (  # noqa: E402
    BoundaryRecord,
    encode_sequence,
    load_mpdb_record,
    read_boundary_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mpdb-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEMO_ROOT / "data" / "local" / "mpdb_cohort",
    )
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--validation-count", type=int, default=200)
    parser.add_argument("--min-length", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--min-domain-residues", type=int, default=20)
    parser.add_argument("--min-label-coverage", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--exclude-test-manifest",
        type=Path,
        help="CSV with uniprot_accession and/or pdb_id columns reserved for testing",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def length_bin(length: int) -> str:
    if length <= 159:
        return "080_159"
    if length <= 255:
        return "160_255"
    return "256_384"


def boundary_span(record: BoundaryRecord) -> int:
    endpoints = [
        value.number
        for domain in record.domain_ranges
        for bounds in domain
        for value in bounds
    ]
    return max(endpoints) - min(endpoints) + 1


def validation_quotas(strata: Dict[Tuple[int, str], List[dict]], total: int):
    population = sum(len(items) for items in strata.values())
    capacity = {
        key: max(0, len(items) - 1)
        for key, items in strata.items()
    }
    if total > sum(capacity.values()):
        raise ValueError("validation count leaves no training example in some strata")
    exact = {key: total * len(items) / population for key, items in strata.items()}
    minimum = {
        key: int(len(items) >= 2)
        for key, items in strata.items()
    }
    quota = {
        key: min(capacity[key], max(minimum[key], int(exact[key])))
        for key in strata
    }
    remaining = total - sum(quota.values())
    while remaining > 0:
        eligible = [key for key in strata if quota[key] < capacity[key]]
        if not eligible:
            raise ValueError("unable to allocate requested validation count")
        key = min(eligible, key=lambda value: (-(exact[value] - quota[value]), value))
        quota[key] += 1
        remaining -= 1
    while remaining < 0:
        eligible = [key for key in strata if quota[key] > minimum[key]]
        if not eligible:
            raise ValueError("stratum minimums exceed requested validation count")
        key = min(eligible, key=lambda value: (exact[value] - quota[value], value))
        quota[key] -= 1
        remaining += 1
    return quota


def main() -> int:
    args = parse_args()
    if args.train_count <= 0 or args.validation_count <= 0:
        raise ValueError("train and validation counts must be positive")
    if not args.mpdb_root.is_dir() or not args.boundaries.is_file():
        raise FileNotFoundError("MPDB root or boundary file is missing")
    total_count = args.train_count + args.validation_count
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "examples"
    cache_dir.mkdir(parents=True, exist_ok=True)

    boundary_errors: List[str] = []
    excluded_test_uniprot = set()
    excluded_test_pdb = set()
    if args.exclude_test_manifest:
        if not args.exclude_test_manifest.is_file():
            raise FileNotFoundError(args.exclude_test_manifest)
        with args.exclude_test_manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("uniprot_accession"):
                    excluded_test_uniprot.add(row["uniprot_accession"])
                if row.get("pdb_id"):
                    excluded_test_pdb.add(row["pdb_id"].upper())

    boundaries = read_boundary_records(
        args.boundaries, strict=False, errors=boundary_errors
    )
    paths = {path.stem.lower(): path for path in args.mpdb_root.glob("*.pdb")}
    candidates = [
        record
        for key, record in boundaries.items()
        if key in paths
        and 2 <= record.domain_count <= 4
        and args.min_length <= boundary_span(record) <= args.max_length
    ]
    random.Random(args.seed).shuffle(candidates)
    selected = []
    seen_sequences = set()
    seen_uniprot = set()
    exclusions = defaultdict(int)
    for boundary in candidates:
        if boundary.target[:4].upper() in excluded_test_pdb:
            exclusions["reserved_test_pdb"] += 1
            continue
        try:
            record = load_mpdb_record(paths[boundary.target.lower()], boundary)
        except Exception:
            exclusions["parse_error"] += 1
            continue
        length = len(record.sequence)
        if not args.min_length <= length <= args.max_length:
            exclusions["observed_length"] += 1
            continue
        labeled = record.domain_ids >= 0
        if float(labeled.mean()) < args.min_label_coverage:
            exclusions["label_coverage"] += 1
            continue
        counts = [int(np.sum(record.domain_ids == index)) for index in range(boundary.domain_count)]
        if min(counts) < args.min_domain_residues:
            exclusions["small_domain"] += 1
            continue
        if record.uniprot_accession in excluded_test_uniprot:
            exclusions["reserved_test_uniprot"] += 1
            continue
        sequence_hash = hashlib.sha256(record.sequence.encode("ascii")).hexdigest()
        if sequence_hash in seen_sequences:
            exclusions["exact_sequence_duplicate"] += 1
            continue
        if record.uniprot_accession and record.uniprot_accession in seen_uniprot:
            exclusions["uniprot_duplicate"] += 1
            continue
        seen_sequences.add(sequence_hash)
        if record.uniprot_accession:
            seen_uniprot.add(record.uniprot_accession)
        selected.append(
            {
                "record": record,
                "definition": boundary.definition,
                "sequence_sha256": sequence_hash,
                "domain_counts": counts,
                "stratum": (boundary.domain_count, length_bin(length)),
            }
        )
        if len(selected) == total_count:
            break
    if len(selected) != total_count:
        raise RuntimeError(f"only {len(selected)} eligible unique proteins found, need {total_count}")

    strata: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    for item in selected:
        strata[item["stratum"]].append(item)
    rng = random.Random(args.seed + 1)
    for items in strata.values():
        rng.shuffle(items)
    quotas = validation_quotas(strata, args.validation_count)
    validation_targets = {
        item["record"].target
        for key, items in strata.items()
        for item in items[: quotas[key]]
    }

    manifest_rows = []
    for item in sorted(selected, key=lambda value: value["record"].target.lower()):
        record = item["record"]
        split = "validation" if record.target in validation_targets else "train"
        cache_path = cache_dir / f"{record.target}.npz"
        temporary = cache_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                target=np.asarray(record.target),
                sequence=np.asarray(record.sequence),
                aatype=encode_sequence(record.sequence),
                coordinates=record.coordinates.astype(np.float32),
                residue_numbers=record.residue_numbers.astype(np.int32),
                insertion_codes=record.insertion_codes.astype("U1"),
                domain_ids=record.domain_ids.astype(np.int32),
                motion_ids=record.motion_ids.astype(np.int32),
            )
        temporary.replace(cache_path)
        manifest_rows.append(
            {
                "target": record.target,
                "split": split,
                "length": len(record.sequence),
                "domain_count": len(item["domain_counts"]),
                "domain_sizes": ";".join(map(str, item["domain_counts"])),
                "definition": item["definition"],
                "uniprot_accession": record.uniprot_accession or "",
                "sequence_sha256": item["sequence_sha256"],
                "source_pdb": str(paths[record.target.lower()].resolve()),
                "cache_npz": str(cache_path.resolve()),
            }
        )

    manifest_path = args.output_dir / "tier1_mpdb_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    split_counts = {
        split: sum(row["split"] == split for row in manifest_rows)
        for split in ("train", "validation")
    }
    summary = {
        "schema_version": "iterblob_tier1_cohort_v1",
        "status": "complete",
        "seed": args.seed,
        "selection": {
            "length": [args.min_length, args.max_length],
            "domain_count": [2, 4],
            "min_domain_residues": args.min_domain_residues,
            "min_label_coverage": args.min_label_coverage,
            "deduplication": "exact_sequence_and_uniprot",
            "claim_boundary": "Pilot split; final paper split still requires 30% sequence clustering.",
            "reserved_test_manifest": (
                str(args.exclude_test_manifest.resolve())
                if args.exclude_test_manifest
                else None
            ),
            "reserved_test_uniprot_count": len(excluded_test_uniprot),
            "reserved_test_pdb_count": len(excluded_test_pdb),
        },
        "counts": split_counts,
        "strata": {
            f"domains_{key[0]}_length_{key[1]}": {
                "total": len(items),
                "validation": quotas[key],
            }
            for key, items in sorted(strata.items())
        },
        "exclusions_before_target_count": dict(sorted(exclusions.items())),
        "invalid_boundary_records": len(boundary_errors),
        "invalid_boundary_examples": boundary_errors[:10],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "boundary_source": str(args.boundaries.resolve()),
        "boundary_sha256": file_sha256(args.boundaries),
    }
    summary_path = args.output_dir / "tier1_mpdb_cohort_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
