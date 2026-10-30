"""Download a bounded AFDB coordinate-only pilot through the official API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEMO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEMO_ROOT
sys.path.insert(0, str(DEMO_ROOT))

from iterblob_demo.tier1_paired import stratified_unique_accession_sample  # noqa: E402


API_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
USER_AGENT = "IterBlob-Tier1/1.0 (research pilot; AFDB API)"


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
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT
        / "output"
        / "iterblob_tier1"
        / "paired_afdb_models_pilot",
    )
    parser.add_argument("--limit-train", type=int, default=16)
    parser.add_argument("--limit-validation", type=int, default=8)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--max-total-gib", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=20260817)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_cif_url(payload: object, accession: str) -> Dict[str, object]:
    predictions = payload if isinstance(payload, list) else [payload]
    candidates = [item for item in predictions if isinstance(item, dict)]
    if not candidates:
        raise ValueError(f"AFDB API returned no predictions for {accession}")
    accession_upper = accession.upper()

    def rank(item: dict):
        entity = str(item.get("modelEntityId", item.get("entryId", ""))).upper()
        accession_match = int(accession_upper in entity)
        first_fragment = int(entity.endswith("-F1"))
        return accession_match, first_fragment

    prediction = max(candidates, key=rank)
    cif_url = prediction.get("cifUrl")
    if not isinstance(cif_url, str) or not cif_url:
        raise ValueError(f"AFDB API response has no cifUrl for {accession}")
    parsed = urlparse(cif_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"AFDB returned an unsafe CIF URL: {cif_url}")
    return {"prediction": prediction, "cif_url": cif_url}


def read_json(url: str, timeout: float):
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def download_atomic(url: str, path: Path, timeout: float, byte_budget: int) -> int:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    temporary = path.with_suffix(path.suffix + ".tmp")
    written = 0
    decoded = temporary.with_suffix(temporary.suffix + ".gunzip")
    try:
        with urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                written += len(block)
                if written > byte_budget:
                    raise ValueError("AFDB pilot download exceeded the configured byte budget")
                handle.write(block)
        if is_gzip_file(temporary):
            decode_gzip_file(temporary, decoded, byte_budget)
            decoded.replace(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
        if decoded.exists():
            decoded.unlink()
    return path.stat().st_size


def is_gzip_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2:
        return False
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def decode_gzip_file(source: Path, destination: Path, byte_budget: int) -> int:
    written = 0
    with gzip.open(source, "rb") as compressed, destination.open("wb") as handle:
        while True:
            block = compressed.read(1024 * 1024)
            if not block:
                break
            written += len(block)
            if written > byte_budget:
                raise ValueError(
                    "decompressed AFDB file exceeded the configured byte budget"
                )
            handle.write(block)
    return written


def normalize_cached_download(path: Path, byte_budget: int) -> bool:
    """Atomically decode a cached HTTP-gzipped payload in place."""

    if not is_gzip_file(path):
        return False
    temporary = path.with_suffix(path.suffix + ".gunzip.tmp")
    try:
        decode_gzip_file(path, temporary, byte_budget)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
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
    if args.max_total_gib <= 0 or args.timeout_seconds <= 0 or args.workers <= 0:
        raise ValueError("download limits must be positive")
    if not args.cohort_manifest.is_file():
        raise FileNotFoundError(args.cohort_manifest)
    with args.cohort_manifest.open("r", encoding="utf-8", newline="") as handle:
        cohort_rows = list(csv.DictReader(handle))
    selected = stratified_unique_accession_sample(
        cohort_rows,
        split_limits,
        seed=args.selection_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    selected_accessions = []
    seen = set()
    for row in selected:
        accession = row["uniprot_accession"].upper()
        if accession not in seen:
            selected_accessions.append(accession)
            seen.add(accession)

    byte_budget = int(args.max_total_gib * 1024**3)
    per_file_budget = max(1, byte_budget // len(selected_accessions))

    def fetch(accession: str):
        try:
            api_url = API_TEMPLATE.format(accession=accession)
            metadata = extract_cif_url(
                read_json(api_url, args.timeout_seconds), accession
            )
            cif_url = str(metadata["cif_url"])
            model_path = model_dir / f"AF-{accession}-F1.cif"
            if not model_path.is_file():
                download_atomic(
                    cif_url,
                    model_path,
                    args.timeout_seconds,
                    per_file_budget,
                )
            else:
                normalize_cached_download(model_path, per_file_budget)
            if model_path.stat().st_size > per_file_budget:
                raise ValueError("AFDB model exceeds its share of the total byte budget")
            prediction = metadata["prediction"]
            return (
                {
                    "accession": accession,
                    "model_entity_id": prediction.get(
                        "modelEntityId", prediction.get("entryId", "")
                    ),
                    "sequence_start": prediction.get(
                        "sequenceStart", prediction.get("uniprotStart", "")
                    ),
                    "sequence_end": prediction.get(
                        "sequenceEnd", prediction.get("uniprotEnd", "")
                    ),
                    "latest_version": prediction.get("latestVersion", ""),
                    "api_url": api_url,
                    "cif_url": cif_url,
                    "model_cif": str(model_path.resolve()),
                    "model_sha256": sha256(model_path),
                    "model_bytes": model_path.stat().st_size,
                },
                None,
            )
        except Exception as error:
            return (
                None,
                {"accession": accession, "reason": f"{type(error).__name__}: {error}"},
            )

    model_rows = []
    exclusions = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(selected_accessions))) as pool:
        future_by_accession = {
            pool.submit(fetch, accession): accession for accession in selected_accessions
        }
        for future in as_completed(future_by_accession):
            model_row, exclusion = future.result()
            if model_row is not None:
                model_rows.append(model_row)
            if exclusion is not None:
                exclusions.append(exclusion)
    accession_order = {accession: index for index, accession in enumerate(selected_accessions)}
    model_rows.sort(key=lambda row: accession_order[row["accession"]])
    exclusions.sort(key=lambda row: accession_order[row["accession"]])

    manifest_path = args.output_dir / "tier1_afdb_model_manifest.csv"
    write_csv(manifest_path, model_rows)
    summary = {
        "schema_version": "iterblob_tier1_afdb_download_v1",
        "status": "complete" if not exclusions else "partial",
        "cohort_manifest": str(args.cohort_manifest.resolve()),
        "cohort_manifest_sha256": sha256(args.cohort_manifest),
        "selected_target_counts": {
            split: sum(row["split"] == split for row in selected)
            for split in split_limits
        },
        "selection": {
            "strategy": "length_domain_stratified_unique_accession_v1",
            "seed": args.selection_seed,
            "stratum_counts": {
                split: {
                    f"{length_bin}|domains_{domain_count}": sum(
                        row["split"] == split
                        and row.get("length_bin") == length_bin
                        and row.get("domain_count") == domain_count
                        for row in selected
                    )
                    for length_bin, domain_count in sorted(
                        {
                            (row.get("length_bin", "unknown"), row.get("domain_count", "unknown"))
                            for row in selected
                            if row["split"] == split
                        }
                    )
                }
                for split in split_limits
            },
        },
        "selected_accessions": len(selected_accessions),
        "downloaded_accessions": len(model_rows),
        "total_model_bytes": sum(int(row["model_bytes"]) for row in model_rows),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
        "exclusions": exclusions,
        "note": "Coordinate-only bounded pilot; no PAE or bulk AFDB database was downloaded.",
    }
    (args.output_dir / "tier1_afdb_download_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if model_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
