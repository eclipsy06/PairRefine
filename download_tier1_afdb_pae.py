"""Download bounded PAE files for accessions in an existing paired manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys
from typing import List
from urllib.parse import urlparse


DEMO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_ROOT))

from download_tier1_afdb_models import (  # noqa: E402
    API_TEMPLATE,
    download_atomic,
    extract_cif_url,
    read_json,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-total-gib", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_pae_url(payload: object, accession: str) -> tuple[str, dict]:
    prediction = extract_cif_url(payload, accession)["prediction"]
    pae_url = prediction.get("paeDocUrl")
    if not isinstance(pae_url, str) or not pae_url:
        raise ValueError(f"AFDB API response has no paeDocUrl for {accession}")
    parsed = urlparse(pae_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"AFDB returned an unsafe PAE URL: {pae_url}")
    return pae_url, prediction


def main() -> int:
    args = parse_args()
    if not args.paired_manifest.is_file():
        raise FileNotFoundError(args.paired_manifest)
    if args.max_total_gib <= 0 or args.timeout_seconds <= 0 or args.workers <= 0:
        raise ValueError("download limits must be positive")
    paired_rows = list(
        csv.DictReader(args.paired_manifest.open("r", encoding="utf-8", newline=""))
    )
    accessions = []
    seen = set()
    for row in paired_rows:
        accession = row["uniprot_accession"].upper()
        if accession not in seen:
            accessions.append(accession)
            seen.add(accession)
    if not accessions:
        raise ValueError("paired manifest contains no accessions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pae_dir = args.output_dir / "pae"
    pae_dir.mkdir(exist_ok=True)
    total_budget = int(args.max_total_gib * 1024**3)
    per_file_budget = max(1, total_budget // len(accessions))

    def fetch(accession: str):
        try:
            api_url = API_TEMPLATE.format(accession=accession)
            pae_url, prediction = extract_pae_url(
                read_json(api_url, args.timeout_seconds), accession
            )
            path = pae_dir / f"AF-{accession}-F1-pae.json"
            if not path.is_file():
                download_atomic(pae_url, path, args.timeout_seconds, per_file_budget)
            if path.stat().st_size > per_file_budget:
                raise ValueError("PAE file exceeds its share of the total byte budget")
            return (
                {
                    "accession": accession,
                    "model_entity_id": prediction.get(
                        "modelEntityId", prediction.get("entryId", "")
                    ),
                    "latest_version": prediction.get("latestVersion", ""),
                    "api_url": api_url,
                    "pae_url": pae_url,
                    "pae_json": str(path.resolve()),
                    "pae_sha256": sha256(path),
                    "pae_bytes": path.stat().st_size,
                },
                None,
            )
        except Exception as error:
            return None, {
                "accession": accession,
                "reason": f"{type(error).__name__}: {error}",
            }

    rows = []
    errors = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(accessions))) as pool:
        futures = {pool.submit(fetch, accession): accession for accession in accessions}
        for future in as_completed(futures):
            row, error = future.result()
            if row is not None:
                rows.append(row)
            if error is not None:
                errors.append(error)
    order = {accession: index for index, accession in enumerate(accessions)}
    rows.sort(key=lambda row: order[row["accession"]])
    errors.sort(key=lambda row: order[row["accession"]])
    total_bytes = sum(int(row["pae_bytes"]) for row in rows)
    if total_bytes > total_budget:
        raise ValueError("PAE pilot exceeded the configured total byte budget")

    manifest = args.output_dir / "tier1_afdb_pae_manifest.csv"
    write_csv(manifest, rows)
    summary = {
        "schema_version": "iterblob_tier1_afdb_pae_download_v1",
        "status": "complete" if not errors else "partial",
        "paired_manifest": str(args.paired_manifest.resolve()),
        "paired_manifest_sha256": sha256(args.paired_manifest),
        "requested_accessions": len(accessions),
        "downloaded_accessions": len(rows),
        "total_pae_bytes": total_bytes,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest) if rows else None,
        "errors": errors,
        "note": "PAE-only bounded download; no coordinates, MSA, or bulk AFDB database.",
    }
    (args.output_dir / "tier1_afdb_pae_download_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if rows and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
