"""Download bounded AFDB structure/PAE pairs for the audited CASP targets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import shutil
from pathlib import Path
import subprocess
import sys
from typing import List


DEMO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEMO_ROOT
sys.path.insert(0, str(DEMO_ROOT))

from download_tier1_afdb_models import (  # noqa: E402
    API_TEMPLATE,
    extract_cif_url,
    normalize_cached_download,
    sha256,
)
from download_tier1_afdb_pae import extract_pae_url  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=WORKSPACE_ROOT
        / "output"
        / "iterblob_tier1"
        / "casp_external_test"
        / "casp_afdb_coverage_audit.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEMO_ROOT / "data" / "source" / "afdb",
    )
    parser.add_argument("--max-total-gib", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def curl_read_json(url: str, timeout: float):
    completed = subprocess.run(
        [
            shutil.which("curl") or shutil.which("curl.exe") or "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--max-time",
            str(int(timeout)),
            url,
        ],
        check=True,
        capture_output=True,
    )
    return json.loads(completed.stdout.decode("utf-8"))


def curl_download_atomic(url: str, path: Path, timeout: float, byte_budget: int) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        subprocess.run(
            [
                shutil.which("curl") or shutil.which("curl.exe") or "curl",
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--max-time",
                str(int(timeout)),
                "--output",
                str(temporary),
                url,
            ],
            check=True,
        )
        if temporary.stat().st_size > byte_budget:
            raise ValueError("AFDB file exceeded its configured byte budget")
        normalize_cached_download(temporary, byte_budget)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path.stat().st_size


def main() -> int:
    args = parse_args()
    if not args.audit.is_file():
        raise FileNotFoundError(args.audit)
    if args.max_total_gib <= 0 or args.timeout_seconds <= 0 or args.workers <= 0:
        raise ValueError("download limits must be positive")
    with args.audit.open("r", encoding="utf-8", newline="") as handle:
        audit_rows = [
            row
            for row in csv.DictReader(handle)
            if row["status"] == "ready_for_afdb_download"
        ]
    accessions = sorted({row["selected_accession"].upper() for row in audit_rows})
    if not accessions:
        raise ValueError("audit contains no downloadable accessions")

    args.output_root.mkdir(parents=True, exist_ok=True)
    total_budget = int(args.max_total_gib * 1024**3)
    per_file_budget = max(1, total_budget // (2 * len(accessions)))

    def fetch(accession: str):
        try:
            api_url = API_TEMPLATE.format(accession=accession)
            payload = curl_read_json(api_url, args.timeout_seconds)
            model_metadata = extract_cif_url(payload, accession)
            pae_url, prediction = extract_pae_url(payload, accession)
            model_url = str(model_metadata["cif_url"])
            latest_version = int(prediction.get("latestVersion", 0))
            if latest_version <= 0:
                raise ValueError("AFDB API did not report a valid latestVersion")
            accession_dir = args.output_root / accession
            accession_dir.mkdir(exist_ok=True)
            model_path = (
                accession_dir
                / f"AF-{accession}-F1-model_v{latest_version}.cif"
            )
            pae_path = (
                accession_dir
                / f"AF-{accession}-F1-predicted_aligned_error_v{latest_version}.json"
            )
            if not model_path.is_file():
                curl_download_atomic(
                    model_url, model_path, args.timeout_seconds, per_file_budget
                )
            if not pae_path.is_file():
                curl_download_atomic(
                    pae_url, pae_path, args.timeout_seconds, per_file_budget
                )
            combined_bytes = model_path.stat().st_size + pae_path.stat().st_size
            if combined_bytes > 2 * per_file_budget:
                raise ValueError("AFDB pair exceeds its share of the byte budget")
            return (
                {
                    "accession": accession,
                    "model_entity_id": prediction.get(
                        "modelEntityId", prediction.get("entryId", "")
                    ),
                    "latest_version": latest_version,
                    "sequence_start": prediction.get(
                        "sequenceStart", prediction.get("uniprotStart", "")
                    ),
                    "sequence_end": prediction.get(
                        "sequenceEnd", prediction.get("uniprotEnd", "")
                    ),
                    "api_url": api_url,
                    "model_url": model_url,
                    "pae_url": pae_url,
                    "model_cif": str(model_path.resolve()),
                    "model_sha256": sha256(model_path),
                    "model_bytes": model_path.stat().st_size,
                    "pae_json": str(pae_path.resolve()),
                    "pae_sha256": sha256(pae_path),
                    "pae_bytes": pae_path.stat().st_size,
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
    total_bytes = sum(
        int(row["model_bytes"]) + int(row["pae_bytes"]) for row in rows
    )
    if total_bytes > total_budget:
        raise ValueError("CASP AFDB download exceeded the total byte budget")

    manifest_path = args.output_root / "casp_afdb_download_manifest.csv"
    write_csv(manifest_path, rows)
    summary = {
        "schema_version": "iterblob_casp_afdb_download_v1",
        "status": "complete" if not errors else "partial",
        "claim_status": "current_afdb_post_casp_baseline_with_temporal_leakage_risk",
        "audit": str(args.audit.resolve()),
        "requested_accessions": len(accessions),
        "downloaded_accessions": len(rows),
        "total_bytes": total_bytes,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path) if rows else None,
        "errors": errors,
    }
    summary_path = args.output_root / "casp_afdb_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if rows and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
