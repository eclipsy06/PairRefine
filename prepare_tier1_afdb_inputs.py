"""Convert local AFDB v6 mmCIF/PAE pairs into dependency-light Tier-1 inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


DEMO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEMO_ROOT
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K",
}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INDEX = {letter: index for index, letter in enumerate(AMINO_ACIDS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--afdb-root",
        type=Path,
        default=WORKSPACE_ROOT / "output" / "stoploss_9_1" / "inputs" / "afdb_v6",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "output" / "iterblob_tier1" / "afdb_inputs",
    )
    parser.add_argument("--min-length", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=384)
    return parser.parse_args()


def values(record: Dict[str, Any], key: str) -> List[str]:
    value = record.get(key, [])
    return value if isinstance(value, list) else [value]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pae(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = payload[0]
    matrix = np.asarray(payload["predicted_aligned_error"], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"PAE is not square: {path}")
    return matrix


def parse_afdb_model(path: Path, length: Optional[int] = None):
    record = MMCIF2Dict(str(path))
    keys = [
        "_atom_site.group_PDB",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_seq_id",
        "_atom_site.label_comp_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_PDB_model_num",
    ]
    columns = [values(record, key) for key in keys]
    if len({len(column) for column in columns}) != 1:
        raise ValueError(f"inconsistent mmCIF columns: {path}")
    if length is None:
        observed_indices = []
        for row in zip(*columns):
            group, atom, altloc, seq_text, _, _, _, _, _, model = row
            if group != "ATOM" or atom != "CA" or altloc not in {".", "?", "A"}:
                continue
            if model not in {"1", ".", "?"}:
                continue
            try:
                observed_indices.append(int(seq_text))
            except ValueError:
                continue
        if not observed_indices:
            raise ValueError(f"no model-1 C-alpha atoms: {path}")
        length = max(observed_indices)
    if length <= 0:
        raise ValueError("length must be positive")
    coordinates = np.zeros((length, 3), dtype=np.float32)
    plddt = np.zeros(length, dtype=np.float32)
    sequence = np.full(length, "X", dtype="U1")
    mask = np.zeros(length, dtype=np.bool_)
    for row in zip(*columns):
        group, atom, altloc, seq_text, residue, x, y, z, confidence, model = row
        if group != "ATOM" or atom != "CA" or altloc not in {".", "?", "A"}:
            continue
        if model not in {"1", ".", "?"}:
            continue
        try:
            index = int(seq_text) - 1
        except ValueError:
            continue
        if not 0 <= index < length or mask[index]:
            continue
        coordinates[index] = [float(x), float(y), float(z)]
        plddt[index] = float(confidence)
        sequence[index] = THREE_TO_ONE.get(residue, "X")
        mask[index] = True
    if int(mask.sum()) < max(20, int(0.8 * length)):
        raise ValueError(f"insufficient mapped C-alpha atoms: {int(mask.sum())}/{length}")
    sequence_text = "".join(sequence.tolist())
    aatype = np.asarray([AA_TO_INDEX.get(letter, 20) for letter in sequence_text], dtype=np.int64)
    return coordinates, mask, plddt, sequence_text, aatype


def main() -> int:
    args = parse_args()
    if not args.afdb_root.is_dir():
        raise FileNotFoundError(args.afdb_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    example_dir = args.output_dir / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    exclusions = []
    for accession_dir in sorted(path for path in args.afdb_root.iterdir() if path.is_dir()):
        accession = accession_dir.name
        model_path = accession_dir / f"AF-{accession}-F1-model_v6.cif"
        pae_path = accession_dir / f"AF-{accession}-F1-predicted_aligned_error_v6.json"
        if not model_path.is_file() or not pae_path.is_file():
            exclusions.append({"accession": accession, "reason": "missing_model_or_pae"})
            continue
        try:
            pae = load_pae(pae_path)
            length = pae.shape[0]
            if not args.min_length <= length <= args.max_length:
                exclusions.append({"accession": accession, "reason": f"length_{length}"})
                continue
            coordinates, mask, plddt, sequence, aatype = parse_afdb_model(model_path, length)
            output_path = example_dir / f"{accession}.npz"
            temporary = output_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    accession=np.asarray(accession),
                    sequence=np.asarray(sequence),
                    aatype=aatype,
                    coordinates=coordinates,
                    residue_mask=mask,
                    plddt=plddt,
                    pae=pae,
                )
            temporary.replace(output_path)
        except Exception as error:
            exclusions.append({"accession": accession, "reason": f"{type(error).__name__}: {error}"})
            continue
        rows.append(
            {
                "accession": accession,
                "length": length,
                "mean_plddt": float(plddt[mask].mean()),
                "input_npz": str(output_path.resolve()),
                "model_cif": str(model_path.resolve()),
                "model_sha256": sha256(model_path),
                "pae_json": str(pae_path.resolve()),
                "pae_sha256": sha256(pae_path),
            }
        )
    manifest_path = args.output_dir / "tier1_afdb_input_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "schema_version": "iterblob_tier1_afdb_inputs_v1",
        "status": "complete",
        "included": len(rows),
        "excluded": len(exclusions),
        "length_range": [args.min_length, args.max_length],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "exclusions": exclusions,
    }
    (args.output_dir / "tier1_afdb_input_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
