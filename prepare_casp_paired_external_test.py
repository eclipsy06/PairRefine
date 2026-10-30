"""Build sequence-safe CASP/AFDB-v6 pairs for local P3a diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


DEMO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEMO_ROOT
sys.path.insert(0, str(DEMO_ROOT))

from audit_casp_afdb_coverage import (  # noqa: E402
    DEFAULT_CASP_ROOT,
    FASTA_BY_ROUND,
    choose_entity,
    parse_entities,
    read_fasta,
    values,
)
from iterblob_demo.domain_labels import parse_domain_boundaries  # noqa: E402
from iterblob_demo.structure_metrics import ca_structure_metrics  # noqa: E402
from iterblob_demo.tier1_paired import (  # noqa: E402
    align_chain_to_afdb,
    project_afdb_pae,
    project_afdb_to_chain,
)
from prepare_tier1_afdb_inputs import load_pae, parse_afdb_model  # noqa: E402


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INDEX = {letter: index for index, letter in enumerate(AMINO_ACIDS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casp-root", type=Path, default=DEFAULT_CASP_ROOT)
    parser.add_argument(
        "--audit",
        type=Path,
        default=WORKSPACE_ROOT
        / "output"
        / "iterblob_tier1"
        / "casp_external_test"
        / "coverage_le1024"
        / "casp_afdb_coverage_audit.csv",
    )
    parser.add_argument(
        "--afdb-manifest",
        type=Path,
        default=DEMO_ROOT / "data" / "source" / "afdb" / "casp_afdb_download_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT
        / "output"
        / "iterblob_tier1"
        / "casp_external_test"
        / "paired_13_reference_domains",
    )
    parser.add_argument("--min-identity", type=float, default=0.95)
    parser.add_argument("--min-query-coverage", type=float, default=0.90)
    parser.add_argument("--min-domain-residues", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def encode_sequence(sequence: str) -> np.ndarray:
    unknown = AA_TO_INDEX["X"]
    return np.asarray(
        [AA_TO_INDEX.get(letter, unknown) for letter in sequence], dtype=np.int64
    )


def fill_motion_labels(domain_ids: np.ndarray) -> np.ndarray:
    labels = np.asarray(domain_ids, dtype=np.int32)
    known = np.flatnonzero(labels >= 0)
    if not len(known):
        raise ValueError("at least one labeled domain residue is required")
    result = labels.copy()
    for index in np.flatnonzero(result < 0):
        nearest = known[np.argmin(np.abs(known - index))]
        result[index] = labels[nearest]
    return result


def parse_native_ca(
    record: Dict[str, object],
    *,
    entity_id: str,
    author_chain: str,
    target_to_entity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keys = [
        "_atom_site.group_PDB",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.auth_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.pdbx_PDB_model_num",
    ]
    columns = [values(record, key) for key in keys]
    if len({len(column) for column in columns}) != 1:
        raise ValueError("atom_site columns have inconsistent lengths")
    entity_to_target = {
        int(entity_index) + 1: target_index
        for target_index, entity_index in enumerate(target_to_entity.tolist())
        if entity_index >= 0
    }
    coordinates = np.zeros((len(target_to_entity), 3), dtype=np.float32)
    mask = np.zeros(len(target_to_entity), dtype=bool)
    for group, atom, alt, chain, entity, seq_id, x, y, z, model in zip(*columns):
        if group != "ATOM" or atom != "CA" or alt not in {".", "?", "A"}:
            continue
        if chain != author_chain or entity != str(entity_id):
            continue
        if model not in {"1", ".", "?"}:
            continue
        try:
            target_index = entity_to_target[int(seq_id)]
            coordinate = [float(x), float(y), float(z)]
        except (KeyError, ValueError):
            continue
        if mask[target_index]:
            continue
        coordinates[target_index] = coordinate
        mask[target_index] = True
    return coordinates, mask


def scalar_metric(metrics: dict, name: str):
    value = metrics.get(name)
    return "" if value is None else f"{float(value):.8f}"


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    for path in (args.audit, args.afdb_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    with args.audit.open("r", encoding="utf-8", newline="") as handle:
        audit_rows = [
            row
            for row in csv.DictReader(handle)
            if row["status"] == "ready_for_afdb_download"
        ]
    with args.afdb_manifest.open("r", encoding="utf-8", newline="") as handle:
        afdb_rows = list(csv.DictReader(handle))
    afdb_by_accession = {row["accession"].upper(): row for row in afdb_rows}
    fasta_by_round = {
        round_name: read_fasta(
            args.casp_root / "metadata" / "sequences" / filename
        )
        for round_name, filename in FASTA_BY_ROUND.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    example_dir = args.output_dir / "examples"
    partition_dir = args.output_dir / "reference_partitions"
    example_dir.mkdir(exist_ok=True)
    partition_dir.mkdir(exist_ok=True)
    manifest_rows = []
    partition_rows = []
    exclusions = []
    for row in sorted(audit_rows, key=lambda item: int(item["target_length"])):
        target = row["target"]
        accession = row["selected_accession"].upper()
        afdb = afdb_by_accession.get(accession)
        if afdb is None:
            exclusions.append({"target": target, "reason": "AFDB pair unavailable"})
            continue
        try:
            fasta = fasta_by_round[row["round"]][target]
            sequence = fasta["sequence"]
            length = len(sequence)
            native_path = args.casp_root / "structure" / f"{row['pdb_id'].lower()}.cif"
            native_record = MMCIF2Dict(str(native_path))
            entity = choose_entity(sequence, parse_entities(native_record))
            entity_mapping = entity["mapping"]
            if str(entity["entity_id"]) != row["entity_id"]:
                raise ValueError("audited entity selection changed")
            native_coordinates, native_mask = parse_native_ca(
                native_record,
                entity_id=row["entity_id"],
                author_chain=row["author_chain"],
                target_to_entity=entity_mapping.query_to_reference,
            )

            pae_path = Path(afdb["pae_json"])
            afdb_pae = load_pae(pae_path)
            model_path = Path(afdb["model_cif"])
            afdb_coordinates, afdb_mask, afdb_plddt, afdb_sequence, _ = (
                parse_afdb_model(model_path, len(afdb_pae))
            )
            afdb_mapping = align_chain_to_afdb(sequence, afdb_sequence)
            if afdb_mapping.identity < args.min_identity:
                raise ValueError(
                    f"AFDB identity {afdb_mapping.identity:.3f} below threshold"
                )
            if afdb_mapping.query_coverage < args.min_query_coverage:
                raise ValueError(
                    f"AFDB query coverage {afdb_mapping.query_coverage:.3f} below threshold"
                )
            coordinates, predicted_mask, plddt = project_afdb_to_chain(
                afdb_mapping, afdb_coordinates, afdb_mask, afdb_plddt
            )
            pae_maximum = max(float(afdb_pae.max()), 1.0)
            pae = project_afdb_pae(
                afdb_mapping, afdb_pae, fill_value=pae_maximum
            )
            labels, covered_residues = parse_domain_boundaries(
                row["domain_boundaries"], length
            )
            domain_ids = np.asarray(labels, dtype=np.int32)
            reference_partition = fill_motion_labels(domain_ids)
            residue_mask = predicted_mask & native_mask
            domain_counts = [
                int(np.sum(residue_mask & (domain_ids == domain_id)))
                for domain_id in range(int(row["domain_count"]))
            ]
            if min(domain_counts) < args.min_domain_residues:
                raise ValueError(
                    f"mapped domain counts {domain_counts} below minimum"
                )
            metrics = ca_structure_metrics(
                coordinates,
                residue_mask,
                native_coordinates,
                residue_mask,
                domain_ids,
            )
            inter_lddt = metrics.get("inter_domain_ca_lddt")
            if inter_lddt is None:
                difficulty = "unscored"
            elif inter_lddt < 0.80:
                difficulty = "hard"
            elif inter_lddt < 0.90:
                difficulty = "medium"
            else:
                difficulty = "easy"

            cache_path = example_dir / f"{target}.paired.npz"
            temporary = cache_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    target=np.asarray(target),
                    sequence=np.asarray(sequence),
                    aatype=encode_sequence(sequence),
                    coordinates=coordinates.astype(np.float32),
                    native_coordinates=native_coordinates.astype(np.float32),
                    residue_mask=residue_mask.astype(bool),
                    predicted_mask=predicted_mask.astype(bool),
                    native_mask=native_mask.astype(bool),
                    plddt=plddt.astype(np.float32),
                    pae=pae.astype(np.float32),
                    pae_maximum=np.asarray(pae_maximum, dtype=np.float32),
                    domain_ids=domain_ids,
                    motion_ids=reference_partition.astype(np.int32),
                    afdb_index_for_target=afdb_mapping.query_to_reference,
                    entity_index_for_target=entity_mapping.query_to_reference,
                )
            temporary.replace(cache_path)

            partition_path = partition_dir / f"{target}.reference_domains.npz"
            temporary_partition = partition_path.with_suffix(".npz.tmp")
            with temporary_partition.open("wb") as handle:
                np.savez_compressed(
                    handle, partition_ids=reference_partition.astype(np.int32)
                )
            temporary_partition.replace(partition_path)

            manifest_rows.append(
                {
                    "target": target,
                    "split": "validation",
                    "round": row["round"],
                    "pdb_id": row["pdb_id"].upper(),
                    "author_chain": row["author_chain"],
                    "uniprot_accession": accession,
                    "length": length,
                    "domain_count": int(row["domain_count"]),
                    "domain_boundaries": row["domain_boundaries"],
                    "difficulty_bin": difficulty,
                    "mapped_residues": int(residue_mask.sum()),
                    "domain_mapped_counts": ";".join(map(str, domain_counts)),
                    "official_domain_covered_residues": covered_residues,
                    "alignment_identity": f"{afdb_mapping.identity:.8f}",
                    "query_coverage": f"{afdb_mapping.query_coverage:.8f}",
                    "baseline_tm_score": scalar_metric(
                        metrics, "fixed_correspondence_ca_tm_score"
                    ),
                    "baseline_global_lddt": scalar_metric(
                        metrics, "global_ca_lddt"
                    ),
                    "baseline_inter_domain_lddt": scalar_metric(
                        metrics, "inter_domain_ca_lddt"
                    ),
                    "cache_npz": str(cache_path.resolve()),
                    "native_cif": str(native_path.resolve()),
                    "native_sha256": sha256(native_path),
                    "afdb_model_cif": str(model_path.resolve()),
                    "afdb_model_sha256": afdb["model_sha256"],
                    "afdb_pae_json": str(pae_path.resolve()),
                    "afdb_pae_sha256": afdb["pae_sha256"],
                    "afdb_version": afdb["latest_version"],
                }
            )
            partition_rows.append(
                {
                    "target": target,
                    "split": "validation",
                    "sequence_sha256": sequence_sha256(sequence),
                    "partition_npz": str(partition_path.resolve()),
                    "provider": "official_casp_reference_domains",
                    "provider_version": "reference_control_v1",
                    "primitive_count": int(row["domain_count"]),
                    "mapped_residues": int(residue_mask.sum()),
                    "coverage": f"{float(residue_mask.mean()):.8f}",
                    "provider_confidence": "",
                    "provider_chopping_afdb_order": row["domain_boundaries"],
                    "input_source": "official_domain_definition_native_label_control",
                }
            )
        except Exception as error:
            exclusions.append(
                {"target": target, "reason": f"{type(error).__name__}: {error}"}
            )

    if not manifest_rows:
        raise RuntimeError("no CASP paired examples were prepared")
    manifest_path = args.output_dir / "casp_paired_manifest.csv"
    partition_manifest_path = args.output_dir / "reference_partition_manifest.csv"
    write_csv(manifest_path, manifest_rows)
    write_csv(partition_manifest_path, partition_rows)
    summary = {
        "schema_version": "iterblob_casp_paired_external_test_v1",
        "status": "complete" if not exclusions else "partial",
        "claim_status": "post_casp_afdb_v6_reference_domain_diagnostic_only",
        "temporal_leakage_risk": True,
        "native_labels_used_for_partition": True,
        "native_coordinates_used_for_model_features": False,
        "included_targets": len(manifest_rows),
        "included_unique_pdb": len({row["pdb_id"] for row in manifest_rows}),
        "included_unique_accessions": len(
            {row["uniprot_accession"] for row in manifest_rows}
        ),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "partition_manifest": str(partition_manifest_path.resolve()),
        "partition_manifest_sha256": sha256(partition_manifest_path),
        "exclusions": exclusions,
    }
    summary_path = args.output_dir / "casp_paired_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
