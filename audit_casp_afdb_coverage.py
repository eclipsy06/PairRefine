"""Audit local CASP multidomain targets for an AFDB-v6 external test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


DEMO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEMO_ROOT
sys.path.insert(0, str(DEMO_ROOT))

from iterblob_demo.tier1_paired import align_chain_to_afdb  # noqa: E402


DEFAULT_CASP_ROOT = DEMO_ROOT / "data" / "source" / "casp"
FASTA_BY_ROUND = {
    "CASP14": "casp14.seq.txt",
    "CASP15": "casp15.seq.txt",
    "CASP16": "casp16.T1.seq.txt",
}
UNIPROT_ACCESSION = re.compile(r"^[A-Z0-9]{6}(?:[A-Z0-9]{4})?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casp-root", type=Path, default=DEFAULT_CASP_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "output" / "iterblob_tier1" / "casp_external_test",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--min-identity", type=float, default=0.95)
    parser.add_argument("--min-query-coverage", type=float, default=0.95)
    return parser.parse_args()


def values(record: Dict[str, object], key: str) -> List[str]:
    value = record.get(key, [])
    return value if isinstance(value, list) else [value]


def normalize_sequence(value: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", str(value))).upper()


def read_fasta(path: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    header = None
    chunks: List[str] = []

    def flush() -> None:
        if header is None:
            return
        identifier = header.split()[0]
        records[identifier] = {
            "header": header,
            "sequence": normalize_sequence("".join(chunks)),
        }

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)
    flush()
    return records


def parse_entities(record: Dict[str, object]) -> List[dict]:
    entity_ids = values(record, "_entity_poly.entity_id")
    sequences = values(record, "_entity_poly.pdbx_seq_one_letter_code_can")
    strands = values(record, "_entity_poly.pdbx_strand_id")
    if not (len(entity_ids) == len(sequences) == len(strands)):
        raise ValueError("entity_poly columns have inconsistent lengths")
    return [
        {
            "entity_id": entity_id,
            "sequence": normalize_sequence(sequence),
            "label_chains": [item.strip() for item in strand.split(",") if item.strip()],
        }
        for entity_id, sequence, strand in zip(entity_ids, sequences, strands)
    ]


def parse_uniprot_references(record: Dict[str, object]) -> Dict[str, List[str]]:
    keys = [
        "_struct_ref.db_name",
        "_struct_ref.pdbx_db_accession",
        "_struct_ref.entity_id",
    ]
    columns = [values(record, key) for key in keys]
    if not columns[0]:
        return {}
    if len({len(column) for column in columns}) != 1:
        raise ValueError("struct_ref columns have inconsistent lengths")
    references: Dict[str, List[str]] = {}
    for database, accession, entity_id in zip(*columns):
        if database.upper() not in {"UNP", "UNIPROT"}:
            continue
        accession = accession.upper().strip()
        if UNIPROT_ACCESSION.fullmatch(accession):
            references.setdefault(entity_id, []).append(accession)
    return {
        entity_id: sorted(set(accessions))
        for entity_id, accessions in references.items()
    }


def header_accessions(header: str) -> List[str]:
    candidates = []
    description = header.split("|", 1)[0]
    for token in re.findall(r"\b[A-Z0-9]{6}(?:[A-Z0-9]{4})?\b", description.upper()):
        if any(character.isdigit() for character in token):
            candidates.append(token)
    return sorted(set(candidates))


def choose_entity(target_sequence: str, entities: Sequence[dict]) -> dict:
    candidates = []
    for entity in entities:
        sequence = entity["sequence"]
        if not sequence:
            continue
        mapping = align_chain_to_afdb(target_sequence, sequence)
        candidates.append(
            {
                **entity,
                "mapping": mapping,
                "length_difference": abs(len(sequence) - len(target_sequence)),
            }
        )
    if not candidates:
        raise ValueError("mmCIF contains no protein entity sequence")
    return max(
        candidates,
        key=lambda item: (
            item["mapping"].query_coverage,
            item["mapping"].identity,
            -item["length_difference"],
            -int(item["entity_id"]) if str(item["entity_id"]).isdigit() else 0,
        ),
    )


def choose_author_chain(
    record: Dict[str, object], entity_id: str, mapped_entity_indices: Iterable[int]
) -> dict:
    keys = [
        "_atom_site.group_PDB",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.auth_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    columns = [values(record, key) for key in keys]
    if len({len(column) for column in columns}) != 1:
        raise ValueError("atom_site columns have inconsistent lengths")
    mapped_positions = {int(index) + 1 for index in mapped_entity_indices if index >= 0}
    chain_positions: Dict[str, set] = {}
    for group, atom, alt, chain, entity, seq_id, model in zip(*columns):
        if group != "ATOM" or atom != "CA" or alt not in {".", "?", "A"}:
            continue
        if entity != str(entity_id) or model not in {"1", ".", "?"}:
            continue
        try:
            position = int(seq_id)
        except ValueError:
            continue
        if position in mapped_positions:
            chain_positions.setdefault(chain, set()).add(position)
    if not chain_positions:
        return {"author_chain": "", "native_ca_mapped": 0}
    chain, positions = min(
        chain_positions.items(), key=lambda item: (-len(item[1]), item[0])
    )
    return {"author_chain": chain, "native_ca_mapped": len(positions)}


def write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    metadata_path = args.casp_root / "metadata" / "casp_multidomain_targets.csv"
    sequence_dir = args.casp_root / "metadata" / "sequences"
    structure_dir = args.casp_root / "structure"
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    fasta_by_round = {
        round_name: read_fasta(sequence_dir / filename)
        for round_name, filename in FASTA_BY_ROUND.items()
    }

    rows = []
    for metadata in metadata_rows:
        target = metadata["target"]
        target_length = int(metadata["total_length"])
        if target_length > args.max_length:
            continue
        fasta = fasta_by_round[metadata["round"]].get(target)
        record_row = {
            "round": metadata["round"],
            "target": target,
            "pdb_id": metadata["pdb_id"],
            "target_length": target_length,
            "domain_count": int(metadata["domain_count"]),
            "domain_boundaries": metadata["domain_boundaries"],
            "sequence_length": 0,
            "entity_id": "",
            "entity_length": 0,
            "entity_identity": 0.0,
            "entity_query_coverage": 0.0,
            "author_chain": "",
            "native_ca_mapped": 0,
            "uniprot_accessions": "",
            "selected_accession": "",
            "accession_source": "",
            "status": "excluded",
            "reason": "",
        }
        try:
            if fasta is None:
                raise ValueError("official target sequence is missing")
            target_sequence = fasta["sequence"]
            record_row["sequence_length"] = len(target_sequence)
            if len(target_sequence) != target_length:
                raise ValueError(
                    f"official sequence length {len(target_sequence)} != {target_length}"
                )
            native_path = structure_dir / f"{metadata['pdb_id'].lower()}.cif"
            if not native_path.is_file():
                raise FileNotFoundError(native_path)
            mmcif = MMCIF2Dict(str(native_path))
            entity = choose_entity(target_sequence, parse_entities(mmcif))
            mapping = entity["mapping"]
            record_row.update(
                {
                    "entity_id": entity["entity_id"],
                    "entity_length": len(entity["sequence"]),
                    "entity_identity": mapping.identity,
                    "entity_query_coverage": mapping.query_coverage,
                }
            )
            chain = choose_author_chain(
                mmcif, entity["entity_id"], mapping.query_to_reference.tolist()
            )
            record_row.update(chain)
            if mapping.identity < args.min_identity:
                raise ValueError(f"entity identity {mapping.identity:.3f} below threshold")
            if mapping.query_coverage < args.min_query_coverage:
                raise ValueError(
                    f"entity query coverage {mapping.query_coverage:.3f} below threshold"
                )
            references = parse_uniprot_references(mmcif).get(entity["entity_id"], [])
            source = "mmcif_struct_ref"
            if not references:
                references = header_accessions(fasta["header"])
                source = "casp_fasta_header"
            record_row["uniprot_accessions"] = ";".join(references)
            record_row["accession_source"] = source if references else ""
            if len(references) != 1:
                raise ValueError(
                    "no unique UniProt accession"
                    if not references
                    else f"multiple UniProt accessions: {';'.join(references)}"
                )
            record_row["selected_accession"] = references[0]
            if int(record_row["native_ca_mapped"]) < 40:
                raise ValueError("fewer than 40 mapped native C-alpha atoms")
            record_row["status"] = "ready_for_afdb_download"
        except Exception as error:
            record_row["reason"] = f"{type(error).__name__}: {error}"
        rows.append(record_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "casp_afdb_coverage_audit.csv"
    write_csv(audit_path, rows)
    ready = [row for row in rows if row["status"] == "ready_for_afdb_download"]
    summary = {
        "schema_version": "iterblob_casp_afdb_coverage_audit_v1",
        "status": "complete",
        "claim_status": "coverage_audit_only_no_model_evaluation",
        "max_length": args.max_length,
        "candidate_targets": len(rows),
        "ready_targets": len(ready),
        "ready_unique_accessions": len(
            {row["selected_accession"] for row in ready}
        ),
        "audit_csv": str(audit_path.resolve()),
        "excluded": [
            {"target": row["target"], "reason": row["reason"]}
            for row in rows
            if row["status"] != "ready_for_afdb_download"
        ],
    }
    summary_path = args.output_dir / "casp_afdb_coverage_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
