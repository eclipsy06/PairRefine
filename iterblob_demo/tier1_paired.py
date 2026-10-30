"""Sequence-safe pairing of AFDB predictions with experimental MPDB chains."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import parasail
except ImportError:  # Optional on lightweight local evaluation environments.
    parasail = None


def stratified_unique_accession_sample(
    rows: Iterable[dict],
    split_limits: Dict[str, int],
    *,
    seed: int,
) -> List[dict]:
    """Deterministically sample unique accessions across length/domain strata."""

    rows = list(rows)
    selected = []
    for split_index, (split, limit) in enumerate(split_limits.items()):
        if limit <= 0:
            raise ValueError("split limits must be positive")
        rng = random.Random(seed + split_index * 1_000_003)
        candidates = [
            row
            for row in rows
            if row.get("split") == split and row.get("uniprot_accession")
        ]
        by_accession: Dict[str, List[dict]] = {}
        for row in candidates:
            by_accession.setdefault(row["uniprot_accession"].upper(), []).append(row)
        representatives = []
        for accession in sorted(by_accession):
            options = sorted(by_accession[accession], key=lambda item: item["target"])
            representatives.append(options[rng.randrange(len(options))])
        if len(representatives) < limit:
            raise ValueError(
                f"split {split} has only {len(representatives)} unique accessions"
            )

        strata: Dict[Tuple[str, str], List[dict]] = {}
        for row in representatives:
            key = (str(row.get("length_bin", "unknown")), str(row.get("domain_count", "unknown")))
            strata.setdefault(key, []).append(row)
        for key in sorted(strata):
            rng.shuffle(strata[key])

        keys = sorted(strata)
        allocation = {key: 0 for key in keys}
        if limit >= len(keys):
            for key in keys:
                allocation[key] = 1
        remaining = limit - sum(allocation.values())
        while remaining > 0:
            available = {
                key: len(strata[key]) - allocation[key]
                for key in keys
                if len(strata[key]) > allocation[key]
            }
            if not available:
                raise ValueError(f"cannot allocate {limit} samples for split {split}")
            total_available = sum(available.values())
            ideal = {
                key: remaining * available[key] / total_available for key in available
            }
            added = 0
            for key in available:
                count = min(available[key], int(math.floor(ideal[key])))
                allocation[key] += count
                added += count
            remaining -= added
            if remaining == 0:
                break
            ranked = sorted(
                available,
                key=lambda key: (ideal[key] - math.floor(ideal[key]), available[key], key),
                reverse=True,
            )
            for key in ranked:
                if remaining == 0:
                    break
                if allocation[key] < len(strata[key]):
                    allocation[key] += 1
                    remaining -= 1

        split_rows = []
        for key in keys:
            split_rows.extend(strata[key][: allocation[key]])
        rng.shuffle(split_rows)
        selected.extend(split_rows)
    return selected


@dataclass(frozen=True)
class SequenceMapping:
    """Local-alignment mapping from an experimental chain to an AFDB sequence."""

    query_to_reference: np.ndarray
    identity: float
    query_coverage: float
    aligned_columns: int
    matches: int


def align_chain_to_afdb(query: str, reference: str) -> SequenceMapping:
    """Map chain residue indices to AFDB indices with Smith-Waterman alignment."""

    query = query.strip().upper()
    reference = reference.strip().upper()
    if not query or not reference:
        raise ValueError("query and reference sequences must be non-empty")
    if parasail is None:
        return _align_chain_to_afdb_biopython(query, reference)
    result = parasail.sw_trace_striped_16(
        query, reference, 10, 1, parasail.blosum62
    )
    aligned_query = result.traceback.query
    aligned_reference = result.traceback.ref
    if not aligned_query or len(aligned_query) != len(aligned_reference):
        raise ValueError("local alignment did not return a valid traceback")

    query_residues = sum(letter != "-" for letter in aligned_query)
    reference_residues = sum(letter != "-" for letter in aligned_reference)
    query_index = int(result.end_query) - query_residues + 1
    reference_index = int(result.end_ref) - reference_residues + 1
    if query_index < 0 or reference_index < 0:
        raise ValueError("local alignment returned invalid start coordinates")

    mapping = np.full(len(query), -1, dtype=np.int32)
    matches = 0
    mapped_query = 0
    for query_letter, reference_letter in zip(aligned_query, aligned_reference):
        current_query = query_index if query_letter != "-" else None
        current_reference = reference_index if reference_letter != "-" else None
        if current_query is not None and current_reference is not None:
            if current_query >= len(query) or current_reference >= len(reference):
                raise ValueError("alignment traceback exceeds sequence length")
            mapping[current_query] = current_reference
            mapped_query += 1
            matches += int(query_letter == reference_letter)
        if current_query is not None:
            query_index += 1
        if current_reference is not None:
            reference_index += 1

    aligned_columns = len(aligned_query)
    return SequenceMapping(
        query_to_reference=mapping,
        identity=float(matches / aligned_columns),
        query_coverage=float(mapped_query / len(query)),
        aligned_columns=aligned_columns,
        matches=matches,
    )


def _align_chain_to_afdb_biopython(query: str, reference: str) -> SequenceMapping:
    """Biopython fallback matching the production Smith-Waterman protocol."""

    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError as error:
        raise ImportError(
            "sequence alignment requires either parasail or biopython"
        ) from error

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -1.0
    alignments = aligner.align(query, reference)
    if not len(alignments):
        raise ValueError("local alignment did not return a valid traceback")
    coordinates = np.asarray(alignments[0].coordinates, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[0] != 2:
        raise ValueError("local alignment returned invalid coordinates")

    mapping = np.full(len(query), -1, dtype=np.int32)
    matches = 0
    mapped_query = 0
    aligned_columns = 0
    for column in range(coordinates.shape[1] - 1):
        query_start, query_end = coordinates[0, column : column + 2]
        reference_start, reference_end = coordinates[1, column : column + 2]
        query_count = int(query_end - query_start)
        reference_count = int(reference_end - reference_start)
        if query_count and reference_count:
            if query_count != reference_count:
                raise ValueError("alignment contains an invalid diagonal segment")
            query_indices = np.arange(query_start, query_end, dtype=np.int32)
            reference_indices = np.arange(
                reference_start, reference_end, dtype=np.int32
            )
            mapping[query_indices] = reference_indices
            mapped_query += query_count
            matches += sum(
                query[int(query_index)] == reference[int(reference_index)]
                for query_index, reference_index in zip(
                    query_indices.tolist(), reference_indices.tolist()
                )
            )
        aligned_columns += max(query_count, reference_count)
    if not aligned_columns:
        raise ValueError("local alignment did not align any residues")
    return SequenceMapping(
        query_to_reference=mapping,
        identity=float(matches / aligned_columns),
        query_coverage=float(mapped_query / len(query)),
        aligned_columns=aligned_columns,
        matches=matches,
    )


def project_afdb_to_chain(
    mapping: SequenceMapping,
    afdb_coordinates: np.ndarray,
    afdb_mask: np.ndarray,
    afdb_plddt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather AFDB coordinates and confidence into experimental-chain order."""

    afdb_coordinates = np.asarray(afdb_coordinates, dtype=np.float32)
    afdb_mask = np.asarray(afdb_mask, dtype=bool)
    afdb_plddt = np.asarray(afdb_plddt, dtype=np.float32)
    reference_length = afdb_coordinates.shape[0]
    if afdb_coordinates.shape != (reference_length, 3):
        raise ValueError("AFDB coordinates must have shape [L, 3]")
    if afdb_mask.shape != (reference_length,) or afdb_plddt.shape != (
        reference_length,
    ):
        raise ValueError("AFDB mask and pLDDT must have shape [L]")

    chain_length = len(mapping.query_to_reference)
    coordinates = np.zeros((chain_length, 3), dtype=np.float32)
    confidence = np.zeros(chain_length, dtype=np.float32)
    residue_mask = np.zeros(chain_length, dtype=bool)
    for chain_index, afdb_index in enumerate(mapping.query_to_reference.tolist()):
        if afdb_index < 0:
            continue
        if afdb_index >= reference_length:
            raise ValueError("sequence mapping exceeds AFDB coordinate length")
        if not afdb_mask[afdb_index]:
            continue
        coordinates[chain_index] = afdb_coordinates[afdb_index]
        confidence[chain_index] = afdb_plddt[afdb_index]
        residue_mask[chain_index] = True
    return coordinates, residue_mask, confidence


def parse_afdb_pae(path: Path) -> Tuple[np.ndarray, float]:
    """Read the current AFDB PAE JSON matrix with legacy-field fallback."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload[0] if isinstance(payload, list) else payload
    if not isinstance(record, dict):
        raise ValueError("AFDB PAE JSON must contain an object")
    values = record.get("predicted_aligned_error", record.get("distance"))
    if values is None:
        raise ValueError("AFDB PAE JSON has no predicted_aligned_error field")
    pae = np.asarray(values, dtype=np.float32)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or not pae.shape[0]:
        raise ValueError("AFDB PAE must be a non-empty square matrix")
    if not np.isfinite(pae).all() or np.any(pae < 0):
        raise ValueError("AFDB PAE contains invalid values")
    maximum = float(record.get("max_predicted_aligned_error", float(pae.max())))
    if not np.isfinite(maximum) or maximum <= 0:
        maximum = max(float(pae.max()), 1.0)
    return pae, maximum


def project_afdb_pae(
    mapping: SequenceMapping,
    afdb_pae: np.ndarray,
    *,
    fill_value: float,
) -> np.ndarray:
    """Project residue-pair uncertainty into experimental-chain order."""

    afdb_pae = np.asarray(afdb_pae, dtype=np.float32)
    reference_length = afdb_pae.shape[0]
    if afdb_pae.shape != (reference_length, reference_length):
        raise ValueError("AFDB PAE must be square")
    indices = mapping.query_to_reference.astype(np.int64, copy=False)
    valid = (indices >= 0) & (indices < reference_length)
    projected = np.full((len(indices), len(indices)), fill_value, dtype=np.float32)
    selected = np.flatnonzero(valid)
    if selected.size:
        reference = indices[selected]
        projected[np.ix_(selected, selected)] = afdb_pae[np.ix_(reference, reference)]
    return projected
