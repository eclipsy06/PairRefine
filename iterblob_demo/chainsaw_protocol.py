"""Utilities for the version-pinned Chainsaw/CATH experiment protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


STANDARD_AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

ResidueKey = Tuple[int, str]


def _normalise_insert(value: str) -> str:
    value = value.strip()
    return "" if value in {"", "-", ".", "?"} else value


@dataclass(frozen=True)
class CathSegment:
    start_chain: str
    start: ResidueKey
    end_chain: str
    end: ResidueKey

    def contains(self, chain: str, residue: ResidueKey) -> bool:
        return (
            chain.casefold() == self.start_chain.casefold()
            and chain.casefold() == self.end_chain.casefold()
            and self.start <= self.end
            and self.start <= residue <= self.end
        )


@dataclass(frozen=True)
class CathDomain:
    domain_id: str
    segments: Tuple[CathSegment, ...]


def parse_cath_domall_line(line: str) -> Tuple[str, List[CathDomain]]:
    """Parse one CATH Domall 2.0 line, excluding fragment records."""

    tokens = line.split()
    if len(tokens) < 4 or not tokens[1].startswith("D") or not tokens[2].startswith("F"):
        raise ValueError(f"invalid CATH Domall line: {line[:100]}")
    chain_id = tokens[0]
    domain_count = int(tokens[1][1:])
    cursor = 3
    domains: List[CathDomain] = []
    for domain_index in range(domain_count):
        if cursor >= len(tokens):
            raise ValueError(f"truncated CATH Domall line for {chain_id}")
        segment_count = int(tokens[cursor])
        cursor += 1
        segments = []
        for _ in range(segment_count):
            if cursor + 5 >= len(tokens):
                raise ValueError(f"truncated CATH segment for {chain_id}")
            start_chain, start_number, start_insert = tokens[cursor : cursor + 3]
            end_chain, end_number, end_insert = tokens[cursor + 3 : cursor + 6]
            cursor += 6
            segment = CathSegment(
                start_chain=start_chain,
                start=(int(start_number), _normalise_insert(start_insert)),
                end_chain=end_chain,
                end=(int(end_number), _normalise_insert(end_insert)),
            )
            segments.append(segment)
        domains.append(
            CathDomain(
                domain_id=f"{chain_id}{domain_index + 1:02d}",
                segments=tuple(segments),
            )
        )
    return chain_id, domains


def read_cath_domall(
    path: Path, target_chains: Optional[Iterable[str]] = None
) -> Dict[str, List[CathDomain]]:
    targets = {value.casefold() for value in target_chains} if target_chains else None
    records: Dict[str, List[CathDomain]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            chain_id = text.split(None, 1)[0]
            if targets is not None and chain_id.casefold() not in targets:
                continue
            try:
                parsed_chain, domains = parse_cath_domall_line(text)
            except Exception as error:
                raise ValueError(f"CATH Domall parse error at line {line_number}") from error
            records[parsed_chain.casefold()] = domains
    return records


def observed_pdb_residues(path: Path, chain: str) -> List[ResidueKey]:
    """Read the same standard-amino-acid CA residue set used by Chainsaw."""

    requested = " " if chain == "0" else chain
    residues: List[ResidueKey] = []
    seen = set()
    with path.open("r", encoding="ascii", errors="replace") as source:
        for line in source:
            if not line.startswith("ATOM") or len(line) < 27:
                continue
            if line[12:16].strip() != "CA" or line[17:20].strip() not in STANDARD_AA:
                continue
            if line[21].casefold() != requested.casefold():
                continue
            try:
                key = (int(line[22:26]), _normalise_insert(line[26]))
            except ValueError:
                continue
            if key not in seen:
                seen.add(key)
                residues.append(key)
    if not residues:
        raise ValueError(f"no standard CA residues for chain {chain!r} in {path}")
    return residues


def map_cath_domains(
    residues: Sequence[ResidueKey], chain: str, domains: Sequence[CathDomain]
) -> np.ndarray:
    labels = np.full(len(residues), -1, dtype=np.int32)
    for domain_index, domain in enumerate(domains):
        domain_indices = set()
        for segment in domain.segments:
            if chain.casefold() not in {
                segment.start_chain.casefold(),
                segment.end_chain.casefold(),
            }:
                continue
            if segment.start <= segment.end:
                domain_indices.update(
                    index
                    for index, residue in enumerate(residues)
                    if segment.start <= residue <= segment.end
                )
                continue
            try:
                start_index = residues.index(segment.start)
                end_index = residues.index(segment.end)
            except ValueError as error:
                raise ValueError(
                    f"cannot map wrapped CATH segment {segment} in chain {chain}"
                ) from error
            if start_index > end_index:
                raise ValueError(
                    f"wrapped CATH segment is reversed in observed order: {segment}"
                )
            domain_indices.update(range(start_index, end_index + 1))
        for residue_index in sorted(domain_indices):
            if labels[residue_index] >= 0:
                raise ValueError(
                    f"overlapping CATH labels at residue {residues[residue_index]} "
                    f"in chain {chain}"
                )
            labels[residue_index] = domain_index
    return labels


def labels_to_chopping(labels: Sequence[int]) -> str:
    labels_array = np.asarray(labels, dtype=np.int32)
    domain_chunks = []
    for domain_id in sorted(set(labels_array.tolist()) - {-1}):
        indices = np.flatnonzero(labels_array == domain_id)
        if indices.size == 0:
            continue
        starts = [int(indices[0])]
        ends = []
        for left, right in zip(indices[:-1], indices[1:]):
            if right != left + 1:
                ends.append(int(left))
                starts.append(int(right))
        ends.append(int(indices[-1]))
        domain_chunks.append(
            "_".join(f"{start}-{end}" for start, end in zip(starts, ends))
        )
    return "|".join(domain_chunks)


def named_segments_to_chopping(bounds: str, names: str) -> str:
    """Rebuild discontinuous domains from Chainsaw's segment/name columns."""

    segments = [item.strip() for item in bounds.split("|") if item.strip()]
    domain_names = [item.strip() for item in names.split("|") if item.strip()]
    if len(segments) != len(domain_names):
        raise ValueError("domain boundary and domain name counts do not match")
    ordered_names: List[str] = []
    grouped: Dict[str, List[str]] = {}
    for segment, domain_name in zip(segments, domain_names):
        if domain_name not in grouped:
            ordered_names.append(domain_name)
            grouped[domain_name] = []
        grouped[domain_name].append(segment)
    return "|".join("_".join(grouped[name]) for name in ordered_names)


def chopping_to_labels(chopping: str, length: int) -> np.ndarray:
    labels = np.full(length, -1, dtype=np.int32)
    text = chopping.strip()
    if not text:
        return labels
    for domain_index, domain_text in enumerate(text.split("|")):
        for segment_text in domain_text.split("_"):
            start_text, end_text = segment_text.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start or end >= length:
                raise ValueError(
                    f"invalid chopping range {segment_text!r} for length {length}"
                )
            if np.any(labels[start : end + 1] >= 0):
                raise ValueError(f"overlapping chopping range {segment_text!r}")
            labels[start : end + 1] = domain_index
    return labels


def pairwise_domain_labels(labels: Sequence[int]) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=np.int32)
    valid = labels_array >= 0
    return (
        (labels_array[:, None] == labels_array[None, :])
        & valid[:, None]
        & valid[None, :]
    ).astype(np.float32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
