"""Local MPDB inputs and synthetic rigid-domain perturbations for Tier 1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INDEX = {letter: index for index, letter in enumerate(AMINO_ACIDS)}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K",
}


@dataclass(frozen=True, order=True)
class ResiduePosition:
    number: int
    insertion_code: str = ""

    def __post_init__(self) -> None:
        code = self.insertion_code.upper()
        if code and (len(code) != 1 or not code.isalpha()):
            raise ValueError(f"invalid PDB insertion code {self.insertion_code!r}")
        object.__setattr__(self, "insertion_code", code)


@dataclass(frozen=True)
class BoundaryRecord:
    target: str
    definition: str
    domain_ranges: Tuple[
        Tuple[Tuple[ResiduePosition, ResiduePosition], ...], ...
    ]

    @property
    def domain_count(self) -> int:
        return len(self.domain_ranges)


@dataclass(frozen=True)
class ProteinRecord:
    target: str
    sequence: str
    coordinates: np.ndarray
    residue_numbers: np.ndarray
    insertion_codes: np.ndarray
    domain_ids: np.ndarray
    motion_ids: np.ndarray
    uniprot_accession: Optional[str]


_BOUNDARY_LINE = re.compile(r"^(\S+)\s+(\S+)\s+(\d+)\s+(.+?)\s*$")


def parse_domain_ranges(text: str, expected_count: Optional[int] = None):
    domains = []
    for domain_text in text.strip().strip(";").split(";"):
        ranges = []
        for range_text in domain_text.split(","):
            match = re.fullmatch(
                r"\s*(-?\d+)([A-Za-z]?)\s*-\s*(-?\d+)([A-Za-z]?)\s*",
                range_text,
            )
            if match is None:
                raise ValueError(f"invalid domain range {range_text!r}")
            start = ResiduePosition(int(match.group(1)), match.group(2))
            end = ResiduePosition(int(match.group(3)), match.group(4))
            if start > end:
                raise ValueError(f"reversed domain range {range_text!r}")
            ranges.append((start, end))
        if ranges:
            domains.append(tuple(ranges))
    if expected_count is not None and len(domains) != expected_count:
        raise ValueError(
            f"expected {expected_count} domains, parsed {len(domains)} from {text!r}"
        )
    return tuple(domains)


def read_boundary_records(
    path: Path,
    *,
    strict: bool = True,
    errors: Optional[List[str]] = None,
) -> Dict[str, BoundaryRecord]:
    records: Dict[str, BoundaryRecord] = {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            match = _BOUNDARY_LINE.match(line)
            try:
                if match is None:
                    raise ValueError(f"invalid boundary line: {line.rstrip()}")
                target, definition, count_text, ranges_text = match.groups()
                record = BoundaryRecord(
                    target=target,
                    definition=definition,
                    domain_ranges=parse_domain_ranges(ranges_text, int(count_text)),
                )
            except ValueError as exc:
                message = f"line {line_number}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                if errors is not None:
                    errors.append(message)
                continue
            records[target.lower()] = record
    return records


def _dbref_uniprot(line: str) -> Optional[str]:
    if not line.startswith("DBREF") or len(line) < 42:
        return None
    database = line[26:32].strip()
    accession = line[33:41].strip()
    return accession if database == "UNP" and accession else None


def parse_pdb_ca(
    path: Path, target: str
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    chain = target[-1]
    atoms: Dict[Tuple[int, str], Tuple[str, np.ndarray]] = {}
    uniprot = None
    with Path(path).open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            if line.startswith("DBREF") and line[12:13].strip() == chain:
                uniprot = _dbref_uniprot(line) or uniprot
            if not line.startswith("ATOM  ") or len(line) < 54:
                continue
            if line[21:22] != chain or line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in {" ", "A"}:
                continue
            try:
                residue_number = int(line[22:26])
                coordinate = np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float32,
                )
            except ValueError:
                continue
            key = (residue_number, line[26:27].strip())
            if key in atoms and altloc == "A":
                continue
            atoms[key] = (THREE_TO_ONE.get(line[17:20].strip(), "X"), coordinate)
    if not atoms:
        raise ValueError(f"no C-alpha atoms found for {target} in {path}")
    ordered = sorted(atoms.items(), key=lambda item: (item[0][0], item[0][1]))
    sequence = "".join(value[0] for _, value in ordered)
    coordinates = np.stack([value[1] for _, value in ordered]).astype(np.float32)
    residue_numbers = np.asarray([key[0] for key, _ in ordered], dtype=np.int32)
    insertion_codes = np.asarray([key[1].upper() for key, _ in ordered], dtype="U1")
    return sequence, coordinates, residue_numbers, insertion_codes, uniprot


def _position_keys(
    residue_numbers: np.ndarray, insertion_codes: Optional[np.ndarray] = None
) -> np.ndarray:
    numbers = np.asarray(residue_numbers, dtype=np.int64)
    if insertion_codes is None:
        ranks = np.zeros(numbers.shape, dtype=np.int64)
    else:
        codes = np.char.upper(np.asarray(insertion_codes, dtype="U1"))
        ranks = np.zeros(numbers.shape, dtype=np.int64)
        nonblank = codes != ""
        ranks[nonblank] = np.asarray(
            [ord(code) - ord("A") + 1 for code in codes[nonblank]], dtype=np.int64
        )
    return numbers * 32 + ranks


def _position_key(position: ResiduePosition) -> int:
    rank = 0 if not position.insertion_code else ord(position.insertion_code) - ord("A") + 1
    return position.number * 32 + rank


def assign_domains(
    residue_numbers: np.ndarray,
    domain_ranges: Sequence[
        Sequence[Tuple[ResiduePosition, ResiduePosition]]
    ],
    insertion_codes: Optional[np.ndarray] = None,
) -> np.ndarray:
    labels = np.full(len(residue_numbers), -1, dtype=np.int32)
    residue_keys = _position_keys(residue_numbers, insertion_codes)
    for domain_id, ranges in enumerate(domain_ranges):
        for start, end in ranges:
            selected = (residue_keys >= _position_key(start)) & (
                residue_keys <= _position_key(end)
            )
            collision = selected & (labels >= 0) & (labels != domain_id)
            if np.any(collision):
                raise ValueError("domain ranges overlap on observed C-alpha residues")
            labels[selected] = domain_id
    return labels


def fill_motion_labels(domain_ids: np.ndarray) -> np.ndarray:
    labels = np.asarray(domain_ids, dtype=np.int32)
    known = np.flatnonzero(labels >= 0)
    if not len(known):
        raise ValueError("at least one labeled domain residue is required")
    result = labels.copy()
    missing = np.flatnonzero(result < 0)
    for index in missing:
        nearest = known[np.argmin(np.abs(known - index))]
        result[index] = labels[nearest]
    return result


def load_mpdb_record(path: Path, boundary: BoundaryRecord) -> ProteinRecord:
    sequence, coordinates, residue_numbers, insertion_codes, uniprot = parse_pdb_ca(
        path, boundary.target
    )
    domain_ids = assign_domains(
        residue_numbers, boundary.domain_ranges, insertion_codes
    )
    return ProteinRecord(
        target=boundary.target,
        sequence=sequence,
        coordinates=coordinates,
        residue_numbers=residue_numbers,
        insertion_codes=insertion_codes,
        domain_ids=domain_ids,
        motion_ids=fill_motion_labels(domain_ids),
        uniprot_accession=uniprot,
    )


def encode_sequence(sequence: str) -> np.ndarray:
    unknown = AA_TO_INDEX["X"]
    return np.asarray([AA_TO_INDEX.get(letter, unknown) for letter in sequence], dtype=np.int64)


def axis_angle_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*axis.shape[:-1], 3, 3)
    identity = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(skew)
    outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    cosine = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
    sine = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    return cosine * identity + sine * skew + (1.0 - cosine) * outer


def perturb_domains(
    coordinates: torch.Tensor,
    motion_ids: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    max_rotation_degrees: float = 30.0,
    max_translation: float = 6.0,
    coordinate_noise: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply independent rigid motions while preserving each domain internally."""

    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [B, N, 3]")
    if motion_ids.shape != coordinates.shape[:-1] or residue_mask.shape != motion_ids.shape:
        raise ValueError("motion_ids and residue_mask must match coordinates")
    output = coordinates.clone()
    max_angle = math.radians(float(max_rotation_degrees))
    for batch_index in range(coordinates.shape[0]):
        valid = residue_mask[batch_index].bool()
        for domain_id in torch.unique(motion_ids[batch_index][valid]).tolist():
            selected = valid & motion_ids[batch_index].eq(int(domain_id))
            if not torch.any(selected):
                continue
            points = coordinates[batch_index, selected]
            center = points.mean(dim=0)
            axis = torch.randn(
                3, device=coordinates.device, dtype=coordinates.dtype, generator=generator
            )
            angle = (
                torch.rand(
                    (), device=coordinates.device, dtype=coordinates.dtype, generator=generator
                )
                * 2.0
                - 1.0
            ) * max_angle
            rotation = axis_angle_rotation(axis, angle)
            translation = (
                torch.rand(
                    3, device=coordinates.device, dtype=coordinates.dtype, generator=generator
                )
                * 2.0
                - 1.0
            ) * float(max_translation)
            output[batch_index, selected] = (points - center) @ rotation.transpose(-1, -2) + center + translation
    if coordinate_noise > 0:
        noise = torch.randn(
            output.shape,
            device=output.device,
            dtype=output.dtype,
            generator=generator,
        ) * float(coordinate_noise)
        output = output + noise * residue_mask.unsqueeze(-1).to(output.dtype)
    return torch.where(residue_mask.unsqueeze(-1), output, coordinates)
