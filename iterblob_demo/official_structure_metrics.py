"""Wrappers for the official TM-score and US-align executables.

The in-process metrics in :mod:`structure_metrics` are useful diagnostics, but
paper comparisons must use the original scorer.  This module deliberately
keeps the raw stdout together with parsed values so every reported number can
be audited later.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def _required(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"TM-score output is missing {label}")
    return match


@dataclass(frozen=True)
class TMScoreResult:
    """Metrics emitted by Zhang lab's ``TMscore`` executable.

    ``structure2_length`` is the normalization length.  The public wrapper
    therefore always passes the native/reference structure as Structure2.
    """

    structure1: str
    structure1_length: int
    structure2: str
    structure2_length: int
    common_residues: int
    common_residue_rmsd: float
    tm_score: float
    d0: float
    maxsub_score: float
    gdt_ts_score: float
    gdt_ha_score: float
    gdt_ts_d1: float
    gdt_ts_d2: float
    gdt_ts_d4: float
    gdt_ts_d8: float
    gdt_ha_d05: float
    gdt_ha_d1: float
    gdt_ha_d2: float
    gdt_ha_d4: float
    raw_stdout: str

    def to_dict(self, include_raw: bool = True) -> Dict[str, object]:
        values = asdict(self)
        if not include_raw:
            values.pop("raw_stdout")
        return values


def parse_tmscore_output(stdout: str) -> TMScoreResult:
    """Parse the stable text format produced by the official TMscore code."""

    if not stdout.strip():
        raise ValueError("TM-score output is empty")

    structure1 = _required(
        r"^Structure1:\s*(.*?)\s+Length=\s*(\d+)\s*$",
        stdout,
        "Structure1",
    )
    structure2 = _required(
        r"^Structure2:\s*(.*?)\s+Length=\s*(\d+)",
        stdout,
        "Structure2",
    )
    common = _required(
        r"^Number of residues in common=\s*(\d+)\s*$",
        stdout,
        "common-residue count",
    )
    common_rmsd = _required(
        rf"^RMSD of\s+the common residues=\s*({_FLOAT})\s*$",
        stdout,
        "common-residue RMSD",
    )
    tm_score = _required(
        rf"^TM-score\s*=\s*({_FLOAT})\s+\(d0=\s*({_FLOAT})\)",
        stdout,
        "TM-score",
    )
    maxsub = _required(
        rf"^MaxSub-score=\s*({_FLOAT})",
        stdout,
        "MaxSub-score",
    )
    gdt_ts = _required(
        rf"^GDT-TS-score=\s*({_FLOAT})\s+%\(d<1\)=({_FLOAT})\s+"
        rf"%\(d<2\)=({_FLOAT})\s+%\(d<4\)=({_FLOAT})\s+%\(d<8\)=({_FLOAT})",
        stdout,
        "GDT-TS-score",
    )
    gdt_ha = _required(
        rf"^GDT-HA-score=\s*({_FLOAT})\s+%\(d<0\.5\)=({_FLOAT})\s+"
        rf"%\(d<1\)=({_FLOAT})\s+%\(d<2\)=({_FLOAT})\s+%\(d<4\)=({_FLOAT})",
        stdout,
        "GDT-HA-score",
    )

    return TMScoreResult(
        structure1=structure1.group(1).strip(),
        structure1_length=int(structure1.group(2)),
        structure2=structure2.group(1).strip(),
        structure2_length=int(structure2.group(2)),
        common_residues=int(common.group(1)),
        common_residue_rmsd=float(common_rmsd.group(1)),
        tm_score=float(tm_score.group(1)),
        d0=float(tm_score.group(2)),
        maxsub_score=float(maxsub.group(1)),
        gdt_ts_score=float(gdt_ts.group(1)),
        gdt_ha_score=float(gdt_ha.group(1)),
        gdt_ts_d1=float(gdt_ts.group(2)),
        gdt_ts_d2=float(gdt_ts.group(3)),
        gdt_ts_d4=float(gdt_ts.group(4)),
        gdt_ts_d8=float(gdt_ts.group(5)),
        gdt_ha_d05=float(gdt_ha.group(2)),
        gdt_ha_d1=float(gdt_ha.group(3)),
        gdt_ha_d2=float(gdt_ha.group(4)),
        gdt_ha_d4=float(gdt_ha.group(5)),
        raw_stdout=stdout,
    )


def run_tmscore(
    predicted_pdb: Path,
    native_pdb: Path,
    binary: Path,
    timeout_seconds: Optional[float] = 300.0,
) -> TMScoreResult:
    """Score ``predicted_pdb`` against ``native_pdb`` with native normalization."""

    predicted_pdb = Path(predicted_pdb).resolve()
    native_pdb = Path(native_pdb).resolve()
    binary = Path(binary).resolve()
    for label, path in (
        ("predicted PDB", predicted_pdb),
        ("native PDB", native_pdb),
        ("TMscore binary", binary),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    completed = subprocess.run(
        [str(binary), str(predicted_pdb), str(native_pdb)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "TMscore failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    return parse_tmscore_output(completed.stdout)

