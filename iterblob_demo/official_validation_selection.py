"""Official TM-score checkpoint selection on a development validation split."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from .official_structure_metrics import run_tmscore
from .tier1_pose_geometry import apply_full_se3_action


def _run_tmscore_with_retry(
    predicted_pdb: Path,
    native_pdb: Path,
    binary: Path,
    *,
    attempts: int = 3,
):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return run_tmscore(predicted_pdb, native_pdb, binary)
        except (OSError, RuntimeError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(0.25 * attempt)
    raise RuntimeError(
        f"official TMscore failed after {attempts} attempts for {predicted_pdb}"
    ) from last_error


def _write_ca_pdb(
    path: Path,
    sequence: str,
    coordinates: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
) -> None:
    one_to_three = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
        "X": "UNK",
    }
    lines = ["REMARK 250 IterBlob official validation C-alpha trace"]
    atom_index = 1
    for residue_index, (letter, coordinate) in enumerate(
        zip(sequence, coordinates), start=1
    ):
        if not mask[residue_index - 1]:
            continue
        x, y, z = coordinate
        lines.append(
            f"ATOM  {atom_index:5d}  CA  "
            f"{one_to_three.get(letter, 'UNK'):>3s} A{residue_index:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}"
            f"{confidence[residue_index - 1]:6.2f}           C"
        )
        atom_index += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def summarize_official_records(records: Sequence[dict]) -> dict:
    delta_tm = np.asarray([row["delta_tm"] for row in records], dtype=np.float64)
    rmsd_reduction = np.asarray(
        [row["rmsd_reduction_angstrom"] for row in records], dtype=np.float64
    )
    return {
        "target_count": len(records),
        "mean_baseline_official_tm_score": float(
            np.mean([row["baseline_official_tm_score"] for row in records])
        ),
        "mean_refined_official_tm_score": float(
            np.mean([row["refined_official_tm_score"] for row in records])
        ),
        "mean_delta_official_tm_score": float(delta_tm.mean()),
        "official_tm_positive_fraction": float(np.mean(delta_tm > 0.0)),
        "mean_baseline_official_rmsd": float(
            np.mean([row["baseline_official_rmsd"] for row in records])
        ),
        "mean_refined_official_rmsd": float(
            np.mean([row["refined_official_rmsd"] for row in records])
        ),
        "mean_official_rmsd_reduction": float(rmsd_reduction.mean()),
        "official_rmsd_improved_fraction": float(np.mean(rmsd_reduction > 0.0)),
    }


class OfficialValidationEvaluator:
    """Cache official baseline scores and score one model state per epoch."""

    def __init__(
        self,
        examples: Sequence[dict],
        binary: Path,
        output_dir: Path,
        *,
        workers: int = 1,
    ):
        self.examples = examples
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = int(workers)
        self.binary = Path(binary).resolve()
        if not self.binary.is_file():
            raise FileNotFoundError(self.binary)
        self.output_dir = Path(output_dir)
        self.pdb_dir = self.output_dir / "pdb"
        self.epoch_dir = self.output_dir / "epochs"
        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
        self.baselines = self._prepare_baselines()

    def _prepare_baselines(self) -> dict[str, dict]:
        def prepare(example: dict) -> tuple[str, dict]:
            target = example["target"]
            mask = (
                np.asarray(example["residue_mask"], dtype=bool)
                & np.all(np.isfinite(example["coordinates"]), axis=1)
                & np.all(np.isfinite(example["native"]), axis=1)
            )
            if int(mask.sum()) < 3:
                raise ValueError(f"official TM-score needs at least 3 residues: {target}")
            baseline_path = self.pdb_dir / f"{target}.baseline.ca.pdb"
            native_path = self.pdb_dir / f"{target}.native.ca.pdb"
            _write_ca_pdb(
                baseline_path,
                example["sequence"],
                example["coordinates"],
                example["confidence"],
                mask,
            )
            _write_ca_pdb(
                native_path,
                example["sequence"],
                example["native"],
                np.full(len(mask), 100.0, dtype=np.float32),
                mask,
            )
            result = _run_tmscore_with_retry(
                baseline_path, native_path, self.binary
            )
            return target, {
                "mask": mask,
                "native_path": native_path,
                "tm_score": result.tm_score,
                "rmsd": result.common_residue_rmsd,
            }

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return dict(executor.map(prepare, self.examples))

    def evaluate(self, poses: Sequence[np.ndarray], epoch: int) -> tuple[dict, list[dict]]:
        if len(poses) != len(self.examples):
            raise ValueError("pose and validation-example counts differ")
        def score(item: tuple[dict, np.ndarray]) -> dict:
            example, pose = item
            target = example["target"]
            baseline = self.baselines[target]
            if example["primitive_id"] < 0:
                refined_tm = baseline["tm_score"]
                refined_rmsd = baseline["rmsd"]
            else:
                refined = apply_full_se3_action(
                    example["coordinates"],
                    example["residue_mask"],
                    example["partition_ids"],
                    primitive_id=example["primitive_id"],
                    rotation_vector_degrees=np.asarray(pose[:3], dtype=np.float32),
                    translation_vector_angstroms=np.asarray(pose[3:], dtype=np.float32),
                    rotation_pivot="hinge",
                    basis_mode=example["basis_mode"],
                )
                refined_path = self.pdb_dir / f"{target}.refined.ca.pdb"
                _write_ca_pdb(
                    refined_path,
                    example["sequence"],
                    refined,
                    example["confidence"],
                    baseline["mask"],
                )
                result = _run_tmscore_with_retry(
                    refined_path, baseline["native_path"], self.binary
                )
                refined_tm = result.tm_score
                refined_rmsd = result.common_residue_rmsd
            return {
                "target": target,
                "epoch": epoch,
                "baseline_official_tm_score": baseline["tm_score"],
                "refined_official_tm_score": refined_tm,
                "delta_tm": refined_tm - baseline["tm_score"],
                "baseline_official_rmsd": baseline["rmsd"],
                "refined_official_rmsd": refined_rmsd,
                "rmsd_reduction_angstrom": baseline["rmsd"] - refined_rmsd,
            }

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            records = list(executor.map(score, zip(self.examples, poses)))
        summary = summarize_official_records(records)
        summary["epoch"] = epoch
        csv_path = self.epoch_dir / f"epoch_{epoch:03d}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (self.epoch_dir / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary, records
