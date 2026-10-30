"""Extract frozen partitions and bind user-provided paired input caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COHORTS = ("train_validation", "mixed_casp13", "cath52")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_release_metadata() -> None:
    checksums = json.loads((DATA / "checksums.json").read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(DATA.resolve()):
            raise ValueError(f"Invalid metadata path: {relative}")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or modified release metadata: {relative}")


def extract_partitions() -> None:
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    expected = {entry["path"]: entry["sha256"] for entry in provenance["partitions"]}
    with zipfile.ZipFile(DATA / "frozen_partitions.zip") as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError("Frozen partition archive contents do not match provenance")
        for relative, expected_hash in expected.items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to((DATA / "partitions").resolve()):
                raise ValueError(f"Invalid archive member: {relative}")
            payload = archive.read(relative)
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ValueError(f"Partition hash mismatch: {relative}")
            if path.exists() and sha256(path) != expected_hash:
                raise ValueError(f"Refusing to replace modified partition: {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    print(f"Extracted and verified {len(expected)} predicted partitions.")


def cache_index(roots: list[Path]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Cache root is not a directory: {root}")
        for path in root.rglob("*.npz"):
            target = path.stem.removesuffix(".paired").casefold()
            result.setdefault(target, []).append(path)
    return result


def native_cath_record(path: Path, target: str) -> dict:
    import numpy as np
    from iterblob_demo.tier1_data import THREE_TO_ONE
    from iterblob_demo.chainsaw_protocol import STANDARD_AA

    chain = " " if target[4] == "0" else target[4]
    atoms = []
    seen = set()
    # Preserve the extraction order used by the original CATH preprocessing.
    with path.open(encoding="ascii", errors="replace") as stream:
        for line in stream:
            if not line.startswith("ATOM") or len(line) < 54:
                continue
            name = line[17:20].strip()
            if line[21:22].casefold() != chain.casefold() or line[12:16].strip() != "CA" or name not in STANDARD_AA:
                continue
            try:
                key = (int(line[22:26]), line[26:27].strip().upper())
                coordinate = np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
            except ValueError:
                continue
            if key not in seen:
                seen.add(key)
                atoms.append((key, THREE_TO_ONE[name], coordinate))
    if not atoms:
        raise ValueError(f"No standard C-alpha atoms: {target}")
    return {
        "sequence": "".join(atom[1] for atom in atoms),
        "coordinates": np.stack([atom[2] for atom in atoms]),
        "residue_numbers": np.asarray([atom[0][0] for atom in atoms], dtype=np.int32),
        "insertion_codes": np.asarray([atom[0][1] for atom in atoms], dtype="U1"),
    }


def build_native_manifests(mpdb_root: Path, boundaries: Path | None, cohorts: list[str], output: Path) -> None:
    import numpy as np
    from iterblob_demo.chainsaw_protocol import chopping_to_labels
    from iterblob_demo.tier1_data import encode_sequence, fill_motion_labels, load_mpdb_record, read_boundary_records

    if "mixed_casp13" in cohorts:
        raise ValueError("Use prepare_casp_paired_external_test.py for mixed-CASP native mapping")
    if "train_validation" in cohorts and boundaries is None:
        raise ValueError("--boundaries is required for MPDB training/validation native caches")
    annotations = read_boundary_records(boundaries, strict=False) if boundaries else {}
    for cohort in cohorts:
        rows = read_csv(DATA / f"manifests/{cohort}.csv")
        cache_dir = output / f"{cohort}_native/examples"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source = mpdb_root / f"{row['target']}.pdb"
            if not source.is_file() or sha256(source) != row["native_source_file_sha256"]:
                raise ValueError(f"Missing or different original MPDB chain: {row['target']}")
            if cohort == "train_validation":
                record = load_mpdb_record(source, annotations[row["target"].casefold()])
                sequence = record.sequence
                payload = {"coordinates": record.coordinates.astype(np.float32), "residue_numbers": record.residue_numbers.astype(np.int32), "insertion_codes": record.insertion_codes.astype("U1"), "domain_ids": record.domain_ids.astype(np.int32), "motion_ids": record.motion_ids.astype(np.int32)}
            else:
                payload = native_cath_record(source, row["target"])
                sequence = payload.pop("sequence")
                payload["domain_ids"] = chopping_to_labels(row["cath_true_chopping_evaluation_only"], len(sequence))
                payload["motion_ids"] = fill_motion_labels(payload["domain_ids"])
            if hashlib.sha256(sequence.encode("ascii")).hexdigest() != row["sequence_sha256"]:
                raise ValueError(f"Native sequence differs from frozen record: {row['target']}")
            cache_path = cache_dir / f"{row['target']}.native.npz"
            np.savez_compressed(cache_path, target=np.asarray(row["target"]), sequence=np.asarray(sequence), aatype=encode_sequence(sequence), **payload)
            row["cache_npz"] = str(cache_path.resolve())
        write_csv(output / f"{cohort}_native.csv", rows)
        print(f"Built {len(rows)} native caches for the frozen {cohort} list.")


def stage_afdb_manifests(roots: list[Path], cohorts: list[str], output: Path) -> None:
    by_name: dict[str, list[Path]] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"AFDB source directory does not exist: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".cif", ".json"}:
                by_name.setdefault(path.name, []).append(path)
    verified = {}
    for cohort in cohorts:
        rows = read_csv(DATA / f"manifests/{cohort}.csv")
        models, paes, paired = [], [], []
        for accession, row in {row["uniprot_accession"]: row for row in rows}.items():
            found = {}
            for kind, suffix, hash_key in (("model", "cif", "afdb_model_sha256"), ("pae", "json", "afdb_pae_sha256")):
                from urllib.parse import urlparse
                filename = Path(urlparse(row[f"afdb_{kind}_url"]).path).name
                alternate = f"AF-{accession}-F1.cif" if kind == "model" else f"AF-{accession}-F1-pae.json"
                matches = by_name.get(filename, []) + by_name.get(alternate, [])
                for path in matches:
                    if path not in verified:
                        verified[path] = sha256(path)
                    if verified[path] == row[hash_key]:
                        found[kind] = path
                        break
                if kind not in found:
                    raise ValueError(f"Missing or different frozen AFDB v6 {kind}: {accession}; acquire {row[f'afdb_{kind}_url']}")
            models.append({"accession": accession, "model_cif": str(found["model"]), "model_sha256": row["afdb_model_sha256"]})
            paes.append({"accession": accession, "pae_json": str(found["pae"]), "pae_sha256": row["afdb_pae_sha256"]})
            paired.append({"accession": accession, "latest_version": row["afdb_version"], "model_cif": str(found["model"]), "model_sha256": row["afdb_model_sha256"], "pae_json": str(found["pae"]), "pae_sha256": row["afdb_pae_sha256"]})
        write_csv(output / f"{cohort}_afdb_models.csv", models)
        write_csv(output / f"{cohort}_afdb_pae.csv", paes)
        write_csv(output / f"{cohort}_afdb_pairs.csv", paired)
        print(f"Verified frozen AFDB model/PAE sources for {len(models)} {cohort} accessions.")


def bind_caches(roots: list[Path], cohorts: list[str], output: Path) -> None:
    candidates = cache_index(roots)
    hash_cache: dict[Path, str] = {}
    all_rows: dict[str, list[dict[str, str]]] = {}
    missing = []
    for cohort in cohorts:
        rows = read_csv(DATA / f"manifests/{cohort}.csv")
        for row in rows:
            matches = []
            for path in candidates.get(row["target"].casefold(), []):
                if path not in hash_cache:
                    hash_cache[path] = sha256(path)
                if hash_cache[path] == row["input_cache_sha256"]:
                    matches.append(path)
            if not matches:
                missing.append(f"{cohort}/{row['target']}")
            else:
                row["cache_npz"] = str(sorted(matches)[0])
        all_rows[cohort] = rows
    if missing:
        preview = ", ".join(missing[:20])
        raise ValueError(
            f"{len(missing)} paired caches are missing or have different SHA256 values: {preview}. "
            "Provide the processed paired caches described in data/README.md; raw CIF/PDB files are not paired caches."
        )
    output.mkdir(parents=True, exist_ok=True)
    for cohort, rows in all_rows.items():
        write_csv(output / f"{cohort}.csv", rows)
        if cohort == "train_validation":
            for split in ("train", "validation"):
                write_csv(output / f"{split}.csv", [row for row in rows if row["split"] == split])
    partition_sets = set()
    if "train_validation" in cohorts:
        partition_sets.update(("train", "validation"))
    partition_sets.update(name for name in cohorts if name != "train_validation")
    for name in sorted(partition_sets):
        rows = read_csv(DATA / f"partitions/{name}.csv")
        for row in rows:
            path = (ROOT / row["partition_npz"]).resolve()
            if not path.is_file() or sha256(path) != row["partition_sha256"]:
                raise ValueError(f"Missing or modified partition: {row['partition_npz']}; use --extract-partitions")
            row["partition_npz"] = str(path)
        write_csv(output / f"{name}_partitions.csv", rows)
    count = sum(len(rows) for rows in all_rows.values())
    print(f"Verified {count} exact paired caches; local manifests written to {output}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check distributed manifests, labels, provenance and archive")
    parser.add_argument("--extract-partitions", action="store_true", help="Extract all 1,634 predicted partitions")
    parser.add_argument("--cache-root", action="append", type=Path, default=[], help="Existing paired-cache directory; repeat for multiple cohorts")
    parser.add_argument("--build-native-manifests", action="store_true", help="Build native caches from the frozen MPDB/CATH lists")
    parser.add_argument("--mpdb-root", type=Path, help="Original MPDB chain PDB directory")
    parser.add_argument("--boundaries", type=Path, help="Original MPDB Domain_boundary.txt")
    parser.add_argument("--stage-afdb-manifests", action="store_true", help="Locate and verify downloaded frozen AFDB v6 sources")
    parser.add_argument("--afdb-root", action="append", type=Path, default=[], help="Directory of downloaded AFDB CIF/PAE files; repeat as needed")
    parser.add_argument("--cohort", action="append", choices=COHORTS, help="Bind selected cohorts; default: all")
    parser.add_argument("--output-dir", type=Path, default=DATA / "local", help="Machine-specific generated manifest directory")
    args = parser.parse_args()
    verify_release_metadata()
    print("Release metadata verified.")
    if args.extract_partitions:
        extract_partitions()
    if args.cache_root:
        bind_caches(args.cache_root, args.cohort or list(COHORTS), args.output_dir.resolve())
    if args.build_native_manifests:
        if args.mpdb_root is None:
            parser.error("--build-native-manifests requires --mpdb-root")
        build_native_manifests(args.mpdb_root.expanduser().resolve(), args.boundaries, args.cohort or ["train_validation", "cath52"], args.output_dir.resolve())
    if args.stage_afdb_manifests:
        if not args.afdb_root:
            parser.error("--stage-afdb-manifests requires --afdb-root")
        stage_afdb_manifests(args.afdb_root, args.cohort or list(COHORTS), args.output_dir.resolve())
    if args.cohort and not (args.cache_root or args.build_native_manifests or args.stage_afdb_manifests):
        parser.error("--cohort requires an input preparation action")


if __name__ == "__main__":
    main()
