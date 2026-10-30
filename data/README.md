# Frozen dataset records

This directory identifies the exact proteins and predicted structural partitions
used in the PairRefine experiments. It contains metadata, pose-supervision labels
and predicted partition labels. It does **not** contain native/AFDB coordinate
caches, full PAE matrices, the original MPDB archive, or external model weights.
Obtain source data under the respective providers' terms before preprocessing.

| Manifest | Targets | Role |
| --- | ---: | --- |
| `manifests/train.csv` | 1,355 | MPDB training |
| `manifests/validation.csv` | 214 | MPDB validation |
| `manifests/train_validation.csv` | 1,569 | Combined MPDB records |
| `manifests/mixed_casp13.csv` | 13 | Mixed-CASP external evaluation |
| `manifests/cath52.csv` | 52 | Chainsaw CATH test subset |

The mixed-CASP cohort contains two CASP14, ten CASP15 and one CASP16 targets,
representing eight PDB entries. Historical `casp13`/`CASP13` filenames mean 13
targets, **not** CASP edition 13. Its historical `split=validation` field is
retained for compatibility; `usage_role=external_evaluation` gives its study role.
Use the supplied predicted partitions, rather than the reference-domain control
partitions from earlier experiments.

## Inspect and extract

Run from the repository root with Python 3.9 or later:

```bash
python prepare_data.py --verify --extract-partitions
```

The archive extracts 1,634 small NPZ files to `data/partitions/<cohort>/`.
Every extracted NPZ contains only `partition_ids`; personal-machine paths from
the original files have been removed. `partitions/*.csv` and `provenance.json`
record hashes, provider metadata and the CATH fallback targets `4xl1A`, `2ew9A`
and `2ly1A`, for which the frozen Chainsaw base was retained when the Merizo
proposal was unavailable.

## Bind existing processed inputs

If the original processed paired caches are available locally, provide their
directories. The tool searches recursively for `<target>.paired.npz` or
`<target>.npz`, verifies the exact cache SHA256 in the frozen manifest, and writes
machine-specific absolute-path manifests under the ignored `data/local/` directory.

```bash
python prepare_data.py --extract-partitions \
  --cache-root /path/to/mpdb_paired/examples \
  --cache-root /path/to/mixed_casp_paired/examples \
  --cache-root /path/to/cath_paired/examples
```

To bind only one evaluation set:

```bash
python prepare_data.py --extract-partitions \
  --cache-root /path/to/cath_paired/examples --cohort cath52
```

Use `data/local/train_validation.csv`, `data/local/train_partitions.csv`,
`data/local/validation_partitions.csv`, and the corresponding evaluation files
in the training/evaluation commands. Raw CIF/PDB files are not processed caches;
renaming a raw file to NPZ does not create a valid model input. The exact source
caches are not published as downloadable assets in this release.

## Acquire and preprocess source data

1. Obtain the MPDB multi-domain chain files and `Domain_boundary.txt` associated
   with [Peng et al., Bioinformatics 2022](https://doi.org/10.1093/bioinformatics/btac553).
   The source chain IDs, source hashes, sequences' hashes, UniProt accessions and
   final split membership are recorded in the MPDB manifests. Native CATH chain
   inputs likewise originated from MPDB, with reference CATH chopping recorded
   in `cath_true_chopping_evaluation_only`.
2. Obtain each frozen AFDB v6 model and PAE document from the explicit
   `afdb_model_url` and `afdb_pae_url` columns. Verify `afdb_model_sha256` and
   `afdb_pae_sha256`. Querying the live AFDB API for the latest model can produce
   a different version and is not equivalent to the recorded v6 input.
3. Generate native NPZ caches containing the schema below from the original
   MPDB chain files and domain annotations with `--build-native-manifests`.
   This verifies archived chain/sequence hashes, preserves residue ordering
   and uses the frozen final target list without sampling.
   A current PDB mmCIF downloaded through `native_structure_url` provides the
   accession's public structure, but can differ from the archived MPDB chain
   extraction; the URL is not a claim of byte-identical native preprocessing.
4. Build the AFDB-to-native paired caches with the supplied
   `prepare_tier1_paired_inputs.py`. Its inputs are a native-cohort CSV, an
   AFDB-model CSV and a PAE CSV. The required schemas and example invocation
   appear below. This performs sequence alignment, projects coordinates/PAE,
   and applies the recorded coverage criteria.
5. For mixed-CASP, use the official CASP target FASTA sequences and native PDB
   chain/entity mapping, rather than substituting the entire UniProt sequence.
   `round`, `pdb_id`, `author_chain` and `domain_boundaries` in the manifest
   identify the selected targets. The 13 target sequences and audited
   entity/domain mapping are included under `casp_source/`, with attribution to
   [CASP](https://predictioncenter.org/). Native coordinates must be obtained
   separately. The mixed-CASP preprocessing is distinct from MPDB chain pairing.

After obtaining the source files, prepare the fixed native lists and downloaded
AFDB file manifests:

```bash
python prepare_data.py --build-native-manifests \
  --mpdb-root /path/to/mpdb/chains --boundaries /path/to/Domain_boundary.txt

python prepare_data.py --stage-afdb-manifests \
  --afdb-root /path/to/afdb/models --afdb-root /path/to/afdb/pae
```

The second command verifies the frozen file hashes and produces
`data/local/<cohort>_afdb_models.csv`, `<cohort>_afdb_pae.csv` and
`<cohort>_afdb_pairs.csv`. Repeat `--afdb-root` for separate source directories.
Use `--cohort train_validation`, `--cohort cath52` or `--cohort mixed_casp13`
to stage only the required AFDB cohort. Native MPDB construction supports
`train_validation` and `cath52`; the dedicated command below handles CASP.

Native NPZ schema for the MPDB/CATH pairing tool:

| Array | Type/shape | Meaning |
| --- | --- | --- |
| `target` | scalar string | Frozen target identifier |
| `sequence` | scalar string | Ordered native-chain sequence |
| `aatype` | integer `[L]` | Index in `ACDEFGHIKLMNPQRSTVWYX` |
| `coordinates` | float32 `[L,3]` | Native C-alpha coordinates |
| `domain_ids` | integer `[L]` | Reference domain labels, `-1` when unannotated |
| `motion_ids` | integer `[L]` | Reference labels filled for missing positions |
| `residue_numbers` | int32 `[L]` | Native PDB residue numbers |
| `insertion_codes` | Unicode `[L]` | Native insertion codes |

The original native-cache functions are `read_boundary_records` and
`load_mpdb_record` in `iterblob_demo/tier1_data.py`. For CATH, use the supplied
reference chopping in native residue order. These reference labels provide
supervision/metrics; the model's primitive input comes from the separately
released predicted partitions.

The native-cohort CSV needs at least `target,split,uniprot_accession,cache_npz`,
with `cache_npz` pointing to the generated **native** NPZ. Retain other frozen
metadata columns when available. The AFDB-model CSV uses
`accession,model_cif,model_sha256`; the PAE CSV uses
`accession,pae_json,pae_sha256`. Their paths point to the user's downloaded files.

```bash
python prepare_tier1_paired_inputs.py \
  --cohort-manifest data/local/train_validation_native.csv \
  --afdb-manifest data/local/train_validation_afdb_models.csv \
  --pae-manifest data/local/train_validation_afdb_pae.csv \
  --output-dir /path/to/mpdb_paired \
  --limit-train 1355 --limit-validation 214 --limit-test 0 \
  --min-identity 0.95 --min-query-coverage 0.80 \
  --min-labeled-coverage 0.80 --min-mapped-residues 60
```

For CATH52, use the three `cath52` local CSV files,
`--limit-train 0 --limit-validation 0 --limit-test 52` and
`--min-identity 0.90`. For mixed-CASP, place the separately obtained native
mmCIF files under `data/casp_source/structure/<lowercase-pdb-id>.cif`, verify
their `native_sha256` values against `manifests/mixed_casp13.csv`, and run:

```bash
python prepare_casp_paired_external_test.py \
  --casp-root data/casp_source \
  --audit data/casp_source/casp_afdb_coverage_audit.csv \
  --afdb-manifest data/local/mixed_casp13_afdb_pairs.csv \
  --output-dir /path/to/mixed_casp_paired
```

This also generates an early reference-partition control. Use the supplied
frozen predicted partitions in `data/local/mixed_casp13_partitions.csv` for
the paper model evaluation.

Container-level NPZ byte hashes can differ when caches
are rebuilt using a different NumPy version or additional mask arrays. The
existing-cache binding command deliberately requires the archived byte hashes;
for newly preprocessed inputs use the resulting `tier1_paired_manifest.csv`
directly and validate residue mappings and source hashes before interpreting
any differences in scores. This release does not claim that original MPDB
coordinates can be reconstructed exactly from current PDB downloads alone.

## Labels and metrics

`labels/train.csv` and `labels/validation.csv` contain the native-supervised
bounded SE(3) teacher poses for the primitive selected by predicted PAE.
These records include internal fixed-correspondence TM-like diagnostics used
for supervision; they are not the official TM-score benchmark results.
`baseline_tm_score` and related preprocessing columns in the input manifests
are likewise internal diagnostics. Official reported TM-score and common-residue
RMSD must come from the Zhang-lab TMscore evaluation pipeline and its result files.

`provenance.json` records cohort construction, the frozen providers, sequence
filter thresholds, source-manifest hashes and per-partition hashes.
`checksums.json` verifies the distributed metadata and partition archive.
See [TERMS.md](TERMS.md) for source attribution and external data terms.
