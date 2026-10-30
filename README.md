# PairRefine

Official implementation of **PairRefine: A Residue-Pair Guided Primitive-Level Refinement Model for Protein Structure Prediction**.

PairRefine refines an existing predicted structure using frozen predicted primitives and a residue-pair latent-attention pose head. The head is trained from scratch. Prediction uses the supplied coordinates, pLDDT, PAE, and predicted partitions; the prediction entry point does not read native coordinates or supervision labels.

The original implementation uses the internal name `IterBlob` and the head identifier `ours_pair_attention`. These names remain in source and result schemas for checkpoint compatibility.

## Included artifacts

- Model, local-frame geometry, training, inference, and official evaluation code.
- The paper's epoch 8 checkpoint, exported to a portable PyTorch format.
- Frozen training/validation and evaluation target lists, predicted partitions, and supervision labels.
- Per-target official results and raw scorer outputs for mixed-CASP 13 and CATH52.
- Data provenance, SHA256 checksums, input specifications, and protocol tests.

Original coordinate caches and third-party executables are obtained separately. See [data preparation](data/README.md) for sources and recovery instructions.

## Environment

Tested with Python 3.9.23, NumPy 2.0.2, and PyTorch 2.8.0. CPU inference is supported. The paper training used an NVIDIA GeForce RTX 4060 Laptop GPU.

```bash
git clone https://github.com/eclipsy06/PairRefine.git
cd PairRefine
python -m venv .venv
# Activate .venv using your operating system's activation command.
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch 2.8.0 build appropriate for the CUDA environment using the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/). For raw mmCIF/PDB preprocessing, also install:

```bash
python -m pip install -r requirements-data.txt
```

Install the original Zhang-lab [TM-score program](https://zhanggroup.org/TM-score/) separately. Evaluation requires an explicit executable path. The archived results used a binary with SHA256 `569244dedf463fcf7164933c1050b06bf63de378ca8940654665e486098fcffa`.

## Prepare data

The frozen lists contain 1,355 training targets, 214 validation targets, 13 mixed-CASP targets, and 52 CATH targets. The `casp13` result folder means 13 targets from a mixture of CASP14, CASP15, and CASP16. It does not denote CASP edition 13.

```bash
python prepare_data.py --extract-partitions
```

Follow [data/README.md](data/README.md) to obtain and prepare the original inputs. If matching paired NPZ caches are already available, bind them by target and verify their hashes:

```bash
python prepare_data.py --cache-root /path/to/paired/caches
```

Multiple `--cache-root` arguments can be supplied. This creates machine-local manifests under `data/local/`; use those manifests in the commands below. Input files and extracted partitions remain local. [The input schema](docs/input_format.md) also describes inference on user-supplied structures.

## Predict and evaluate

```bash
python predict.py \
  --manifest data/local/mixed_casp13.csv \
  --partition-manifest data/local/mixed_casp13_partitions.csv \
  --checkpoint checkpoints/pairrefine_epoch008.pt \
  --output-dir outputs/mixed_casp13 --device cpu --write-pdb

python evaluate.py \
  --saved-results outputs/mixed_casp13/predicted_actions.csv \
  --manifest data/local/mixed_casp13.csv \
  --partition-manifest data/local/mixed_casp13_partitions.csv \
  --binary /path/to/TMscore \
  --output-dir outputs/mixed_casp13/official --dataset-name mixed_CASP13
```

For CATH52, replace `mixed_casp13` with `cath52`. Run multi-line commands in a compatible shell, or join the lines on Windows PowerShell. Inference exports a six-component rigid pose and can export refined CA PDB traces. Full-atom reconstruction is outside this release.

Evaluation uses the original executable's TM-score and **RMSD of the common residues**. The sequence-mapped reference CA trace is always Structure2 and determines TM-score normalization. Higher TM-score and lower RMSD indicate improvement. Internal training diagnostics are separately named and are not substitutes for these official metrics.

## Paper checkpoint and results

| Setting | Value |
| --- | --- |
| Model | `ours_pair_attention`, from scratch |
| Hidden dimension / latent queries / attention heads | 96 / 8 / 4 |
| Trainable parameters | 68,550 |
| Optimizer | AdamW, weight decay 0.0001 |
| Learning rate / batch size | 0.0002 / 32 |
| Random seed / training epochs | 20260820 / 40 |
| Paper artifact | Epoch 8 |
| Primitive selection / local frame | PAE deployment / `equivariant_v2` |
| Pose scale / confidence gating | 1.0 / disabled |

The paper artifact is the epoch 8 checkpoint from the recorded 40-epoch run. It was selected retrospectively after external results were inspected; the provenance records this selection history. A new run's validation-selected `best_model.pt` is a separate artifact.

| Dataset | Baseline TM-score | PairRefine TM-score | Baseline RMSD (A) | PairRefine RMSD (A) |
| --- | ---: | ---: | ---: | ---: |
| Mixed-CASP 13 | 0.539577 | 0.543862 | 36.063077 | 36.006692 |
| CATH52 | 0.881046 | 0.881513 | 3.925000 | 3.895115 |

All target-level results, including worsened and unchanged targets, are retained in [results](results). The result files reproduce the original scorer output precision.

The portable file `checkpoints/pairrefine_epoch008.pt` contains exactly the same 17 tensors as the original paper checkpoint. Its SHA256 is `5896137fe2880cd4c00e2fdaccc02b61bf86951393720aaea1799c66da503470`. Serialization was changed to remove machine-specific paths; original and portable hashes and verification details are in [checkpoint provenance](checkpoints/provenance.json).

## Train

After preparing the fixed inputs and installing TMscore:

```bash
python train_tier1_residue_pair_attention.py \
  --train-manifest data/local/train_validation.csv \
  --validation-manifest data/local/train_validation.csv \
  --train-partition-manifest data/local/train_partitions.csv \
  --validation-partition-manifest data/local/validation_partitions.csv \
  --train-labels data/labels/train.csv \
  --validation-labels data/labels/validation.csv \
  --head-type ours_pair_attention --primitive-selection pae_deployment \
  --basis-mode equivariant_v2 --device cuda:0 \
  --learning-rate 0.0002 --batch-size 32 --epochs 40 \
  --hidden-dim 96 --latent-count 8 --attention-heads 4 \
  --weight-decay 0.0001 --endpoint-scale-angstroms 4 \
  --seed 20260820 --deterministic --save-epochs 8 \
  --official-tmscore-binary /path/to/TMscore \
  --output-dir outputs/training
```

Explicit arguments override historical CLI defaults. `--save-epochs 8` retains that epoch independently of validation ranking. Different hardware or library builds can affect floating-point results; use the supplied checkpoint and recorded input/scorer hashes when checking the archived paper results.

## Checks

```bash
python -B -m unittest discover -s tests -v
```

The tests cover native-independent prediction, fixed PAE selection, identity fallback, and parsing and normalization of official scores. [Release verification](docs/release_verification.json) records the comparison against all 65 archived evaluation targets.

## License and upstream tools

The project code is covered by the repository's [MIT license](LICENSE). External databases, structures, partitioning tools, and TMscore retain their own terms. See [third-party notices](THIRD_PARTY.md) and [data terms](data/TERMS.md). Chainsaw and Merizo implementations and weights are installed from their upstream projects and are not bundled here.
