# PairRefine

Model implementation of **PairRefine: A Residue-Pair Guided Primitive-Level Refinement Model for Protein Structure Prediction**. (Under Review by Scientific Reports)

PairRefine refines predicted C-alpha coordinates with a residue-pair attention pose head. This repository contains the model, training and inference entry points, official evaluation wrapper, and paper checkpoint. The implementation retains the internal package name `iterblob_demo` and head identifier `ours_pair_attention` for checkpoint compatibility.

## Install

Tested with Python 3.9.23, NumPy 2.0.2, and PyTorch 2.8.0.

```bash
python -m pip install -r requirements.txt
```

For GPU training, install the appropriate PyTorch 2.8.0 CUDA build using the [PyTorch instructions](https://pytorch.org/get-started/previous-versions/). Training validation and evaluation also require the original [Zhang-lab TM-score executable](https://zhanggroup.org/TM-score/), installed separately.

## Predict

Prepare your input NPZ files and CSV manifests using the [input format](#input-format).

```bash
python predict.py \
  --manifest /absolute/path/inputs.csv \
  --partition-manifest /absolute/path/partitions.csv \
  --checkpoint checkpoints/pairrefine_epoch008.pt \
  --output-dir outputs/prediction --device cpu --write-pdb
```

Prediction uses predicted coordinates, pLDDT, PAE, and predicted partitions. It exports `predicted_actions.csv` and, with `--write-pdb`, refined C-alpha PDB traces. Native coordinates and supervision labels are used only for evaluation or training.

## Evaluate

The input cache must additionally contain reference coordinates on the same residue mapping.

```bash
python evaluate.py \
  --saved-results outputs/prediction/predicted_actions.csv \
  --manifest /absolute/path/inputs.csv \
  --partition-manifest /absolute/path/partitions.csv \
  --binary /absolute/path/TMscore \
  --output-dir outputs/official --dataset-name user_cohort
```

Evaluation reports the official executable's TM-score and RMSD of common residues. The reference trace is Structure2 and defines TM-score normalization. Internal training diagnostics are named separately.

## Train

Supply training and validation manifests, predicted partitions, and supervision labels in the documented format. The paper configuration is:

```bash
python train_tier1_residue_pair_attention.py \
  --train-manifest /absolute/path/train.csv \
  --validation-manifest /absolute/path/validation.csv \
  --train-partition-manifest /absolute/path/train_partitions.csv \
  --validation-partition-manifest /absolute/path/validation_partitions.csv \
  --train-labels /absolute/path/train_labels.csv \
  --validation-labels /absolute/path/validation_labels.csv \
  --head-type ours_pair_attention --primitive-selection pae_deployment \
  --basis-mode equivariant_v2 --device cuda:0 \
  --learning-rate 0.0002 --batch-size 32 --epochs 40 \
  --hidden-dim 96 --latent-count 8 --attention-heads 4 \
  --weight-decay 0.0001 --endpoint-scale-angstroms 4 \
  --seed 20260820 --deterministic --save-epochs 8 \
  --official-tmscore-binary /absolute/path/TMscore \
  --output-dir outputs/training
```

These commands use Bash continuation syntax; join the lines when running them in Windows PowerShell. Input data and generated outputs are supplied locally.

## Checkpoint

`checkpoints/pairrefine_epoch008.pt` contains the paper's epoch 8 model from a 40-epoch run: 68,550 parameters, hidden dimension 96, 8 latent queries, and 4 attention heads. Training starts from scratch with deterministic execution. The protocol uses PAE deployment primitive selection, `equivariant_v2` local frames, pose scale 1.0, and no confidence gate.



The portable checkpoint preserves all 17 original tensors and loads with `torch.load(..., weights_only=True)`. It was selected retrospectively after external evaluation; it is distinct from a new run's validation-selected `best_model.pt`. [Checkpoint provenance](checkpoints/provenance.json) records its configuration and selection history.

## Input format

All structures use a shared sequence index of length L and coordinates in angstroms. CSV and NPZ files are supplied by the caller; NPZ arrays must load with `allow_pickle=False`.

### CSV manifests

The input manifest requires `target` and `cache_npz` columns. The partition manifest requires `target` and `partition_npz`. Target identifiers must be unique ignoring case, and match between manifests. Use letters, numbers, underscores, hyphens, or periods in target names, beginning with a letter or number.

For training, both manifests also require `split`, with values `train` or `validation`. Paths in training manifests must be absolute or resolvable from the current working directory. Prediction and evaluation also accept paths relative to their containing manifest directory.

```csv
target,cache_npz
protein1,/absolute/path/protein1.npz
```

```csv
target,partition_npz
protein1,/absolute/path/protein1_partitions.npz
```

### Prediction inputs

The cache NPZ contains:

| Array | Shape and type | Meaning |
| --- | --- | --- |
| `sequence` | Scalar Unicode string | Sequence defining the residue index |
| `coordinates` | L x 3, float32 | Predicted C-alpha coordinates |
| `residue_mask` | L, bool | Valid mapped residues |
| `plddt` | L, float32 | Per-residue predicted confidence |
| `pae` | L x L, float32 | Directed predicted aligned error |

The partition NPZ contains `partition_ids`, an L-element integer vector of predicted primitive identifiers. Negative identifiers mark residues outside named primitives. Arrays must have matching lengths and finite numeric values; PAE values must be nonnegative.

Partitions must come from predicted structures. Optional partition-manifest fields `provider`, `provider_version`, and `input_source` describe their origin. Prediction rejects these fields when they indicate native or reference partitions.

Primitive selection uses highest symmetric cross-partition PAE. A moving primitive requires at least three valid residues and a nonempty anchor. A single predicted primitive uses the identity fallback. pLDDT is an input feature; output confidence gating is disabled.

### Evaluation inputs

The cache additionally contains `native_coordinates`, an L x 3 float32 reference C-alpha array on the same residue mapping. Baseline, refined, and reference PDB traces use matching residue masks and indices. Evaluation reads the saved `predicted_actions.csv` produced by `predict.py` and requires the original Zhang-lab TMscore executable.

### Training inputs

The cache additionally contains `native_coordinates` and `domain_ids`, an L-element integer vector for training diagnostics. Training and validation use separate supervision CSV files with these columns:

| Column | Value |
| --- | --- |
| `target` | Matching manifest target |
| `best_primitive_id` | Integer identifier of the selected moving primitive; -1 for identity fallback |
| `primitive_selection` | `highest_symmetric_cross_partition_pae` |
| `best_rotation_x_degrees` | Target local rotation x component in degrees |
| `best_rotation_y_degrees` | Target local rotation y component in degrees |
| `best_rotation_z_degrees` | Target local rotation z component in degrees |
| `best_translation_x_angstroms` | Target local translation x component in angstroms |
| `best_translation_y_angstroms` | Target local translation y component in angstroms |
| `best_translation_z_angstroms` | Target local translation z component in angstroms |

Pose targets are supervised rigid actions in the `equivariant_v2` local frame, using the hinge pivot. The trainer checks that `best_primitive_id` matches PAE deployment selection. Reference coordinates, domain annotations, and pose targets are supervision inputs; inference reads the prediction arrays and predicted partitions described above.

## License

Project code is covered by the [MIT license](LICENSE). External dependencies retain their own licenses. The Zhang-lab TMscore executable is installed separately from its authors; this repository provides the evaluation wrapper. Caller-supplied structures and predicted partitions remain subject to their original source terms.
