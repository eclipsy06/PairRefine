# PairRefine

Model implementation of **PairRefine: A Residue-Pair Guided Primitive-Level Refinement Model for Protein Structure Prediction**.

PairRefine refines predicted C-alpha coordinates with a residue-pair attention pose head. This repository contains the model, training and inference entry points, official evaluation wrapper, and paper checkpoint. The implementation retains the internal package name `iterblob_demo` and head identifier `ours_pair_attention` for checkpoint compatibility.

## Install

Tested with Python 3.9.23, NumPy 2.0.2, and PyTorch 2.8.0.

```bash
python -m pip install -r requirements.txt
```

For GPU training, install the appropriate PyTorch 2.8.0 CUDA build using the [PyTorch instructions](https://pytorch.org/get-started/previous-versions/). Training validation and evaluation also require the original [Zhang-lab TM-score executable](https://zhanggroup.org/TM-score/), installed separately.

## Predict

Prepare your input NPZ files and CSV manifests using the [input format](docs/input_format.md).

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

Checkpoint SHA256: `5896137fe2880cd4c00e2fdaccc02b61bf86951393720aaea1799c66da503470`.

The portable checkpoint preserves all 17 original tensors and loads with `torch.load(..., weights_only=True)`. It was selected retrospectively after external evaluation; it is distinct from a new run's validation-selected `best_model.pt`. [Checkpoint provenance](checkpoints/provenance.json) records its configuration and selection history.

## License

Project code is covered by the [MIT license](LICENSE). See [third-party resources](THIRD_PARTY.md) for external dependencies.
