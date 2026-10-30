# Input format

All structures use a shared sequence index of length L and coordinates in angstroms. CSV and NPZ files are supplied by the caller; NPZ arrays must load with `allow_pickle=False`.

## CSV manifests

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

## Prediction inputs

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

## Evaluation inputs

The cache additionally contains `native_coordinates`, an L x 3 float32 reference C-alpha array on the same residue mapping. Baseline, refined, and reference PDB traces use matching residue masks and indices. Evaluation reads the saved `predicted_actions.csv` produced by `predict.py` and requires the original Zhang-lab TMscore executable.

## Training inputs

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
