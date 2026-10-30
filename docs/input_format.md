# Input format

The release operates on C-alpha traces. Each CSV manifest has one row per target and contains `target` and `cache_npz`. A partition manifest contains `target` and `partition_npz`. Target names must be unique, ignoring case.

`predict.py` and `evaluate.py` resolve relative asset paths against the containing manifest directory. The frozen metadata under `data/manifests` is repository-root-relative; `prepare_data.py` translates it into absolute machine-local manifests under `data/local`. The original training entry point expects those resolved paths.

## Prediction cache

NPZ files must use arrays loadable with `allow_pickle=False`:

| Array | Shape | Meaning |
| --- | --- | --- |
| `sequence` | scalar Unicode string | Sequence defining the shared residue index |
| `coordinates` | L x 3, float32 | Predicted CA coordinates in angstroms |
| `residue_mask` | L, bool | Valid residues retained by the supplied mapping |
| `plddt` | L, float32 | Per-residue predicted confidence |
| `pae` | L x L, float32 | Directed predicted aligned error in angstroms |

The partition NPZ contains `partition_ids`, an L-element integer vector on the same residue index. Negative identifiers denote residues outside named primitives. All inputs must have matching lengths and finite values. Prediction neither loads nor requires native coordinates, reference domain labels, or supervision files.

A single nonnegative predicted primitive identifier triggers the original identity fallback. With multiple primitive identifiers, the selected moving primitive must have at least three valid residues and a nonempty anchor; invalid geometry is rejected. Primitive selection is fixed by highest symmetric cross-partition PAE; all moving-to-anchor residue pairs are retained as tokens. pLDDT is an input feature, and is not used as an output confidence gate.

## Evaluation and training

Evaluation additionally reads `native_coordinates` (L x 3, float32), on the same fixed mapping. The official comparison writes baseline, refined, and reference CA traces with the same mask and residue numbers. Mapping or mask changes define a different evaluation protocol.

Training additionally requires `domain_ids` (L integers), `split` fields equal to `train` or `validation`, and the supplied supervision CSVs. `domain_ids` supports training diagnostics and is not a prediction input. The supervision CSV pose targets were generated separately from native structures; see `data/README.md` for their role and provenance.

For new proteins, prepare the predicted structure, its corresponding PAE/pLDDT, and a predicted partition on a common residue index. Supply native coordinates only for evaluation or supervision.
