# Existing paper evaluation evidence

`casp13/` contains the mixed-CASP 13-target subset (CASP14, CASP15, and CASP16); the legacy folder name does not denote the CASP13 competition. `cath52/` contains the Chainsaw CATH 52-target subset.

- `official_tmscore_summary.json` and `official_tmscore_per_target.csv` contain the official Zhang-lab TMscore results: native mapped CA-trace normalization and common-residue RMSD. Positive delta TM and positive RMSD reduction indicate improvement.
- `p3a_vs_afdb_per_target.csv` is the byte-identical saved prediction/action record used by the official evaluator. Its internal coordinate metrics are auxiliary and differ from the official metrics.
- `predicted_actions.csv` extracts the selected primitive, PAE information, and predicted rotation/translation from that record.
- `prediction_metadata.json` preserves original prediction metadata with local paths sanitized.
- `raw_tmscore.zip` contains the original parsed official tool outputs with local structure paths rewritten, plus source/export hashes. Logical `mapped_ca/` filenames identify evaluation inputs; the mapped PDB files are not included in this archive.

`evidence_index.json` records original and packaged hashes. Numeric values and existing provenance hashes were preserved. The official summary's `inputs.saved_results_sha256` can be checked against the included byte-identical prediction CSV. Original manifest and partition hashes identify the local evaluation artifacts; sanitized public manifests may have different hashes.

These are existing results from the paper's retrospectively selected epoch-8 checkpoint. A separate release regression with the portable checkpoint reproduced all 65 target-level official results; it did not replace the archived evidence. The mixed-CASP inputs are AFDB v6 structures acquired after the CASP events.
