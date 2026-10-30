# Third-party resources

The repository's MIT license applies to the project code. It does not relicense upstream software, structures, or databases.

| Resource | Role | Distribution in this repository |
| --- | --- | --- |
| PyTorch and NumPy | Model execution and numerical operations | Installed separately |
| Biopython and Parasail | Optional raw-structure parsing and sequence alignment | Installed separately |
| AlphaFold Protein Structure Database | Frozen predicted structures, pLDDT, and PAE | Source identifiers, versioned URLs, and hashes; coordinate caches are obtained separately |
| MPDB | Reference multi-domain structures and annotations used for training/validation | Frozen identifiers and derived project artifacts; original structures are obtained separately |
| CASP and PDB | Evaluation references and target metadata | Identifiers and source information; original coordinate files are obtained separately |
| Chainsaw | Frozen base partitions | Predicted partition labels; upstream code/weights are obtained separately |
| Merizo | Frozen proposal partitions | Predicted partition labels; upstream code/weights are obtained separately |
| Zhang-lab TMscore | Official TM-score and common-residue RMSD | Parsed results and raw output; executable is obtained separately |

The local Chainsaw distribution identifies its code license as MIT, copyright 2024 Jude Wells. The local Merizo distribution identifies its code license as GPL v3. Neither implementation nor its weights is incorporated into PairRefine's MIT source package.

Use the original resource's terms for downloading, using, or redistributing its materials. Detailed versioned AFDB URLs, source identifiers, and hashes are preserved in `data/manifests` and `data/provenance.json`.
