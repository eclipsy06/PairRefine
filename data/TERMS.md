# Data and external dependencies

PairRefine's code license does not replace the terms of any source database,
external program, pretrained weight file, or third-party annotation.

| Source | Use in this project | Availability and terms |
| --- | --- | --- |
| MPDB multi-domain chains and annotations | Native structures and training/validation domain annotations; native chain extraction for CATH inputs | Obtain the original data associated with [Peng et al. 2022](https://doi.org/10.1093/bioinformatics/btac553). An explicit redistribution license was not present in the local source archive, so its original/processed coordinate caches are not distributed here. |
| AlphaFold Protein Structure Database | Frozen v6 predicted structures, pLDDT and PAE | [AFDB](https://alphafold.ebi.ac.uk/); obtain the explicit recorded versioned URLs under the provider's data terms and preserve attribution. |
| Protein Data Bank | Experimental native structures | [RCSB PDB usage policies](https://www.rcsb.org/pages/policies); per-target PDB IDs and structure download URLs are provided. |
| CASP | Target identifiers, official target sequences and domain metadata | [Prediction Center](https://predictioncenter.org/); obtain the corresponding CASP14/15/16 source material and retain its attribution. |
| Chainsaw | Fixed structure-derived partition provider and CATH test split | [Wells et al. 2024](https://doi.org/10.1093/bioinformatics/btae296). The inspected source revision has an MIT license, copyright 2024 Jude Wells; install its software/weights separately under their original terms. |
| Merizo | Fixed structure-derived partition proposals | [Lau et al. 2023](https://doi.org/10.1038/s41467-023-43934-4), [source repository](https://github.com/psipred/merizo). The inspected source uses GPL v3; its source and weights are not bundled into this repository. |

The included partition archive contains prediction labels only, with provider
provenance. The pose-label tables contain PairRefine's computed supervision
targets. No grant of rights to the original databases or external model
implementations is implied by their inclusion.
