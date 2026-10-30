# Third-party resources

PairRefine project code is covered by the repository's [MIT license](LICENSE). External dependencies retain their own licenses.

| Resource | Role | Installation |
| --- | --- | --- |
| [PyTorch](https://pytorch.org/) | Model training and inference | `requirements.txt` or the appropriate CUDA build |
| [NumPy](https://numpy.org/) | Numerical arrays and NPZ input | `requirements.txt` |
| [Zhang-lab TMscore](https://zhanggroup.org/TM-score/) | Official TM-score and common-residue RMSD | Obtain the executable separately from its authors |

The repository includes the project's wrapper for TMscore; the third-party executable is installed separately. Caller-supplied structures and predicted partitions remain subject to their original source terms.
