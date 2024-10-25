# ml4paleo

This repository contains code for automated segmentation of fossil X-ray tomography data. This is a companion repository to [_"Automated 3D image segmentation of synchrotron scanned fossils"_](https://www.biorxiv.org/content/10.1101/2024.10.23.619778v1) by During and Matelsky et al. (2024).

This codebase comprises two main components:

1. A Python package for volumetric imagery analysis, training, and evaluating segmentation models, as well as post-processing of segmentation results such as mesh generation and image-stack export.
2. A web application for no-code image conversion, segmentation, and post-processing.

The web application is contained in [`webapp/`](webapp/), and the Python package is contained in [`ml4paleo/`](ml4paleo/). The `ml4paleo` package is also available on PyPI as [`ml4paleo`](https://pypi.org/project/ml4paleo/), and can be installed with `pip install ml4paleo`.

For more details on each of the components, see the READMEs in their respective directories.

## To eject poetry to requirements.txt

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```


## Citation

If this work is useful to your research, please cite:

> **Automated segmentation of synchrotron-scanned fossils**
> Melanie A.D. During, Jordan K. Matelsky, Fredrik K. Gustafsson, Dennis F.A.E. Voeten, Donglei Chen, Brock A. Wester, Konrad Paul Kording, Per E. Ahlberg, Thomas B Schön
> bioRxiv 2024.10.23.619778; doi: https://doi.org/10.1101/2024.10.23.619778 

```bibtex
@article {During2024.10.23.619778,
	author = {During, Melanie A.D. and Matelsky, Jordan K. and Gustafsson, Fredrik K. and Voeten, Dennis F.A.E. and Chen, Donglei and Wester, Brock A. and Kording, Konrad Paul and Ahlberg, Per E. and Sch{\"o}n, Thomas B},
	title = {Automated segmentation of synchrotron-scanned fossils},
	elocation-id = {2024.10.23.619778},
	year = {2024},
	doi = {10.1101/2024.10.23.619778},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/10/25/2024.10.23.619778},
	eprint = {https://www.biorxiv.org/content/early/2024/10/25/2024.10.23.619778.full.pdf},
	journal = {bioRxiv}
}
```
