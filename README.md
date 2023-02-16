# ml4paleo

This repository contains code for automated segmentation of fossil X-ray tomography data. This is a companion repository to [_"Automated 3D image segmentation of synchrotron scanned fossils"_](#) by During, Gustafsson, and Matelsky et al. (2023).

This codebase comprises two main components:

1. A Python package for volumetric imagery analysis, training, and evaluating segmentation models, as well as post-processing of segmentation results such as mesh generation and image-stack export.
2. A web application for no-code image conversion, segmentation, and post-processing.

The web application is contained in [`webapp/`](webapp/), and the Python package is contained in [`ml4paleo/`](ml4paleo/). The `ml4paleo` package is also available on PyPI as [`ml4paleo`](https://pypi.org/project/ml4paleo/), and can be installed with `pip install ml4paleo`.

For more details on each of the components, see the READMEs in their respective directories.
