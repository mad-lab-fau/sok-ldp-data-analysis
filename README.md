# SoK: Descriptive Statistics Under Local Differential Privacy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13683977.svg)](https://doi.org/10.5281/zenodo.13683977)

This repository contains the code for the corresponding paper "SoK: Descriptive Statistics Under Local Differential
Privacy" accepted at PETS 2025.

An eprint of the paper is available at [https://eprint.iacr.org/2024/1464](https://eprint.iacr.org/2024/1464).

## Usage

To work with the project you need to install [poetry](https://python-poetry.org/docs/#installation) and have a working
python environment. The project was tested with python 3.9 and 3.10.

See https://github.com/mad-lab-fau/mad-cookiecutter/blob/main/python-setup-tips.md#global-tooling for a convenient way
of installing poetry globally without polluting your global python environment based on pipx.

Afterwards run:

```
poetry install
```

All dependencies are manged using `poetry`.
Poetry will automatically create a new venv for the project, when you run `poetry install`.

See [ARTIFACT-EVALUATION.md](ARTIFACT-EVALUATION.md) for instructions on how to reproduce the results of the paper.

## Results from the paper

We have uploaded the simulation results used for producing the figures in the paper to Zenodo. You can find the data
here: [https://doi.org/10.5281/zenodo.13683977](https://doi.org/10.5281/zenodo.13683977)

Unzip the results into the `experiments` folder (`experiments/results` and `experiments/results_grouped`) to reproduce
the figures using the scripts in the `plotting` folder.