# Artifact Appendix

Paper title: **SoK: Descriptive Statistics Under Local Differential Privacy**

Artifacts HotCRP Id: **7**

Requested Badge: **Reproduced**

## Description

This repository provides the implementation of the LDP algorithms, simulations, and analysis/plotting scripts for the
paper "SoK: Descriptive Statistics Under Local Differential Privacy" accepted at PETS 2025.

### Security/Privacy Issues and Ethical Concerns (All badges)

The first run of the simulation may need to download the required real-world datasets.
Once these are downloaded, the simulation will run locally without any further need for internet access and without
any security/privacy issues or ethical concerns.

## Basic Requirements (Only for Functional and Reproduced badges)

### Hardware Requirements

We provide 4 experiments/simulations. All simulations can in theory be run on a standard multi-core laptop or desktop
computer. However, they may take a very long time to run. We recommend running the simulations on a machine with at
least 16 cores and 64GB of RAM to have a reasonable runtime (see below for estimated runtimes).

### Software Requirements

The code has only been tested on Linux operating systems running Python 3.9 or 3.10, but should work on other OSs and
newer Python versions as well.

Make sure you have Python 3.9 (or 3.10) and Poetry installed on your system (see README.md for installation
instructions). We use Poetry to manage the dependencies of the project (see "Environment" section below).

On the artifact evaluation VM (we recommend the "compute VM", you may run the following commands to install the
necessary software:

```
sudo apt install python3-pip python3-venv
pip install pipx
pipx ensurepath
pipx install poetry
```

To reproduce the plots from the paper, you need a LaTeX installation on your system. The following packages are
minimally required (example for Ubuntu / artifact evaluation VM):

```
sudo apt install texlive-latex-recommended cm-super texlive-science texlive-fonts-extra
```

On some systems, a manual installation of `dvipng` may be necessary to generate the plots. You can either install it
via your system package manager (e.g., `sudo apt install dvipng`) or if you have `tlmgr` installed
as `sudo tlmgr install dvipng`.

### Estimated Time and Storage Consumption

Running all experiments on an artifact evaluation VM (16 cores, 64GB RAM) takes 2-3 days. The individual runtimes
are as follows:

- Experiment 1: 5.5 hours
- Experiment 2: 2.5 hours
- Experiment 3: 1 hour
- Experiment 4: 2.5 days

The times can be cut down significantly by running on a machine with more cores.

The full set of results for all experiments takes up roughly 2.5GB of disk space.

## Environment

### Accessibility (All badges)

The project code is available
at [https://github.com/mad-lab-fau/sok-ldp-data-analysis](https://github.com/mad-lab-fau/sok-ldp-data-analysis).
It is licensed under the MIT license.

### Set up the environment (Only for Functional and Reproduced badges)

Make sure you have Python 3.9 and Poetry installed on your system (see README.md for installation instructions).
Clone the repository from the provided link.
Navigate to the repository root and run the following command to install all dependencies:

```bash
poetry install
```

This will create a new virtual environment in the directory `.venv` and install all required dependencies.

If you intend to run the simulation on an HPC cluster and make use of MPI support, you need to install the `mpi4py`
package:

```bash
poetry add mpi4py
```

### Testing the Environment (Only for Functional and Reproduced badges)

If the `poetry install` command ran without any errors, the environment should be set up correctly.

## Artifact Evaluation (Only for Functional and Reproduced badges)

### Main Results and Claims

Our paper mainly consists of a comparison of existing LDP algorithms. Our results are based on simulations of these
algorithms for different datasets, numbers of participants, and privacy parameters. This artifact provides the
implementation of these algorithms, the simulations, and the analysis/plotting scripts to reproduce the figures from the
paper.

The four main results/experiments are:

1. **One-Dimensional Mean Estimation**: We simulate a number of differentially private algorithms for mean estimation
   and compare their utility.
2. **Multi-Dimensional Mean Estimation**: We simulate a number of differentially private algorithms for
   multi-dimensional mean estimation and compare their utility.
3. **Variance Estimation**: We simulate three differentially private algorithms for variance estimation and compare
   their utility for both the mean estimation and the variance estimation aspects.
4. **Frequency Estimation**: We simulate a number of differentially private algorithms for frequency estimation and
   compare their utility.

### Experiments

We assume that your working directory is the root of the repository before each experiment.

#### Experiment 1: One-Dimensional Mean Estimation

Run the simulation:

```bash
cd [path_to_repository]
cd experiments
poetry run python run_simulation.py mean 
```

This should take about 5.5 hours to run on an artifact evaluation "compute" VM (16 cores, 64GB RAM).
After the simulation has finished, the results will be stored in the `results/simulation/mean` folder.
Each method will have its own subfolder with a subfolder for each dataset.
The results will be stored in a pickle files named by the corresponding privacy parameter $\varepsilon$.

Run the simulation of the non-private random-rounding method (needed for the plots). This takes less than 5-10 minutes
on most machines:

```bash
poetry run python run_mean_rr.py 
```

The results will be stored in the `results/simulation/mean_rr` and `results_grouped/simulation/mean_rr` folders.

To further process the results for plotting, run the following script:

```bash
poetry run python preprocess_results.py
```

Note that the script may complain about missing directories for the other experiments, but this can be ignored.

To re-create the plots from the paper, run the following script:

```bash
cd ../plotting
poetry run python plot_paper_mean.py 
```

This will produce the plots for the mean estimation experiment:

- `plotting/plots_paper/mean_methods_diff/n10000_scaled_abs_error.pdf` for Figure 1
- `plotting/plots_paper/mean_methods_diff/n10000_scaled_abs_error_std.pdf` for Figure 5 (appendix)
- `plotting/plots_paper/mean_grid/mean_scaled_abs_error.pdf` for Figure 7 (appendix)
- `plotting/plots_paper/mean_grid/mean_scaled_mse.pdf` for Figure 8 (appendix)
- `plotting/plots_paper/mean_grid/mean_resp_var.pdf` for Figure 9 (appendix)

This experiment supports main result 1: One-Dimensional Mean Estimation.

#### Experiment 2: Multi-Dimensional Mean Estimation

Run the simulation:

```bash
cd [path_to_repository]
cd experiments
poetry run python run_simulation.py mean_multi_dim
```

This should take about 2.5 hours to run on an artifact evaluation "compute" VM (16 cores, 64GB RAM).
After the simulation has finished, the results will be stored in the `results/simulation/mean_multi_dim` folder.
Each method will have its own subfolder with a subfolder for each dataset.
The results will be stored in a pickle files named by the corresponding privacy parameter $\varepsilon$.

To further process the results for plotting, run the following script:

```bash
poetry run python preprocess_results.py
```

To re-create the plots from the paper, run the following script:

```bash
cd ../plotting
poetry run python plot_paper_mean_multi.py 
```

This will produce the plots for the multi-dimensional mean estimation experiment:

- `plotting/plots_paper/mean_multi_dim_method_comparison/mean_multi_scaled_abs_error.pdf` for Figure 2
- `plotting/plots_paper/mean_dim_grid_n_method/mean_multi_scaled_abs_error.pdf` for Figure 10 (appendix)
- `plotting/plots_paper/mean_dim_grid/mean_multi_grid_scaled_abs_error.pdf` for Figure 11 (appendix)

This experiment supports main result 2: Multi-Dimensional Mean Estimation.

#### Experiment 3: Variance Estimation

Run the simulation:

```bash
cd [path_to_repository]
cd experiments
poetry run python run_simulation.py variance --i 2
```

The option `--i 2` specifies that the simulation should only be run for the Piecewise method by Wang et al., 2019.
This is the method that is presented in the paper. Without this option, all mean estimation methods will be run.

After the simulation has finished, the results will be stored in the `results/simulation/variance` folder.
Each method will have its own subfolder with a subfolder for each dataset.
The results will be stored in a pickle files named by the corresponding privacy parameter $\varepsilon$.

To further process the results for plotting, run the following script:

```bash
poetry run python preprocess_results.py
```

To re-create the plots from the paper, run the following script:

```bash
cd ../plotting
poetry run python plot_paper_variance.py
```

This will produce the plots for the variance estimation experiment:

- `plotting/plots_paper/variance_methods_diff/n10000.pdf` for Figure 3
- `plotting/plots_paper/variance_methods_diff/grid_0.pdf` for Figure 12 (appendix)
- `plotting/plots_paper/variance_methods_diff/grid_1.pdf` for Figure 13 (appendix)

This experiment supports main result 3: Variance Estimation.

#### Experiment 4: Frequency Estimation

First, download further required datasets:

```bash
cd [path_to_repository]
mkdir -p data/tmp
cd data/tmp
wget https://archive.ics.uci.edu/static/public/116/us+census+data+1990.zip
unzip us+census+data+1990.zip USCensus1990.data.txt
rm us+census+data+1990.zip
```

Run the simulation (on a computer without MPI support - may take a long time on a laptop):

```bash
cd [path_to_repository]
cd experiments
poetry run python run_simulation.py fo
```

This should take about 2.5 days to run on an artifact evaluation "compute" VM (16 cores, 64GB RAM).
Note that the script may produce some warnings about high privacy and non-convergence, but these can be ignored.
Note that the simulation may skip several instances of the method by Murakami et al., 2018, as they require a lot of
RAM (>12GB per core). This may affect the resulting plots.

Run the simulation (only on an HPC cluster with MPI support):

```bash
poetry run python run_simulation.py --mpi fo
```

After the simulation has finished, the results will be stored in the `results/simulation/fo` folder.
Each method will have its own subfolder with a subfolder for each dataset.
The results will be stored in a pickle files named by the corresponding privacy parameter $\varepsilon$.

To further process the results for plotting, run the following script:

```bash
poetry run python preprocess_results.py
```

To re-create the plots from the paper, run the following script:

```bash
cd ../plotting
poetry run python plot_paper_fo.py
```

This will produce the plots for the frequency estimation experiment:

- `plotting/plots_paper/fo_mse_3n/fo_mse_3n.pdf` for Figure 4
- `plotting/plots_paper/fo_mse_3n/fo_mse_3n_std.pdf` for Figure 6 (appendix)
- `plotting/plots_paper/fo_grid/fo_mse_all.pdf` for Figure 14 (appendix)
- `plotting/plots_paper/fo_grid_var/fo_var_all.pdf` for Figure 15 (appendix)
- `plotting/plots_paper/fo_mse_domainsize_3n/fo_mse_domainsize_grid_3n.pdf` for Figure 16 (appendix)

Note that the plots for Murakami et al., 2018, may differ from the paper due to the skipped instances.

This experiment supports main result 4: Frequency Estimation.

## Limitations

All relevant figures from the paper can be reproduced using the provided artifact.
However, the experiments may take a long time to run on a standard laptop or desktop computer, especially the frequency
estimation experiment.

## Notes on Reusability

The `sok_ldp_analysis` library is a general-purpose library and can be used by other projects to run a number of locally
differentially private algorithms. Currently, the library supports methods for mean and variance estimation,
multi-dimensional mean estimation, and frequency estimation.
Note that the library is optimized for performance and does not fully simulate the communication-aspects of a real-world
implementation.
