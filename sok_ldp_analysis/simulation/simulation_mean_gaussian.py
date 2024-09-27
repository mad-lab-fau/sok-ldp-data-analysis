import itertools
import time
import traceback

import numpy as np
import pandas as pd

from sok_ldp_analysis.simulation.data.synthetic_continuous import BimodalData1D, UniformLargeData1D
from sok_ldp_analysis.simulation.data.synthetic_gaussian import GaussianData1D
from sok_ldp_analysis.simulation.methods.mean_gaussian import (
    _init_gaussian_method,
    one_dim_mean_gaussian_methods,
    one_dim_mean_gaussian_eps_delta_methods,
)
from sok_ldp_analysis.simulation.simulation import simulation_loop


def run_mean_gaussian_1d(args):
    store_path, n_range, dataset, eps, delta, beta, method, num_runs, seed = args

    # Skip if delta > 0 for methods by Joseph et al. (they are (eps, 0)-LDP)
    if delta > 0 and "Joseph" in method.__name__:
        return []

    # Skip if delta == 0 for methods by Gaboardi et al. (they are (eps, delta)-LDP)
    if delta == 0 and "Gaboardi" in method.__name__:
        return []

    data = dataset.data
    pop_mean = dataset.pop_mean
    pop_sigma = dataset.pop_sigma

    result_list = []

    rng = np.random.default_rng(seed)

    # set special parameters for Gaboardi methods
    r = abs(pop_mean) + 3 * pop_sigma

    # Set special parameter for UnknownVar methods
    sigma_interval = [1e-10, 2 * pop_sigma]

    for i in range(num_runs):
        for n in n_range:
            # Initialize the mechanism
            mechanism = _init_gaussian_method(method, rng, eps, beta, pop_sigma, sigma_interval, r, delta)

            # Run the mechanism
            try:
                start = time.time()
                est_mean = mechanism.estimate_mean(data[:n])
                end = time.time()

                run_time = end - start
                failed = False
            except (ValueError, AssertionError, RuntimeWarning) as e:
                print(e)
                print(traceback.format_exc())
                est_mean = np.nan
                run_time = np.nan
                failed = True

            result_list.append(
                {
                    "method": method.__name__,
                    "data": str(dataset),
                    "n": n,
                    "eps": eps,
                    "delta": delta,
                    "beta": beta,
                    "run": i,
                    "run_time": run_time,
                    "pop_mean": pop_mean,
                    "pop_sigma": pop_sigma,
                    "sample_mean": np.mean(data[:n]),
                    "sample_sigma": np.std(data[:n]),
                    "est_mean": est_mean,
                    "failed": failed,
                }
            )

    return store_path, result_list


def simulate_mean_gaussian(output_path, use_mpi=False, i=None):
    """
    Run the mean simulation for Gaussian data and Gaussian mechanisms.

    Args:
        output_path: The path to the output directory.
        use_mpi: Whether to use MPI or not. If False, multiprocessing is used.
        i: The index of the method to run. If None, all methods are run.
    """
    seed = 0
    rng = np.random.default_rng(seed)

    # parameters
    n_range = np.logspace(2, 7, 31, dtype=int)[::-1]  # 10^2 to 10^7
    eps_range = np.logspace(-1, 1, 31)  # 10^(-1) to 10^1
    delta_range = [0, 1e-10, 1e-5, 1e-3, 1e-2, 1e-1]  # TODO: which delta values are sensible?
    beta_range = [0.01, 0.05, 0.1]
    num_runs = [100]

    # Set up the data
    n = n_range[0]

    data_rng = rng.spawn(1)[0]
    data = [
        GaussianData1D(data_rng, n, 0, 1),
        GaussianData1D(data_rng, n, 0, 5),
        GaussianData1D(data_rng, n, 42, 0.1),
        GaussianData1D(data_rng, n, 42, 10),
        GaussianData1D(data_rng, n, -123, 10),
        GaussianData1D(data_rng, n, 50000, 50),
        BimodalData1D(data_rng, n, 1, 0),
        UniformLargeData1D(data_rng, n),
    ]  # TODO: decide on Gaussian distributions

    # Select the methods
    methods = one_dim_mean_gaussian_methods + one_dim_mean_gaussian_eps_delta_methods

    if i is not None:
        methods = [methods[i]]

    # Run for each method and write into individual output files
    for method in methods:
        for dataset in data:
            print()
            print(method.__name__)
            print(dataset)

            output_file = output_path / method.__name__ / f"{dataset}.csv"

            if output_file.exists():
                # count the number of rows in the output file
                df = pd.read_csv(output_file)
                num_rows = df.shape[0]

                deltas = len(delta_range) - 1 if method in one_dim_mean_gaussian_eps_delta_methods else 1
                print(f"Output file exists with {num_rows} rows")
                print(
                    f"Expected number of rows: "
                    f"{len(n_range) * len(eps_range) * deltas * len(beta_range) * num_runs[0]}"
                )
                if num_rows == len(n_range) * len(eps_range) * deltas * len(beta_range) * num_runs[0]:
                    print("Skipping this task")
                    continue
                else:
                    print("Task did not finish. Restarting...")
                    output_file.unlink()

            # Prepare random seeds
            seeds = np.random.SeedSequence(seed).spawn(len(num_runs))

            # Set up the task list
            tasks = list(
                itertools.product([n_range], [dataset], eps_range, delta_range, beta_range, [method], num_runs, seeds)
            )

            # Run the simulation
            simulation_loop(output_file, run_mean_gaussian_1d, tasks, use_mpi=use_mpi)
