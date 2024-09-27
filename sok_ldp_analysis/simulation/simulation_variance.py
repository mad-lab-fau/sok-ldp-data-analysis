import copy
import itertools
import os
import time
import traceback

import numpy as np
import psutil

from sok_ldp_analysis.ldp.mean.variance import (
    MeanAndVarianceSplitEps,
    MeanAndVarianceSplitN,
    MeanAndVarianceSplitNSequential,
)
from sok_ldp_analysis.simulation.data.real import RealDataset
from sok_ldp_analysis.simulation.data.synthetic_continuous import BimodalData1D, UniformLargeData1D, UniformSmallData1D
from sok_ldp_analysis.simulation.data.synthetic_discrete import BinomialData1D
from sok_ldp_analysis.simulation.methods.mean import one_dim_mean_methods, multi_dim_mean_methods
from sok_ldp_analysis.simulation.simulation import filter_eps_range, simulation_loop

tasks_per_node = os.environ.get("SLURM_TASKS_PER_NODE")  # 10(x2)
tasks_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE").split("(")[0]) if tasks_per_node is not None else 1

mem_limit = psutil.virtual_memory().total * 0.8 / tasks_per_node


def run_var(args):
    store_path, n_range, dataset, eps, var_method, base_method, split_ratios, num_runs, seed = args

    print(f"Running {var_method.__name__}/{base_method.__name__} for {dataset} with eps={eps} and seed={seed}")

    data = dataset.data
    rng = np.random.default_rng(seed)

    result_list = []

    for i in range(num_runs):
        for n in n_range:
            for split_ratio in split_ratios:
                if len(data) < n:
                    continue

                data_ = data[:n]

                try:
                    # Initialize the mechanism
                    mechanism = var_method(
                        base_method=base_method,
                        epsilon=eps,
                        rng=rng,
                        input_range=dataset.input_range,
                        split_ratio=split_ratio,
                    )

                    # Run the mechanism
                    start = time.time()
                    est_mean, est_var = mechanism.estimate_mean_and_variance(data_)
                    end = time.time()

                    run_time = end - start
                    failed = False
                except (ValueError, AssertionError, RuntimeWarning) as e:
                    print(e)
                    print(traceback.format_exc())
                    est_mean = np.nan
                    est_var = np.nan
                    run_time = np.nan
                    failed = True

                result_list.append(
                    {
                        "var_method": var_method.__name__,
                        "base_method": base_method.__name__,
                        "data": str(dataset),
                        "n": n,
                        "eps": eps,
                        "split_ratio": split_ratio,
                        "run": i,
                        "failed": failed,
                        "run_time": run_time,
                        "sample_mean": np.mean(data_),
                        "sample_var": np.var(data_),
                        "est_mean": est_mean,
                        "est_var": est_var,
                    }
                )

    return store_path, result_list


def simulate_variance(output_path, use_mpi=False, i=None):
    """
    Run the frequency oracle simulation.

    Args:
        output_path: The path to the output directory.
        use_mpi: Whether to use MPI for simulations. If False, multiprocessing is used.
        i: The index of the method to run. If None, all methods are run.
    """
    seed = 0
    rng = np.random.default_rng(seed)

    # parameters
    n_range = np.logspace(2, 7, 6, dtype=int)[::-1]  # 10^2 to 10^7
    eps_range = [0.1, 0.5, 1, 2, 4, 6, 8, 10]

    num_runs = [20]

    split_ratios = [x * 0.1 for x in range(1, 10)]

    # Set up the data
    n = n_range[0]
    data_rng = rng.spawn(1)[0]
    data_discrete = [BinomialData1D(data_rng, n, 100, 0.2)]
    data_continuous = [
        BimodalData1D(data_rng, n, 1, 0),
        UniformLargeData1D(data_rng, n),
        UniformSmallData1D(data_rng, n),
    ]
    datasets = data_continuous + data_discrete

    # Select the methods
    base_methods = one_dim_mean_methods + multi_dim_mean_methods

    variance_methods = [MeanAndVarianceSplitEps, MeanAndVarianceSplitN, MeanAndVarianceSplitNSequential]

    if i is not None:
        base_methods = [base_methods[i]]

    print(f"Memory limit: {mem_limit / 1024**3} GB")

    tasks = []
    for variance_method in variance_methods:
        for base_method in base_methods:
            for dataset in datasets:
                print()
                print(variance_method.__name__)
                print(base_method.__name__)
                print(dataset)

                store_path = output_path / variance_method.__name__ / base_method.__name__ / str(dataset)

                n = len(n_range)

                if issubclass(type(dataset), RealDataset):
                    count = len(dataset.data)
                    if count < n_range[0]:
                        n = sum([x <= count for x in n_range])

                eps_range_ = filter_eps_range(store_path, eps_range, n * num_runs[0] * len(split_ratios))

                print(eps_range_)

                if len(eps_range_) == 0:
                    continue

                # Prepare random seeds
                if type(seed) is np.random.Generator:
                    seeds = seed.spawn(len(num_runs))
                else:
                    seeds = np.random.SeedSequence(seed).spawn(len(num_runs))

                # Load/Generate the data
                dataset_ = copy.deepcopy(dataset)
                dataset_.load_data()

                store_path.mkdir(parents=True, exist_ok=True)
                # Set up the task list
                tasks.extend(
                    list(
                        itertools.product(
                            [store_path],
                            [n_range],
                            [dataset_],
                            eps_range_,
                            [variance_method],
                            [base_method],
                            [split_ratios],
                            num_runs,
                            seeds,
                        )
                    )
                )

    # Run the simulation
    simulation_loop(run_var, tasks, use_mpi=use_mpi)
