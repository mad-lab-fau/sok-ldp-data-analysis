import os
import time
import traceback

import numpy as np
import psutil

from sok_ldp_analysis.ldp.distribution.frequency.k_subset import KSubset
from sok_ldp_analysis.ldp.distribution.frequency.murakami2018 import Murakami2018
from sok_ldp_analysis.simulation.data.real import USCensus
from sok_ldp_analysis.simulation.data.synthetic_discrete import BinomialData1D, GeometricData1D
from sok_ldp_analysis.simulation.methods.frequency import pure_frequency_oracles, non_pure_frequency_oracles
from sok_ldp_analysis.simulation.simulation import simulation_core_loop

tasks_per_node = os.environ.get("SLURM_TASKS_PER_NODE")  # 10(x2)
tasks_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE").split("(")[0]) if tasks_per_node is not None else 1

mem_limit = psutil.virtual_memory().total * 0.8 / tasks_per_node


def run_fo(args):
    store_path, n_range, dataset, eps, method, num_runs, seed = args

    data = dataset.data
    domain_size = dataset.domain_size
    rng = np.random.default_rng(seed)

    result_list = []

    for n in n_range:
        for i in range(num_runs):
            if len(data) < n:
                continue

            data_ = data[:n]

            try:
                if method == KSubset and n * domain_size > mem_limit:
                    raise ValueError(f"KSubset needs too much memory for n={n} and domain_size={domain_size}")

                if (
                    method == Murakami2018
                    and (3 * n * domain_size + 4 * n + 3 * n * domain_size * domain_size) * 8 > mem_limit
                ):
                    raise ValueError(f"Murakami2018 needs too much memory for n={n} and domain_size={domain_size}")

                # Initialize the mechanism
                mechanism = method(eps=eps, rng=rng, domain_size=domain_size)

                # Run the mechanism
                start = time.time()
                est_frequencies = mechanism.estimate_frequencies(data_)
                end = time.time()

                del mechanism

                run_time = end - start
                failed = False
            except (ValueError, AssertionError, RuntimeWarning) as e:
                print(e)
                print(traceback.format_exc())
                est_frequencies = np.nan
                run_time = np.nan
                failed = True

            sample_freq = np.bincount(data_, minlength=domain_size) / len(data_)

            result_list.append(
                {
                    "method": method.__name__,
                    "data": str(dataset),
                    "n": n,
                    "eps": eps,
                    "run": i,
                    "failed": failed,
                    "run_time": run_time,
                    "sample_freq": sample_freq,
                    "est_freq": est_frequencies,
                }
            )

    return store_path, result_list


def simulate_fo(output_path, use_mpi=False, i=None):
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
    n_range = [100, 1000, 10000, 100000, 1000000][::-1]
    eps_range = np.logspace(-1, 1, 16)  # 10^(-1) to 10^1

    num_runs = [20]

    # Set up the data
    n = n_range[0]

    data_rng = rng.spawn(1)[0]
    datasets = []

    for domain_size in [8, 128]:  # , 1024]:  # , 4096]:
        datasets += [
            BinomialData1D(data_rng, n, domain_size, 0.2),
            GeometricData1D(data_rng, n, domain_size, None),
        ]

    datasets.extend([GeometricData1D(data_rng, n, d, None) for d in [16, 32, 64, 256, 512]])

    datasets.append(USCensus(data_rng, data_path="../data"))

    # Select the methods
    if i is None:
        methods = non_pure_frequency_oracles + pure_frequency_oracles
    elif i == -1:
        methods = non_pure_frequency_oracles
    elif i == -2:
        methods = pure_frequency_oracles
    else:
        methods = [(non_pure_frequency_oracles + pure_frequency_oracles)[i]]

    print(f"Memory limit: {mem_limit / 1024**3} GB")

    simulation_core_loop(output_path, run_fo, methods, datasets, rng, num_runs, n_range, eps_range, use_mpi)
