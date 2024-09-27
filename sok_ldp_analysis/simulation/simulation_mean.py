import time
import traceback

import numpy as np

from sok_ldp_analysis.ldp.mean.mean import MeanMultiDim
from sok_ldp_analysis.simulation.data.real import Adult, NYCTaxiData
from sok_ldp_analysis.simulation.data.synthetic_continuous import (
    BimodalData1D,
    UniformLargeData1D,
    UniformSmallData1D,
)
from sok_ldp_analysis.simulation.data.synthetic_discrete import BinomialData1D
from sok_ldp_analysis.simulation.methods.mean import one_dim_mean_methods, multi_dim_mean_methods
from sok_ldp_analysis.simulation.simulation import simulation_core_loop


def run_mean_1d(args):
    store_path, n_range, dataset, eps, method, num_runs, seed = args

    data = dataset.data
    input_range = dataset.input_range
    rng = np.random.default_rng(seed)

    # Add a dimension to the data for the multi-dimensional methods
    if issubclass(method, MeanMultiDim):
        data = data.reshape((-1, 1))

    result_list = []

    for i in range(num_runs):
        for n in n_range:
            data_ = data[:n]

            try:
                # Initialize the mechanism
                mechanism = method(eps=eps, rng=rng, input_range=input_range)

                # Run the mechanism
                start = time.time()
                resp = mechanism.mechanism(data_)
                est_mean = mechanism.mean(resp)
                end = time.time()

                resp_mean = np.mean(resp)
                resp_var = np.var(resp)

                run_time = end - start
                failed = False
            except (ValueError, AssertionError, RuntimeWarning) as e:
                print(e)
                print(traceback.format_exc())
                est_mean = np.nan
                resp_mean = np.nan
                resp_var = np.nan
                run_time = np.nan
                failed = True

            result_list.append(
                {
                    "method": method.__name__,
                    "data": str(dataset),
                    "n": n,
                    "eps": eps,
                    "run": i,
                    "failed": failed,
                    "run_time": run_time,
                    "sample_mean": np.mean(data_),
                    "sample_sigma": np.std(data_),
                    "est_mean": est_mean,
                    "resp_mean": resp_mean,
                    "resp_var": resp_var,
                }
            )

    return store_path, result_list


def simulate_mean(output_path, use_mpi=False, i=None):
    """
    Run the mean simulation.

    Args:
        output_path: The path to the output directory.
        use_mpi: Whether to use MPI or not. If False, multiprocessing is used.
        i: The index of the method to run. If None, all methods are run.
    """
    seed = 0
    rng = np.random.default_rng(seed)

    # parameters
    n_range = np.logspace(2, 7, 16, dtype=int)[::-1]  # 10^2 to 10^7
    eps_range = np.logspace(-1, 1, 16)  # 10^(-1) to 10^1
    num_runs = [100]

    # Set up the data
    n = n_range[0]
    data_rng = rng.spawn(1)[0]
    data_discrete = [BinomialData1D(data_rng, n, 100, 0.2)]
    data_continuous = [
        BimodalData1D(data_rng, n, 1, 0),
        UniformLargeData1D(data_rng, n),
        UniformSmallData1D(data_rng, n),
    ]
    data_real = [
        Adult(data_rng, data_path="../data"),
        NYCTaxiData(data_rng, data_path="../data"),
    ]

    datasets = data_continuous + data_discrete + data_real

    # Select the methods
    methods = one_dim_mean_methods + multi_dim_mean_methods

    if i is not None:
        methods = [methods[i]]

    simulation_core_loop(output_path, run_mean_1d, methods, datasets, seed, num_runs, n_range, eps_range, use_mpi)
