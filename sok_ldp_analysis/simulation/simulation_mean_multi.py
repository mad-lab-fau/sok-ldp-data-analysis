import time
import traceback

import numpy as np

from sok_ldp_analysis.ldp.mean import Duchi2018
from sok_ldp_analysis.ldp.mean.duchi2018 import Duchi2018LInf
from sok_ldp_analysis.ldp.mean.mean import MeanMultiDim
from sok_ldp_analysis.ldp.mean.wang2019 import Wang2019
from sok_ldp_analysis.simulation.data.synthetic import MultiDatasetSynth
from sok_ldp_analysis.simulation.data.synthetic_discrete import BinomialData1D
from sok_ldp_analysis.simulation.simulation import simulation_core_loop


def run_mean_multi_dim(args):
    store_path, n_range, dataset, eps, method, num_runs, seed = args

    data = dataset.data
    input_range = dataset.input_range
    rng = np.random.default_rng(seed)

    result_list = []

    for i in range(num_runs):
        for n in n_range:
            data_ = data[:n, :]

            try:
                # Initialize the mechanism
                mechanism = method(eps=eps, rng=rng, input_range=input_range)

                # Run the mechanism
                start = time.time()
                est_mean = mechanism.estimate_mean(data_)
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
                    "d": dataset.d,
                    "n": n,
                    "eps": eps,
                    "run": i,
                    "failed": failed,
                    "run_time": run_time,
                    "sample_mean": np.mean(data_, axis=0),
                    "sample_sigma": np.std(data_, axis=0),
                    "est_mean": est_mean,
                }
            )

    return store_path, result_list


def simulate_mean_multi_dim(output_path, use_mpi=False, i=None):
    """
    Run the multi-dimensional mean simulation.

    Args:
        output_path: The path to the output directory.
        use_mpi: Whether to use MPI or not. If False, multiprocessing is used.
        i: The index of the method to run. If None, all methods are run.
    """

    seed = 0
    rng = np.random.default_rng(seed)

    # parameters
    n_range = [100, 1000, 10000, 100000][::-1]
    eps_range = np.logspace(-1, 1, 16)  # 10^(-1) to 10^1
    num_runs = [100]

    # Set up the data
    n = n_range[0]
    data_rng = rng.spawn(1)[0]

    # create datasets of different dimensionality
    base_data = BinomialData1D(rng=data_rng, size=n, domain_size=100, p=0.2)
    datasets = [MultiDatasetSynth(n, base_data, d) for d in [1, 2, 4, 8, 16, 32, 64]]

    # Select the methods
    methods = [Duchi2018, Duchi2018LInf, Wang2019, Wang2019K1, Wang2019SplitEps, Wang2019SplitN]

    if i is not None:
        methods = [methods[i]]

    simulation_core_loop(
        output_path, run_mean_multi_dim, methods, datasets, seed, num_runs, n_range, eps_range, use_mpi
    )


class Wang2019K1(Wang2019):
    def __init__(self, eps, input_range, rng):
        super().__init__(eps, input_range, rng, k=1)


class Wang2019SplitEps(MeanMultiDim):
    def __init__(self, eps, input_range, rng):
        super().__init__(eps, input_range, rng)

    def mechanism(self, data):
        d = data.shape[1]

        self.base_method = Wang2019(self.eps / d, self.input_range, self.rng)

        output = np.zeros(data.shape)
        for i in range(d):
            output[:, i] = self.base_method.mechanism(data[:, i][:, np.newaxis])[:, 0]

        return output

    def mean(self, u):
        d = u.shape[1]
        means = np.zeros(d)
        for i in range(d):
            means[i] = self.base_method.mean(u[:, i])

        return means


class Wang2019SplitN(MeanMultiDim):
    def __init__(self, eps, input_range, rng):
        self.base_method = Wang2019(eps, input_range, rng)
        super().__init__(eps, input_range, rng)

    def mechanism(self, data):
        n = data.shape[0]
        d = data.shape[1]
        n_partial = n // d
        output = []
        for i in range(d):
            output.append(self.base_method.mechanism(data[i * n_partial : (i + 1) * n_partial, i][:, np.newaxis])[:, 0])

        return output

    def mean(self, u):
        d = len(u)
        means = np.zeros(d)
        for i, vals in enumerate(u):
            means[i] = self.base_method.mean(vals)

        return means
