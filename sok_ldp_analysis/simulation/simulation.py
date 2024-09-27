import copy
import itertools
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd

try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError:
    MPIPoolExecutor = None

from sok_ldp_analysis.simulation.data.real import RealDataset


def simulation_loop(run_method, tasks, use_mpi=False):
    """
    Run a simulation in parallel and save the results to pickle files.

    Args:
        run_method: The method to run in parallel.
        tasks: The list of tasks to run.
        use_mpi: Whether to use MPI or not. If False, multiprocessing is used.
    """

    if MPIPoolExecutor is None and use_mpi:
        raise ImportError("mpi4py is not installed. Cannot use MPI.")

    # Run the simulation in parallel
    pool_ = MPIPoolExecutor if use_mpi and MPIPoolExecutor is not None else multiprocessing.Pool
    max_workers = None if use_mpi else os.cpu_count()
    with pool_(max_workers) as pool:
        if use_mpi:
            mapping = pool.map(run_method, tasks, chunksize=1, unordered=True)
        else:
            mapping = pool.imap_unordered(run_method, tasks, chunksize=1)

        for output_path, result in mapping:
            print(f"got results - len {len(result)}")
            if len(result) == 0:
                continue

            eps = result[0]["eps"]
            filename = output_path / f"{eps}.pkl"

            print(f"Saving results for eps={eps} to {filename}...")

            if os.path.exists(filename):
                os.remove(filename)

            # pickle result
            with open(filename, "wb") as f:
                pickle.dump(result, f)

            print(f"Saved results for eps={eps} to {filename}")


def _migrate_old_results(output_file, output_path):
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(output_file)

    eps = df.groupby(["eps"]).size().index

    for eps_ in eps:
        df_eps = df[df["eps"] == eps_]
        filename = output_path / f"{eps_}.pkl"

        if os.path.exists(filename):
            os.remove(filename)

        df_eps.to_pickle(filename)

    output_file.unlink()


def filter_eps_range(store_path, eps_range, count):
    # Check if this task is already done (new storage format)
    eps_to_remove = []
    for file in store_path.glob("*.pkl"):
        df_or_list = pickle.load(open(file, "rb"))
        if len(df_or_list) < count:
            print(f"Removing {file}. {len(df_or_list)} != {count}")
            file.unlink()
        else:
            print(f"Already done {file}")
            if type(df_or_list) is list:
                eps = df_or_list[0]["eps"]
            elif type(df_or_list) is pd.DataFrame:
                eps = df_or_list["eps"].iloc[0]
            else:
                raise ValueError(f"Unknown storage type: {type(df_or_list)}")

            eps_to_remove.append(eps)

    print(eps_to_remove)

    # Precision of stored eps might lead to different eps values
    eps_range_ = [eps for eps in eps_range if not any(abs(eps - eps_) < 1e-10 for eps_ in eps_to_remove)]

    return eps_range_


def simulation_core_loop(output_path, run_method, methods, datasets, seed, num_runs, n_range, eps_range, use_mpi=False):
    tasks = []
    for method in methods:
        for dataset in datasets:
            print()
            print(method.__name__)
            print(dataset)

            output_file = output_path / method.__name__ / f"{dataset}.csv"
            store_path = output_path / method.__name__ / str(dataset)

            n = len(n_range)

            if issubclass(type(dataset), RealDataset):
                count = len(dataset.data)
                if count < n_range[0]:
                    n = sum([x <= count for x in n_range])

            # Migrate the results from the old format
            if output_file.exists():
                print("Migrating old results")
                _migrate_old_results(output_file, store_path)

            eps_range_ = filter_eps_range(store_path, eps_range, n * num_runs[0])

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
                        [method],
                        num_runs,
                        seeds,
                    )
                )
            )

    # Run the simulation
    simulation_loop(run_method, tasks, use_mpi=use_mpi)
