from pathlib import Path

import numpy as np
import pandas as pd

from sok_ldp_analysis.simulation.data.real import Adult, NYCTaxiData
from sok_ldp_analysis.simulation.data.synthetic_continuous import UniformSmallData1D, BimodalData1D, UniformLargeData1D
from sok_ldp_analysis.simulation.data.synthetic_discrete import BinomialData1D
from sok_ldp_analysis.simulation.processing import _calculate_errors


def random_rounding(data, rng):
    # data is in [0,1]
    assert np.all(data >= 0) and np.all(data <= 1)

    # sample from a bernoulli distribution with probability equal to the data
    return rng.binomial(1, data)


def simulate_rr(result_path):
    rng = np.random.default_rng(0)
    n_range = [1000000, 100000, 10000, 1000, 100]
    num_runs = 1000

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
        Adult(data_path="../data", data_rng=data_rng),
        NYCTaxiData(data_path="../data", data_rng=data_rng),
    ]

    datasets = data_continuous + data_discrete + data_real

    results = {
        "method": [],
        "data": [],
        "n": [],
        "eps": [],
        "run": [],
        "failed": [],
        "run_time": [],
        "sample_mean": [],
        "est_mean": [],
    }

    for dataset in datasets:
        print(dataset)
        data = dataset.data
        input_range = dataset.input_range

        # Transform the data to [0,1]
        data_trans = (data - input_range[0]) / (input_range[1] - input_range[0])

        for n in n_range:
            print(n)
            data_ = data[:n]
            data_trans_ = data_trans[:n]

            for i in range(num_runs):
                rr = random_rounding(data_trans_, rng)
                rr_mean = np.mean(rr)

                # Transform the mean back to the original domain
                rr_mean = rr_mean * (input_range[1] - input_range[0]) + input_range[0]

                results["method"].append("RandomRounding")
                results["data"].append(str(dataset))
                results["n"].append(n)
                results["eps"].append(np.nan)
                results["run"].append(i)
                results["failed"].append(False)
                results["run_time"].append(np.nan)
                results["sample_mean"].append(np.mean(data_))
                results["est_mean"].append(rr_mean)

    # create a Dataframe and store it
    df = pd.DataFrame(results)
    df.to_pickle(result_path)


def process_rr(result_path, processed_path):
    df = pd.read_pickle(result_path)

    df, agg_columns = _calculate_errors(df)

    agg_columns = ["est_mean", "sample_mean"] + agg_columns
    df_grouped = df.groupby(
        [
            "method",
            "data",
            "n",
        ]
    ).agg({k: ["mean", "std"] for k in agg_columns})

    df_grouped_all_data = df.groupby(
        [
            "method",
            "n",
        ]
    ).agg({k: ["mean", "std"] for k in agg_columns})

    df_grouped.columns = ["_".join(col).strip() for col in df_grouped.columns.values]
    df_grouped.reset_index(inplace=True)

    df_grouped_all_data.columns = ["_".join(col).strip() for col in df_grouped_all_data.columns.values]
    df_grouped_all_data.reset_index(inplace=True)
    df_grouped_all_data["data"] = "all"

    # Merge the two dataframes
    df_grouped = pd.concat([df_grouped, df_grouped_all_data], ignore_index=True)

    df_grouped.to_pickle(processed_path)


if __name__ == "__main__":
    result_path = Path("results") / "simulation" / "mean_rr" / "rr_results.pkl"
    processed_path = Path("results_grouped") / "simulation" / "mean_rr" / "rr_results.pkl"

    result_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    simulate_rr(result_path)
    process_rr(result_path, processed_path)
