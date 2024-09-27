import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from sok_ldp_analysis.ldp.distribution.util import project_onto_prob_simplex
from sok_ldp_analysis.simulation.simulation import _migrate_old_results


def _calculate_errors_fo(df: pd.DataFrame):
    df["error"] = df["est_freq"] - df["sample_freq"]
    df["abs_error"] = df["error"].abs()
    df["mse"] = df["error"] ** 2

    errors = ["error", "abs_error", "mse"]

    return df, errors


def parse_string_array(string_array):
    # check if string_array is a nan float
    if type(string_array) is float:
        return string_array

    if type(string_array) is np.ndarray:
        return string_array

    if string_array == "nan":
        return np.nan

    if "..." in string_array:
        return np.nan

    return np.array(list(map(float, string_array[1:-1].split())))


def post_process_results_fo(array):
    if np.isnan(array).any():
        return array

    # check if the sum is not close to 1
    if not np.isclose(np.sum(array), 1) or np.any(array < 0):
        return project_onto_prob_simplex(array)
    else:
        return array


def agg_mean(x):
    for y in x.values:
        if np.isnan(y).any() and not np.all(np.isnan(y)):
            print("Some values are nan, but not all", y)

    values = [y for y in x.values if not np.isnan(y).any()]
    if len(values) == 0:
        return np.nan

    return np.mean(np.stack(values), axis=0)


def agg_std(x):
    values = [y for y in x.values if not np.isnan(y).any()]
    if len(values) == 0:
        return np.nan

    return np.std(np.stack(values), axis=0)


def agg_mean_all(x):
    for y in x.values:
        if np.isnan(y).any() and not np.all(np.isnan(y)):
            print("Some values are nan, but not all", y)

    values = [y for y in x.values if not np.isnan(y).any()]
    if len(values) == 0:
        return np.nan

    return np.mean(np.concatenate(values))


def agg_std_all(x):
    values = [y for y in x.values if not np.isnan(y).any()]
    if len(values) == 0:
        return np.nan

    return np.std(np.concatenate(values))


def group_results_fo():
    output_dir = Path("results_grouped") / "simulation" / "fo"
    input_dir = Path("results") / "simulation" / "fo"

    # Check if the input path exists
    if not input_dir.exists():
        print(f"Path {input_dir} does not exist. Skipping.")
        return

    # delete output dir
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Migrate CSV files
    for method_dir in input_dir.iterdir():
        # find csv files in method dir
        files = list(method_dir.glob("*.csv"))
        for file in files:
            dataset = file.stem
            _migrate_old_results(file, method_dir / dataset)

    # Iterate over all results files
    for method_dir in input_dir.iterdir():
        dfs = []
        for filename in method_dir.glob("**/*.pkl"):
            print(f"Processing {filename}")

            df = pd.read_pickle(filename)
            if type(df) is list:
                df = pd.DataFrame(df, columns=df[0].keys())

            multi_dim_columns = ["sample_freq", "est_freq"]
            for col in multi_dim_columns:
                df[col] = df[col].apply(parse_string_array)

            df["est_freq"] = df["est_freq"].apply(post_process_results_fo)

            dfs.append(df)

        # Join the partial results for different datasets and epsilons
        df = pd.concat(dfs)

        # filter rows where failed=True
        df = df[~df["failed"]]

        # Round epsilons to 10 decimal places
        df["eps"] = df["eps"].round(10)

        # calculate errors
        df, agg_columns = _calculate_errors_fo(df)

        # Calculate the mean and standard deviation of the run time and the errors
        agg_columns = ["est_freq", "sample_freq"] + agg_columns

        agg_columns_multi = []
        for col in agg_columns:
            if type(df[col].iloc[0]) is np.ndarray and len(df[col].iloc[0]) > 1:
                agg_columns_multi.append(col)

        df_groupby = df.groupby(
            [
                "method",
                "data",
                "n",
                "eps",
            ]
        )

        df_groupby_all_data = df.groupby(
            [
                "method",
                "n",
                "eps",
            ]
        )

        df_grouped = df_groupby.agg(
            **{
                **{
                    "run_time_mean": pd.NamedAgg(column="run_time", aggfunc="mean"),
                    "run_time_std": pd.NamedAgg(column="run_time", aggfunc="std"),
                },
                **{f"{k}_mean": pd.NamedAgg(column=k, aggfunc=agg_mean) for k in agg_columns},
                **{f"{k}_std": pd.NamedAgg(column=k, aggfunc=agg_std) for k in agg_columns},
                **{f"{k}_mean_all": pd.NamedAgg(column=k, aggfunc=agg_mean_all) for k in agg_columns_multi},
                **{f"{k}_std_all": pd.NamedAgg(column=k, aggfunc=agg_std_all) for k in agg_columns_multi},
            }
        )

        df_grouped_all_data = df_groupby_all_data.agg(
            **{
                **{
                    "run_time_mean": pd.NamedAgg(column="run_time", aggfunc="mean"),
                    "run_time_std": pd.NamedAgg(column="run_time", aggfunc="std"),
                },
                **{f"{k}_mean_all": pd.NamedAgg(column=k, aggfunc=agg_mean_all) for k in agg_columns_multi},
                **{f"{k}_std_all": pd.NamedAgg(column=k, aggfunc=agg_std_all) for k in agg_columns_multi},
            }
        )

        # Flatten the multi-index columns
        df_grouped.reset_index(inplace=True)

        # df_grouped_all_data.columns = ["_".join(col).strip() for col in df_grouped_all_data.columns.values]
        df_grouped_all_data.reset_index(inplace=True)
        df_grouped_all_data["data"] = "all"

        # Merge the two dataframes
        df_grouped = pd.concat([df_grouped, df_grouped_all_data], ignore_index=True)

        # Store results
        method_name = df_grouped["method"].iloc[0]
        output_path = output_dir / f"{method_name}.pkl"

        df_grouped.to_pickle(output_path)
