import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from sok_ldp_analysis.simulation.processing_fo import agg_mean, agg_mean_all, agg_std, agg_std_all

data_ranges = {
    "BimodalData1D(1, 0)": (0, 1),
    "BinomialData1D(100, 0.2)": (0, 100),
    "UniformLargeData1D": (-100, 100),
    "UniformSmallData1D": (0, 1),
    "MixedDataMultiDim": (-100, 100),
    "Adult": (16, 100),
    "NYCTaxiData": (0, 24 * 60 * 60),
}


def _calculate_errors(df: pd.DataFrame, true_col="sample_mean", est_col="est_mean", prefix=""):
    if len(prefix) > 0 and not prefix.endswith("_"):
        prefix = prefix + "_"

    df["data"] = df["data"].apply(lambda x: x[18:].split(")")[0] + ")" if x.startswith("MultiDatasetSynth") else x)

    # add data range to df
    df["data_range_size"] = df["data"].apply(lambda x: data_ranges[x][1] - data_ranges[x][0])

    df[prefix + "error"] = df[est_col] - df[true_col]
    df[prefix + "abs_error"] = df[prefix + "error"].abs()
    df[prefix + "rel_error"] = df[prefix + "error"] / df[true_col]
    df[prefix + "abs_rel_error"] = df[prefix + "rel_error"].abs()
    df[prefix + "mse"] = df[prefix + "error"] ** 2

    df[prefix + "scaled_abs_error"] = df[prefix + "abs_error"] / df["data_range_size"]
    df[prefix + "scaled_mse"] = (df[prefix + "error"] / df["data_range_size"]) ** 2

    # scale the estimate by the data range size
    df["scaled_" + est_col] = df[est_col] / df["data_range_size"]

    errors = ["error", "abs_error", "rel_error", "abs_rel_error", "mse", "scaled_abs_error", "scaled_mse"]

    errors = [prefix + e for e in errors] + ["scaled_" + est_col, "data_range_size"]

    return df, errors


def _calculate_errors_gaussian(df: pd.DataFrame):
    df["error"] = df["est_mean"] - df["pop_mean"]
    df["sample_error"] = df["est_mean"] - df["sample_mean"]
    df["abs_error"] = df["error"].abs()
    df["sample_abs_error"] = df["sample_error"].abs()
    df["rel_error"] = df["error"] / df["pop_mean"]
    df["sample_rel_error"] = df["sample_error"] / df["sample_mean"]
    df["abs_rel_error"] = df["rel_error"].abs()
    df["sample_abs_rel_error"] = df["sample_rel_error"].abs()
    df["mse"] = df["error"] ** 2
    df["sample_mse"] = df["sample_error"] ** 2

    errors = [
        "error",
        "sample_error",
        "abs_error",
        "sample_abs_error",
        "rel_error",
        "sample_rel_error",
        "abs_rel_error",
        "sample_abs_rel_error",
        "mse",
        "sample_mse",
    ]

    return df, errors


def group_results_mean(df: pd.DataFrame, output_dir: Path):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Multi-dimensional methods output a size 1 array for the estimated mean (sometimes stored as a string)
    if df["est_mean"].dtypes == "object" and type(df["est_mean"].iloc[0]) is not np.ndarray:
        df["est_mean"] = df["est_mean"].str[1:-1].astype(float)

    # If the estimated mean is a numpy array, take the first (and only) element
    if df["est_mean"].dtypes == "object" and type(df["est_mean"].iloc[0]) is np.ndarray:
        df["est_mean"] = df["est_mean"].apply(lambda x: x[0])

    # calculate errors
    df, agg_columns = _calculate_errors(df)

    # Calculate the mean and standard deviation of the run time and the errors
    agg_columns = ["run_time", "est_mean", "sample_mean", "sample_sigma", "resp_var"] + agg_columns
    df_grouped = df.groupby(
        [
            "method",
            "data",
            "n",
            "eps",
        ]
    ).agg({k: ["mean", "std"] for k in agg_columns})

    df_grouped_all_data = df.groupby(
        [
            "method",
            "n",
            "eps",
        ]
    ).agg({k: ["mean", "std"] for k in agg_columns})

    # Flatten the multi-index columns
    df_grouped.columns = ["_".join(col).strip() for col in df_grouped.columns.values]
    df_grouped.reset_index(inplace=True)

    df_grouped_all_data.columns = ["_".join(col).strip() for col in df_grouped_all_data.columns.values]
    df_grouped_all_data.reset_index(inplace=True)
    df_grouped_all_data["data"] = "all"

    # Merge the two dataframes
    df_grouped = pd.concat([df_grouped, df_grouped_all_data], ignore_index=True)

    # Store results
    method_name = df_grouped["method"].iloc[0]
    output_path = output_dir / f"{method_name}.csv"

    df_grouped.to_csv(output_path, index=False, mode="a", header=not output_path.exists())


def group_results_variance(df: pd.DataFrame, output_dir: Path):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Multi-dimensional methods output a size 1 array for the estimated mean (sometimes stored as a string)
    for col in ["est_mean", "est_var"]:
        if df[col].dtypes == "object" and type(df[col].iloc[0]) is not np.ndarray:
            df[col] = df[col].str[1:-1].astype(float)

        # If the estimated mean is a numpy array, take the first (and only) element
        if df[col].dtypes == "object" and type(df[col].iloc[0]) is np.ndarray:
            df[col] = df[col].apply(lambda x: x[0])

    # calculate errors
    df, agg_columns1 = _calculate_errors(df, "sample_mean", "est_mean", "mean")
    df, agg_columns2 = _calculate_errors(df, "sample_var", "est_var", "var")

    df["scaled_abs_error_ratio"] = df["var_scaled_abs_error"] / df["mean_scaled_abs_error"]
    df["scaled_mse_ratio"] = df["var_scaled_mse"] / df["mean_scaled_mse"]

    # Calculate the mean and standard deviation of the run time and the errors
    agg_columns = (
        ["run_time", "est_mean", "est_var", "sample_mean", "sample_var", "scaled_abs_error_ratio", "scaled_mse_ratio"]
        + agg_columns1
        + agg_columns2
    )
    df_grouped = df.groupby(
        [
            "var_method",
            "base_method",
            "data",
            "n",
            "eps",
            "split_ratio",
        ]
    ).agg({k: ["mean", "std"] for k in agg_columns})

    df_grouped_all_data = df.groupby(
        [
            "var_method",
            "base_method",
            "n",
            "eps",
            "split_ratio",
        ]
    ).agg({k: ["mean", "std"] for k in agg_columns})

    # Flatten the multi-index columns
    df_grouped.columns = ["_".join(col).strip() for col in df_grouped.columns.values]
    df_grouped.reset_index(inplace=True)

    df_grouped_all_data.columns = ["_".join(col).strip() for col in df_grouped_all_data.columns.values]
    df_grouped_all_data.reset_index(inplace=True)
    df_grouped_all_data["data"] = "all"

    # Merge the two dataframes
    df_grouped = pd.concat([df_grouped, df_grouped_all_data], ignore_index=True)

    # Store results
    method_name = df_grouped["var_method"].iloc[0]
    output_path = output_dir / f"{method_name}.csv"

    df_grouped.to_csv(output_path, index=False, mode="a", header=not output_path.exists())


def group_results_mean_multi_dim(df: pd.DataFrame, output_dir: Path):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # calculate errors
    df, agg_columns = _calculate_errors(df)

    agg_columns_multi = []
    for col in agg_columns:
        if type(df[col].iloc[0]) is np.ndarray:
            agg_columns_multi.append(col)

    df_groupby = df.groupby(
        [
            "method",
            "data",
            "n",
            "eps",
            "d",
        ]
    )

    df_groupby_all_data = df.groupby(
        [
            "method",
            "n",
            "eps",
            "d",
        ]
    )

    # Calculate the mean and standard deviation of the run time and the errors
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

    method_name = df_grouped["method"].iloc[0]

    # Store results
    output_path = output_dir / f"{method_name}.csv"

    df_grouped.to_csv(output_path, index=False, mode="a", header=not output_path.exists())


def group_results_gaussian_mean(df: pd.DataFrame, output_dir: Path):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # calculate errors
    df, agg_columns = _calculate_errors_gaussian(df)

    # Calculate the mean and standard deviation of the run time and the errors
    agg_columns = ["run_time", "est_mean", "sample_mean", "sample_sigma"] + agg_columns
    df_grouped = df.groupby(
        [
            "method",
            "data",
            "n",
            "eps",
            "delta",
            "beta",
        ]
    ).agg({k: ["mean", "std", "min", "max"] for k in agg_columns})

    # Flatten the multi-index columns
    df_grouped.columns = ["_".join(col).strip() for col in df_grouped.columns.values]
    df_grouped.reset_index(inplace=True)

    # Store results
    method_name = df_grouped["method"].iloc[0]
    output_path = output_dir / f"{method_name}.csv"

    df_grouped.to_csv(output_path, index=False, mode="a", header=not output_path.exists())


def _read_pickle(filename):
    df = pd.read_pickle(filename)

    if type(df) is list:
        df = pd.DataFrame(df, columns=df[0].keys())

    return df


def group_results(method_type):
    output_dir = Path("results_grouped") / "simulation" / method_type

    # delete output dir
    shutil.rmtree(output_dir, ignore_errors=True)

    # Input path
    path = Path("results/simulation") / method_type

    # Check if the input path exists
    if not path.exists():
        print(f"Path {path} does not exist. Skipping.")
        return

    # Iterate over all directories
    directories = [x for x in path.iterdir() if x.is_dir()]

    if method_type == "mean_multi_dim":
        group_method = group_results_mean_multi_dim
    elif method_type == "mean_gaussian":
        group_method = group_results_gaussian_mean
    elif method_type == "mean":
        group_method = group_results_mean
    elif method_type == "variance":
        group_method = group_results_variance
    else:
        raise ValueError(f"Unknown method type: {method_type}")

    for method_dir in directories:
        # join all .pkl files into one dataframe
        files = list(method_dir.glob("**/*.pkl"))

        df = pd.concat(map(_read_pickle, files))
        group_method(df, output_dir)
