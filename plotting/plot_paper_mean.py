from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter

from plotting.plot_paper import (
    _fig_size,
    _read_all_csvs,
    method_mapping_mean,
    _add_subplot_label,
    method_mapping_mean_short,
)


def plot_mean_method_diff(df_grouped, df_rr, output_dir):
    output_dir = output_dir / ("mean_method_diff")

    methods = method_mapping_mean_short.keys()

    n = 10000

    for std in [True, False]:
        for error in ["scaled_abs_error", "scaled_est_mean_std", "scaled_mse"]:
            # for data in datasets:
            fig = plt.figure(figsize=_fig_size(two_columns=False, height=2))
            gs = fig.add_gridspec(1, 1, left=0.15, right=0.95, top=0.85, bottom=0.2, wspace=0.3, hspace=0.2)
            ax = gs.subplots(sharex="col", sharey="row")

            for method in methods:
                df_n = df_grouped[
                    (df_grouped["method"] == method) & (df_grouped["n"] == n) & (df_grouped["data"] == "all")
                ]

                all_data = df_n.sort_values("eps")

                if not error.endswith("std"):
                    ax.plot(all_data["eps"], all_data[f"{error}_mean"], label=method_mapping_mean_short[method])
                else:
                    ax.plot(all_data["eps"], all_data[f"{error}"], label=method_mapping_mean_short[method])

                if not error.endswith("std") and std:
                    ax.fill_between(
                        all_data["eps"],
                        all_data[f"{error}_mean"] - all_data[f"{error}_std"],
                        all_data[f"{error}_mean"] + all_data[f"{error}_std"],
                        alpha=0.3,
                    )

            if not error.endswith("std"):
                rr_value = df_rr[(df_rr["data"] == "all") & (df_rr["n"] == n)][f"{error}_mean"]
                ax.hlines(
                    y=rr_value,
                    xmin=0.1,
                    xmax=10,
                    linestyles="dotted",
                    color="black",
                    label="Random Rounding",
                )

            store_dir = output_dir  # / data
            if not store_dir.exists():
                store_dir.mkdir(parents=True, exist_ok=True)

            ax.set_xlabel("$\\varepsilon$")
            if error == "scaled_abs_error":
                ax.set_ylabel("Range-Scaled MAE")
            elif error == "scaled_est_mean_std":
                ax.set_ylabel("Variance of the Mean Estimate")
            elif error == "scaled_mse":
                ax.set_ylabel("Range-Scaled MSE")

            ax.grid(which="major", linestyle="--", linewidth=0.5)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.1, 10)

            ax.spines[["right", "top"]].set_visible(False)
            ax.label_outer()

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, frameon=False, loc="outside upper center", ncol=3, fontsize=6)

            plt.savefig(store_dir / f"n{n}_{error}{'_std' if std else ''}.pdf")
            plt.close()


def plot_mean_method_diff_priv(df_grouped, df_rr, output_dir):
    output_dir = output_dir / ("mean_method_diff_priv")

    methods = method_mapping_mean.keys()

    n = 10000

    for error in ["scaled_abs_error", "scaled_est_mean_std", "scaled_mse"]:
        # for data in datasets:
        fig = plt.figure(figsize=_fig_size(two_columns=False, height=2))
        gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.1)
        gs.update(left=0.15, right=0.98, top=0.85, bottom=0.2, wspace=0.3, hspace=0.2)
        axes = gs.subplots(sharex=False, sharey=False)

        for method in methods:
            df_n = df_grouped[(df_grouped["method"] == method) & (df_grouped["n"] == n) & (df_grouped["data"] == "all")]

            all_data = df_n.sort_values("eps")

            for ax in axes:
                if not error.endswith("std"):
                    ax.plot(all_data["eps"], all_data[f"{error}_mean"], label=method_mapping_mean[method])
                else:
                    ax.plot(all_data["eps"], all_data[f"{error}"], label=method_mapping_mean[method])

                if not error.endswith("std"):
                    ax.fill_between(
                        all_data["eps"],
                        all_data[f"{error}_mean"] - all_data[f"{error}_std"],
                        all_data[f"{error}_mean"] + all_data[f"{error}_std"],
                        alpha=0.3,
                    )

        if not error.endswith("std"):
            rr_value = df_rr[(df_rr["data"] == "all") & (df_rr["n"] == n)][f"{error}_mean"]
            for ax in axes:
                ax.hlines(
                    y=rr_value,
                    xmin=0.1,
                    xmax=10,
                    linestyles="dotted",
                    color="black",
                    label="Random Rounding",
                )

        if error == "scaled_abs_error":
            axes[0].set_ylabel("Range-Scaled MAE")
        elif error == "scaled_est_mean_std":
            axes[0].set_ylabel("Variance of the Mean Estimate")
        elif error == "scaled_mse":
            axes[0].set_ylabel("Range-Scaled MSE")

        for ax in axes:
            ax.set_xlabel("$\\varepsilon$")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(which="major", linestyle="--", linewidth=0.5)

            ax.spines[["right", "top"]].set_visible(False)

        axes[0].set_xlim(0.1, 1)
        axes[0].xaxis.set_minor_formatter(NullFormatter())

        axes[1].set_xlim(1, 10)
        axes[1].xaxis.set_minor_formatter(NullFormatter())

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="outside upper center", ncol=3, fontsize=5)

        store_dir = output_dir
        if not store_dir.exists():
            store_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(store_dir / f"n{n}_{error}.pdf")
        plt.close()


def plot_grid_n(df_grouped, df_rr, output_dir):
    output_dir = output_dir / "mean_grid"

    data_mapping = {
        "UniformSmallData1D": "Uniform Small",
        "UniformLargeData1D": "Uniform Large",
        "BimodalData1D(1, 0)": "Bimodal",
        "BinomialData1D(100, 0.2)": "Binomial",
        "Adult": "Adult",
        "NYCTaxiData": "NYC Taxi",
    }

    datasets = list(data_mapping.keys())
    n_range = [1000, 10000, 100000]

    for error in ["scaled_abs_error", "scaled_mse", "resp_var"]:
        mapping = method_mapping_mean_short

        if error == "resp_var":
            mapping["Ding2017LargeM"] = "Ding"

        methods = mapping.keys()

        fig = plt.figure(figsize=_fig_size(two_columns=True, height=6.5))
        gs = fig.add_gridspec(
            len(datasets), len(n_range), left=0.08, right=0.98, top=0.92, bottom=0.08, wspace=0.17, hspace=0.1
        )
        axes = gs.subplots(sharex=True, sharey=True)

        for col, n in enumerate(n_range):
            for row, data in enumerate(datasets):
                ax = axes[row, col]

                for method in methods:
                    df_n = df_grouped[
                        (df_grouped["method"] == method) & (df_grouped["n"] == n) & (df_grouped["data"] == data)
                    ]

                    all_data = df_n.sort_values("eps")

                    ax.plot(all_data["eps"], all_data[f"{error}_mean"], label=method_mapping_mean_short[method])
                    ax.fill_between(
                        all_data["eps"],
                        all_data[f"{error}_mean"] - all_data[f"{error}_std"],
                        all_data[f"{error}_mean"] + all_data[f"{error}_std"],
                        alpha=0.3,
                    )

                if error != "resp_var":
                    rr_value = df_rr[(df_rr["data"] == "all") & (df_rr["n"] == n)][f"{error}_mean"]
                    ax.hlines(
                        y=rr_value,
                        xmin=0.1,
                        xmax=10,
                        linestyles="dotted",
                        color="black",
                        label="Random Rounding",
                    )

                ax.set_yscale("log")
                ax.set_xscale("log")
                ax.set_xlim(0.1, 10)

                ax.set_xlabel("$\\varepsilon$")

                if error == "scaled_abs_error":
                    ax.set_ylabel("Scaled MAE")
                elif error == "scaled_mse":
                    ax.set_ylabel("Scaled MSE")
                elif error == "resp_var":
                    ax.set_ylabel("Response Variance")

                # add grid
                ax.grid(which="major", linestyle="--", linewidth=0.5)

                _add_subplot_label(ax, fig, f"{data_mapping[data]}", 0.0, 0.3)

                ax.spines[["right", "top"]].set_visible(False)
                ax.label_outer()

                # make sure the y-ticks are labeled
                ax.yaxis.set_tick_params(which="both", labelleft=True)

            axes[0, col].set_title(f"$n=\\num{{{n}}}$")

        store_dir = output_dir  # / data
        if not store_dir.exists():
            store_dir.mkdir(parents=True, exist_ok=True)

        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(handles, labels, frameon=False, loc="outside upper center", ncol=6)

        plt.savefig(store_dir / f"mean_{error}.pdf")


if __name__ == "__main__":
    df_grouped = _read_all_csvs("mean")

    # Read random rounding results
    df_rr = pd.read_pickle("../experiments/results_grouped/simulation/mean_rr/rr_results.pkl")

    output_dir = Path("plots_paper")
    plot_mean_method_diff(df_grouped, df_rr, output_dir)
    plot_mean_method_diff_priv(df_grouped, df_rr, output_dir)
    plot_grid_n(df_grouped, df_rr, output_dir)
