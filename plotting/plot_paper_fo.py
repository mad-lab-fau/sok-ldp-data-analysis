from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plotting.plot_paper import _fig_size, _add_subplot_label, _read_all_pkls
from sok_ldp_analysis.simulation.processing_fo import parse_string_array

non_pure_frequency_oracles = {
    "Murakami2018": "Murakami et al. (2018)",
    "KSubset": "$l$-Subset",
    "Nguyen2016FO": "Nguyen et al. (2016)",
    "RAPPOR": "RAPPOR",
}

pure_frequency_oracles = {
    "DRandomizedResponse": "$k$-RR / DE / O-RR (Kairouz et al. (2016))",
    "Duchi2013": "SUE / Duchi et al. (2013)",
    "OptimizedUnaryEncoding": "Optimized Unary Encoding",
    "OptimizedLocalHashing": "Optimized Local Hashing",
    "HadamardMechanism": "Hadamard Mechanism",
    "HadamardResponse": "Hadamard Response",
}


def plot_fo_mse_3n(df_grouped, output_dir):
    output_dir = output_dir / ("fo_mse_3n")

    method_mapping = {**pure_frequency_oracles, **non_pure_frequency_oracles}
    methods = list(method_mapping.keys())

    n_range = [1000, 10000, 100000]

    error = "mse"

    for std in [True, False]:
        fig = plt.figure(figsize=_fig_size(two_columns=True, height=2))
        gs = fig.add_gridspec(1, 3, hspace=0.1, wspace=0.1)
        gs.update(left=0.08, right=0.98, top=0.85, bottom=0.2, wspace=0.1, hspace=0.1)
        axes = gs.subplots(sharex=True, sharey=True)

        for n, ax in zip(n_range, axes):
            for method in methods:
                df_n = df_grouped[
                    (df_grouped["method"] == method) & (df_grouped["n"] == n) & (df_grouped["data"] == "all")
                ]

                x_key = "eps"

                if len(df_n) == 0:
                    continue

                # sort by n
                df_n = df_n.sort_values(x_key)

                ax.plot(
                    df_n[x_key],
                    df_n[f"{error}_mean_all"],
                    label=f"{method_mapping[method]}",
                    linewidth=0.5,
                    linestyle="--",
                )

                if std:
                    ax.fill_between(
                        df_n[x_key],
                        (df_n[f"{error}_mean_all"] - df_n[f"{error}_std_all"]).clip(0),
                        df_n[f"{error}_mean_all"] + df_n[f"{error}_std_all"],
                        alpha=0.3,
                    )

                # add grid
                ax.grid(which="major", linestyle="--", linewidth=0.5)

                _add_subplot_label(ax, fig, f"$n=\\num{{{n}}}$", 0.0, 0.25)

                ax.set_xlabel("$\\varepsilon$")
                ax.set_ylabel("Mean Squared Error")

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(0.1, 10)
                ax.set_ylim(1e-8, 1e0)

                ax.spines[["right", "top"]].set_visible(False)
                ax.label_outer()

        store_dir = output_dir  # / data
        if not store_dir.exists():
            store_dir.mkdir(parents=True, exist_ok=True)

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(handles, labels, frameon=False, loc="outside upper center", fontsize=6, ncol=5)

        plt.savefig(store_dir / f"fo_mse_3n{'_std' if std else ''}.pdf")
        plt.close()


def plot_fo_mse(df_grouped, output_dir):
    output_dir = output_dir / ("fo_mse")

    # get the unique values for n, eps and method
    n_range = df_grouped["n"].unique()

    method_mapping = {**non_pure_frequency_oracles, **pure_frequency_oracles}
    methods = list(method_mapping.keys())

    error = "mse"

    for n in n_range:
        fig = plt.figure(figsize=_fig_size(two_columns=False, height=2))
        gs = fig.add_gridspec(1, 1, hspace=0, wspace=0)
        ax = gs.subplots(sharex="col", sharey="row")

        for method in methods:
            df_n = df_grouped[(df_grouped["method"] == method) & (df_grouped["n"] == n)]

            for col in [f"{error}_mean", f"{error}_std"]:
                df_n[col] = df_n[col].apply(parse_string_array).apply(lambda x: np.mean(x))

            all_data = (
                df_n.groupby(["method", "eps"])[[f"{error}_mean", f"{error}_std"]].agg(["mean", "std"]).reset_index()
            )

            all_data.columns = ["_".join(col).strip() for col in all_data.columns.values]

            x_key = "eps_"

            if len(all_data) == 0:
                continue

            # sort by n
            all_data = all_data.sort_values(x_key)

            ax.plot(
                all_data[x_key],
                all_data[f"{error}_mean_mean"],
                label=f"{method_mapping[method]}",
                linewidth=0.5,
                linestyle="--",
            )
            ax.fill_between(
                all_data[x_key],
                (all_data[f"{error}_mean_mean"] - all_data[f"{error}_std_mean"]).clip(0),
                all_data[f"{error}_mean_mean"] + all_data[f"{error}_std_mean"],
                alpha=0.3,
            )

        store_dir = output_dir  # / data
        if not store_dir.exists():
            store_dir.mkdir(parents=True, exist_ok=True)

        ax.set_xlabel("$\\varepsilon$")
        ax.set_ylabel("Mean Squared Error")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 10)

        ax.spines[["right", "top"]].set_visible(False)
        ax.label_outer()

        fig.legend(frameon=False, loc="outside lower center", fontsize=5, ncol=5)

        plt.tight_layout()
        plt.savefig(store_dir / f"n{n}.pdf")
        plt.close()


def plot_method_grid(df_grouped, output_dir):
    output_dir = output_dir / "fo_grid"

    datasets = df_grouped["data"].unique().tolist()

    method_mapping = {**non_pure_frequency_oracles, **pure_frequency_oracles}

    method_types = [list(non_pure_frequency_oracles.keys()), list(pure_frequency_oracles.keys())]
    methods = [(i, j, method) for j in range(2) for i, method in enumerate(method_types[j])]

    n_range = [100, 1000, 10000, 100000, 1000000]

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", matplotlib.colormaps["Blues"](np.linspace(0.5, 1, len(n_range)))
    )

    for data in datasets:
        # paper height 8.69in
        fig = plt.figure(figsize=_fig_size(two_columns=True, height=8.2))
        gs = fig.add_gridspec(6, 2, hspace=0.1, wspace=0.1)
        gs.update(left=0.08, right=0.98, top=0.95, bottom=0.08, wspace=0.15, hspace=0.1)
        axes = gs.subplots(sharex=True, sharey=True)

        for i, j, method in methods:
            ax = axes[i, j]

            for n in n_range:
                df_n = df_grouped[
                    (df_grouped["method"] == method) & (df_grouped["n"] == n) & (df_grouped["data"] == data)
                ]

                x_key = "eps"

                if len(df_n) == 0:
                    continue

                # sort by n
                df_n = df_n.sort_values(x_key)

                ax.plot(
                    df_n[x_key],
                    df_n["mse_mean_all"],
                    label=f"n=\\num{{{n}}}",
                    linestyle="--",
                    # marker=".",
                    linewidth=0.5,
                )
                ax.fill_between(
                    df_n[x_key],
                    (df_n["mse_mean_all"] - df_n["mse_std_all"]).clip(0),
                    df_n["mse_mean_all"] + df_n["mse_std_all"],
                    alpha=0.3,
                )

                ax.set_xlabel("$\\varepsilon$")
                ax.set_ylabel("Mean Squared Error")

                # add grid
                ax.grid(which="major", linestyle="--", linewidth=0.5)

                _add_subplot_label(ax, fig, method_mapping[method], 0.0, 0.25)

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(0.1, 10)

                ax.spines[["right", "top"]].set_visible(False)
                ax.label_outer()

                # make sure the y-ticks are labeled
                ax.yaxis.set_tick_params(which="both", labelleft=True)

        axes[0, 0].set_title("Non-Pure Frequency Oracles")
        axes[0, 1].set_title("Pure Frequency Oracles")

        store_dir = output_dir  # / data
        if not store_dir.exists():
            store_dir.mkdir(parents=True, exist_ok=True)

        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=6)

        plt.savefig(store_dir / f"fo_mse_{data}.pdf")


def plot_method_grid_var(df_grouped, output_dir):
    output_dir = output_dir / "fo_grid_var"

    datasets = df_grouped["data"].unique().tolist()

    method_mapping = {**non_pure_frequency_oracles, **pure_frequency_oracles}

    method_types = [list(non_pure_frequency_oracles.keys()), list(pure_frequency_oracles.keys())]
    methods = [(i, j, method) for j in range(2) for i, method in enumerate(method_types[j])]

    n_range = [100, 1000, 10000, 100000, 1000000]

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", matplotlib.colormaps["Blues"](np.linspace(0.5, 1, len(n_range)))
    )

    for data in datasets:
        # paper height 8.69in
        fig = plt.figure(figsize=_fig_size(two_columns=True, height=8.2))
        gs = fig.add_gridspec(6, 2, hspace=0.1, wspace=0.1)
        gs.update(left=0.08, right=0.98, top=0.95, bottom=0.08, wspace=0.15, hspace=0.1)
        axes = gs.subplots(sharex=True, sharey=True)

        for i, j, method in methods:
            ax = axes[i, j]

            for n in n_range:
                df_n = df_grouped[
                    (df_grouped["method"] == method) & (df_grouped["n"] == n) & (df_grouped["data"] == data)
                ]

                x_key = "eps"

                if len(df_n) == 0:
                    continue

                # sort by n
                df_n = df_n.sort_values(x_key)

                ax.plot(
                    df_n[x_key],
                    df_n["est_freq_std_all"] ** 2,
                    label=f"n=\\num{{{n}}}",
                    linestyle="--",
                    linewidth=0.5,
                )

                ax.set_xlabel("$\\varepsilon$")
                ax.set_ylabel("Variance")

                # add grid
                ax.grid(which="major", linestyle="--", linewidth=0.5)

                _add_subplot_label(ax, fig, method_mapping[method], 0.0, 0.25)

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(0.1, 10)

                ax.spines[["right", "top"]].set_visible(False)
                ax.label_outer()

                # make sure the y-ticks are labeled
                ax.yaxis.set_tick_params(which="both", labelleft=True)

        axes[0, 0].set_title("Non-Pure Frequency Oracles")
        axes[0, 1].set_title("Pure Frequency Oracles")

        store_dir = output_dir  # / data
        if not store_dir.exists():
            store_dir.mkdir(parents=True, exist_ok=True)

        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=6)

        plt.savefig(store_dir / f"fo_var_{data}.pdf")
        plt.close(fig)


def plot_domain_size_3n_grid(df_grouped, output_dir):
    output_dir = output_dir / ("fo_mse_domainsize_3n")

    method_mapping = {**pure_frequency_oracles, **non_pure_frequency_oracles}
    methods = list(method_mapping.keys())

    # fix color mapping for methods
    colors = plt.cm.get_cmap("tab10", len(methods))

    n_range = [1000, 10000, 100000]

    error = "mse"

    # Filter datasets, keep only GeometricData1D
    df_grouped = df_grouped[df_grouped["data"].str.startswith("GeometricData1D")]

    # Extract the domain size
    df_grouped["domain_size"] = df_grouped["data"].str.extract(r"GeometricData1D\(\((\d+),.*\)\)").astype(int)
    domain_sizes = np.sort(df_grouped["domain_size"].unique())

    fig = plt.figure(figsize=_fig_size(two_columns=True, height=8))
    gs = fig.add_gridspec(len(domain_sizes), len(n_range), hspace=0.1, wspace=0.1)
    gs.update(left=0.08, right=0.98, top=0.92, bottom=0.08, wspace=0.1, hspace=0.15)
    axes = gs.subplots(sharex=True, sharey="row")

    for col, n in enumerate(n_range):
        axes[0, col].set_title(f"$n=\\num{{{n}}}$")
        for row, domain_size in enumerate(domain_sizes):
            ax = axes[row, col]

            data_missing = False

            for method in methods:
                df_n = df_grouped[
                    (df_grouped["method"] == method)
                    & (df_grouped["n"] == n)
                    & (df_grouped["domain_size"] == domain_size)
                ]

                x_key = "eps"

                if len(df_n) == 0:
                    print(f"Skipping {method} {n} {domain_size}")
                    data_missing = True
                    continue

                # sort by n
                df_n = df_n.sort_values(x_key)

                ax.plot(
                    df_n[x_key],
                    df_n[f"{error}_mean_all"],
                    label=f"{method_mapping[method]}",
                    linewidth=0.5,
                    linestyle="--",
                    color=colors(methods.index(method)),
                )

            # add grid
            ax.grid(which="major", linestyle="--", linewidth=0.5)

            _add_subplot_label(ax, fig, f"$k=\\num{{{domain_size}}}${'*' if data_missing else ''}", 0.0, 0.35)

            ax.set_xlabel("$\\varepsilon$")
            ax.set_ylabel("MSE")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.1, 10)

            ax.spines[["right", "top"]].set_visible(False)
            ax.label_outer()

    store_dir = output_dir  # / data
    if not store_dir.exists():
        store_dir.mkdir(parents=True, exist_ok=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(handles, labels, frameon=False, loc="outside upper center", fontsize=6, ncol=5)

    plt.savefig(store_dir / f"fo_mse_domainsize_grid_3n.pdf")
    plt.close()


if __name__ == "__main__":
    df_grouped = _read_all_pkls("fo")

    output_dir = Path("plots_paper")
    plot_fo_mse_3n(df_grouped, output_dir)

    plot_domain_size_3n_grid(df_grouped, output_dir)

    plot_method_grid(df_grouped, output_dir)
    plot_method_grid_var(df_grouped, output_dir)
