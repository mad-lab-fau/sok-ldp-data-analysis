from pathlib import Path

import matplotlib.pyplot as plt

from plotting.plot_paper import _fig_size, _read_all_csvs, _add_subplot_label

var_method_mapping = {
    "MeanAndVarianceSplitEps": "Split by $\\varepsilon$",
    "MeanAndVarianceSplitN": "Split by $n$",
    "MeanAndVarianceSplitNSequential": "Sequential Split",
}


def plot_mean_var(df_grouped, output_dir):
    output_dir = output_dir / ("variance_method_diff")

    fig = plt.figure(figsize=_fig_size(two_columns=False, height=1.8))
    gs = fig.add_gridspec(1, 2, left=0.18, right=0.95, top=0.9, bottom=0.2, wspace=0.3, hspace=0.1)
    ax_mean, ax_var = gs.subplots(sharex=False, sharey=False)

    df_grouped = df_grouped[df_grouped["base_method"] == "Wang2019Piecewise1D"]

    n = 10000

    eps = 2.0

    ax_mean.set_ylabel("Range-Scaled MAE")

    var_methods = list(var_method_mapping.keys())

    ax_mean.set_title("Mean")
    ax_var.set_title("Variance")

    ax_mean.set_ylim(1e-3, 1e-1)
    ax_var.set_ylim(1e-1, 1e1)

    for ax, error in [(ax_mean, "mean_scaled_abs_error"), (ax_var, "var_scaled_abs_error")]:
        ax.set_xlabel("Split Ratio")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 0.9)

        # set ticks
        ax.set_xticks([0.1, 0.5, 0.9])

        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(which="major", linestyle="--", linewidth=0.5)

        for method in var_methods:
            df_n = df_grouped[
                (df_grouped["var_method"] == method)
                & (df_grouped["n"] == n)
                & (df_grouped["eps"] == eps)
                & (df_grouped["data"] == "all")
            ]

            # sort by n
            all_data = df_n.sort_values("split_ratio")

            ax.plot(all_data["split_ratio"], all_data[f"{error}_mean"], label=f"{var_method_mapping[method]}")
            ax.fill_between(
                all_data["split_ratio"],
                all_data[f"{error}_mean"] - all_data[f"{error}_std"],
                all_data[f"{error}_mean"] + all_data[f"{error}_std"],
                alpha=0.3,
            )

    store_dir = output_dir
    if not store_dir.exists():
        store_dir.mkdir(parents=True, exist_ok=True)

    handles, labels = ax_mean.get_legend_handles_labels()
    ax_mean.legend(handles, labels, frameon=False, loc="upper right", fontsize=5)

    plt.savefig(store_dir / f"n{n}.pdf")
    plt.close()


def plot_mean_var_grid_ratio(df_grouped, output_dir):
    output_dir = output_dir / ("variance_method_diff")

    n_range = [1000, 10000, 100000]
    eps_range = [0.1, 1.0, 10.0]

    fig = plt.figure(figsize=_fig_size(two_columns=True, height=6.5))
    gs = fig.add_gridspec(
        len(eps_range), len(n_range), left=0.18, right=0.95, top=0.9, bottom=0.2, wspace=0.3, hspace=0.1
    )
    axes = gs.subplots(sharex=True, sharey="row")

    df_grouped = df_grouped[df_grouped["base_method"] == "Wang2019Piecewise1D"]
    var_methods = list(var_method_mapping.keys())

    for i, n in enumerate(n_range):
        for j, eps in enumerate(eps_range):
            ax = axes[j, i]

            ax.set_ylabel("Ratio: MAE Mean / MAE Variance")
            ax.set_xlabel("Split Ratio")
            # ax.set_title(f"$n={n}$, $\\varepsilon={eps}$")

            ax.grid(which="major", linestyle="--", linewidth=0.5)

            _add_subplot_label(ax, fig, f"$n={n}$, $\\varepsilon={eps}$", 0.6, 0.9)

            ax.spines[["right", "top"]].set_visible(False)
            ax.label_outer()

            # make sure the y-ticks are labeled
            ax.yaxis.set_tick_params(which="both", labelleft=True)

            for method in var_methods:
                df_n = df_grouped[
                    (df_grouped["var_method"] == method)
                    & (df_grouped["n"] == n)
                    & (df_grouped["eps"] == eps)
                    & (df_grouped["data"] == "all")
                ]

                # sort by n
                all_data = df_n.sort_values("split_ratio")

                ratio = all_data["mean_scaled_abs_error_mean"] / all_data["var_scaled_abs_error_mean"]
                ax.plot(all_data["split_ratio"], ratio, label=f"{var_method_mapping[method]}")

    store_dir = output_dir
    if not store_dir.exists():
        store_dir.mkdir(parents=True, exist_ok=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", fontsize=5)

    plt.savefig(store_dir / f"grid_ratio.pdf")


def plot_mean_var_grid(df_grouped, output_dir):
    output_dir = output_dir / ("variance_method_diff")

    n_ranges = [[1000, 10000], [100000, 1000000]]

    # Get unique epsilons from df_grouped and select 4 equally spaced epsilons
    eps_range = df_grouped["eps"].unique()
    eps_range = eps_range[:: len(eps_range) // 4]

    for fig_id, n_range in enumerate(n_ranges):
        fig = plt.figure(figsize=_fig_size(two_columns=True, height=8), layout="constrained")
        subfigs = fig.subfigures(len(eps_range), len(n_range))

        df_grouped = df_grouped[df_grouped["base_method"] == "Wang2019Piecewise1D"]
        var_methods = list(var_method_mapping.keys())

        for i, n in enumerate(n_range):
            for j, eps in enumerate(eps_range):
                subfig = subfigs[j, i]

                axes = subfig.subplots(1, 2)

                ax_mean = axes[0]
                ax_var = axes[1]

                ax_mean.set_ylabel("Range-Scaled MAE")
                ax_var.set_ylabel("Range-Scaled MAE")
                ax_mean.set_xlabel("Split Ratio")
                ax_var.set_xlabel("Split Ratio")

                ax_mean.set_title("Mean")
                ax_var.set_title("Variance")

                ax_mean.set_yscale("log")
                ax_var.set_yscale("log")

                ax_mean.grid(which="major", linestyle="--", linewidth=0.5)
                ax_var.grid(which="major", linestyle="--", linewidth=0.5)

                subfig.suptitle(f"$n=\\num{{{n}}}$, $\\varepsilon={eps}$")

                ax_mean.spines[["right", "top"]].set_visible(False)
                ax_var.spines[["right", "top"]].set_visible(False)

                for method in var_methods:
                    df_n = df_grouped[
                        (df_grouped["var_method"] == method)
                        & (df_grouped["n"] == n)
                        & (df_grouped["eps"] == eps)
                        & (df_grouped["data"] == "all")
                    ]

                    # sort by n
                    all_data = df_n.sort_values("split_ratio")

                    for ax, error in [(ax_mean, "mean_scaled_abs_error"), (ax_var, "var_scaled_abs_error")]:
                        ax.plot(
                            all_data["split_ratio"], all_data[f"{error}_mean"], label=f"{var_method_mapping[method]}"
                        )
                        ax.fill_between(
                            all_data["split_ratio"],
                            all_data[f"{error}_mean"] - all_data[f"{error}_std"],
                            all_data[f"{error}_mean"] + all_data[f"{error}_std"],
                            alpha=0.3,
                        )

        store_dir = output_dir
        if not store_dir.exists():
            store_dir.mkdir(parents=True, exist_ok=True)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="upper right", fontsize=5)

        plt.savefig(store_dir / f"grid_{fig_id}.pdf")


if __name__ == "__main__":
    df_grouped = _read_all_csvs("variance")

    output_dir = Path("plots_paper")
    plot_mean_var(df_grouped, output_dir)
    plot_mean_var_grid_ratio(df_grouped, output_dir)
    plot_mean_var_grid(df_grouped, output_dir)
