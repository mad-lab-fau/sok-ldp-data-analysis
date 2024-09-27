from pathlib import Path

from matplotlib import pyplot as plt

from plotting.plot_paper import _read_all_csvs, _fig_size, _add_subplot_label

method_mapping_mean_multi = {
    "Wang2019SplitEps": "Wang - split $\\varepsilon$",
    "Wang2019SplitN": "Wang - split $n$",
    "Wang2019K1": "Wang - $k=1$",
    "Wang2019": "Wang - optimal $k$",
    "Duchi2018": "Duchi - $\\ell_2$",
    "Duchi2018LInf": "Duchi - $\\ell_{\\infty}$",
}


def plot_grid_n_method(df_grouped, output_dir):
    output_dir = output_dir / "mean_dim_grid_n_method"

    methods = method_mapping_mean_multi.keys()
    n_range = [1000, 10000, 100000]
    d_range = [1, 2, 4, 8, 16, 32, 64]
    errors = ["scaled_abs_error"]

    data = "all"

    for error in errors:
        fig = plt.figure(figsize=_fig_size(two_columns=True, height=6.5))
        gs = fig.add_gridspec(6, 3, left=0.08, right=0.98, top=0.95, bottom=0.1, wspace=0.15, hspace=0.1)
        axes = gs.subplots(sharex=True, sharey=True)

        for row, method in enumerate(methods):
            if error == "scaled_abs_error":
                axes[row, 0].set_ylabel("Range-Scaled MAE")
            elif error == "scaled_mse":
                axes[row, 0].set_ylabel("Range-Scaled MSE")

            for ax in axes[row, :]:
                _add_subplot_label(ax, fig, f"{method_mapping_mean_multi[method]}", 0.0, 0.3)

        for col, n in enumerate(n_range):
            axes[0, col].set_title(f"$n=\\num{{{n}}}$")

        for col, n in enumerate(n_range):
            for row, method in enumerate(methods):
                ax = axes[row, col]

                for d in d_range:
                    df_n = df_grouped[
                        (df_grouped["method"] == method)
                        & (df_grouped["n"] == n)
                        & (df_grouped["data"] == data)
                        & (df_grouped["d"] == d)
                    ]

                    # sort by n
                    df_n = df_n.sort_values("eps")

                    ax.plot(df_n["eps"], df_n[f"{error}_mean_all"], label=f"$d={d}$")
                    ax.fill_between(
                        df_n["eps"],
                        df_n[f"{error}_mean_all"] - df_n[f"{error}_std_all"],
                        df_n[f"{error}_mean_all"] + df_n[f"{error}_std_all"],
                        alpha=0.3,
                    )

        for ax in axes.flatten():
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlim(0.1, 10)
            ax.set_ylim(1e-4, 10)
            ax.set_xlabel("$\\varepsilon$")
            ax.grid(which="major", linestyle="--", linewidth=0.5)

            # make sure the y-ticks are labeled
            ax.yaxis.set_tick_params(which="both", labelleft=True)

            ax.spines[["right", "top"]].set_visible(False)
            ax.label_outer()

        store_dir = output_dir  # / data
        if not store_dir.exists():
            store_dir.mkdir(parents=True, exist_ok=True)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=len(d_range))

        plt.savefig(store_dir / f"mean_multi_{error}.pdf")


def plot_method_comparison(df_grouped, output_dir):
    output_dir = output_dir / "mean_dim_method_comparison"
    methods = method_mapping_mean_multi.keys()

    errors = ["scaled_abs_error"]
    data = "all"

    n = 10000
    d = 64

    for error in errors:
        fig = plt.figure(figsize=_fig_size(two_columns=False, height=2))
        gs = fig.add_gridspec(1, 1, hspace=0.1, wspace=0.1)
        gs.update(left=0.15, right=0.95, top=0.8, bottom=0.2, wspace=0.15, hspace=0.1)
        ax = gs.subplots(sharex=True, sharey=True)

        for method in methods:
            df_n = df_grouped[
                (df_grouped["method"] == method)
                & (df_grouped["n"] == n)
                & (df_grouped["data"] == data)
                & (df_grouped["d"] == d)
            ]

            # sort by n
            df_n = df_n.sort_values("eps")

            ax.plot(df_n["eps"], df_n[f"{error}_mean_all"], label=f"{method_mapping_mean_multi[method]}")
            ax.fill_between(
                df_n["eps"],
                df_n[f"{error}_mean_all"] - df_n[f"{error}_std_all"],
                df_n[f"{error}_mean_all"] + df_n[f"{error}_std_all"],
                alpha=0.3,
            )

        _add_subplot_label(ax, fig, f"$d={d}, n=\\num{{{n}}}$", 0.0, 0.2)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(0.1, 10)
        ax.set_ylim(1e-3, 10)
        ax.set_xlabel("$\\varepsilon$")
        ax.set_ylabel("Range-Scaled MAE")
        ax.grid(which="major", linestyle="--", linewidth=0.5)

        # make sure the y-ticks are labeled
        ax.yaxis.set_tick_params(which="both", labelleft=True)

        ax.spines[["right", "top"]].set_visible(False)
        ax.label_outer()

        store_dir = output_dir
        if not store_dir.exists():
            store_dir.mkdir(parents=True, exist_ok=True)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="outside upper center", ncol=len(methods) // 2, fontsize=6)

        plt.savefig(store_dir / f"mean_multi_{error}.pdf")


def plot_mean_dim_3n(df_grouped, output_dir):
    output_dir = output_dir / "mean_dim_3n"
    methods = method_mapping_mean_multi.keys()

    errors = ["scaled_abs_error"]
    data = "all"

    n_range = [100, 10000, 100000]
    d = 32

    for error in errors:
        fig = plt.figure(figsize=_fig_size(two_columns=True, height=2))
        gs = fig.add_gridspec(1, 3, hspace=0.1, wspace=0.1)
        gs.update(left=0.08, right=0.98, top=0.85, bottom=0.2, wspace=0.1, hspace=0.1)
        axes = gs.subplots(sharex=True, sharey=True)

        for n, ax in zip(n_range, axes):
            for method in methods:
                df_n = df_grouped[
                    (df_grouped["method"] == method)
                    & (df_grouped["n"] == n)
                    & (df_grouped["data"] == data)
                    & (df_grouped["d"] == d)
                ]

                # sort by n
                df_n = df_n.sort_values("eps")

                ax.plot(df_n["eps"], df_n[f"{error}_mean_all"], label=f"{method_mapping_mean_multi[method]}")
                ax.fill_between(
                    df_n["eps"],
                    df_n[f"{error}_mean_all"] - df_n[f"{error}_std_all"],
                    df_n[f"{error}_mean_all"] + df_n[f"{error}_std_all"],
                    alpha=0.3,
                )

            _add_subplot_label(ax, fig, f"$d={d}, n=\\num{{{n}}}$", 0.0, 0.2)

            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlim(0.1, 10)
            ax.set_ylim(1e-3, 10)
            ax.set_xlabel("$\\varepsilon$")
            ax.set_ylabel("Range-Scaled MAE")
            ax.grid(which="major", linestyle="--", linewidth=0.5)

            # make sure the y-ticks are labeled
            ax.yaxis.set_tick_params(which="both", labelleft=True)

            ax.spines[["right", "top"]].set_visible(False)
            ax.label_outer()

        store_dir = output_dir  # / data
        if not store_dir.exists():
            store_dir.mkdir(parents=True, exist_ok=True)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="outside upper center", ncol=len(methods), fontsize=6)

        plt.savefig(store_dir / f"mean_multi_3n_{error}.pdf")


def plot_mean_dim_3d(df_grouped, output_dir):
    output_dir = output_dir / "mean_dim_grid"
    methods = method_mapping_mean_multi.keys()

    errors = ["scaled_abs_error"]
    data = "all"

    n_range = [1000, 10000, 100000]
    d_range = [2, 4, 8, 16]

    for error in errors:
        fig = plt.figure(figsize=_fig_size(two_columns=True, height=6.5))
        gs = fig.add_gridspec(
            len(d_range), len(n_range), left=0.08, right=0.98, top=0.95, bottom=0.1, wspace=0.15, hspace=0.1
        )
        axes = gs.subplots(sharex=True, sharey=True)

        for i, d in enumerate(d_range):
            for j, n in enumerate(n_range):
                ax = axes[i, j]
                for method in methods:
                    df_n = df_grouped[
                        (df_grouped["method"] == method)
                        & (df_grouped["n"] == n)
                        & (df_grouped["data"] == data)
                        & (df_grouped["d"] == d)
                    ]

                    # sort by n
                    df_n = df_n.sort_values("eps")

                    ax.plot(df_n["eps"], df_n[f"{error}_mean_all"], label=f"{method_mapping_mean_multi[method]}")
                    ax.fill_between(
                        df_n["eps"],
                        df_n[f"{error}_mean_all"] - df_n[f"{error}_std_all"],
                        df_n[f"{error}_mean_all"] + df_n[f"{error}_std_all"],
                        alpha=0.3,
                    )

                _add_subplot_label(ax, fig, f"$d={d}, n=\\num{{{n}}}$", 0.0, 0.2)

                ax.set_yscale("log")
                ax.set_xscale("log")
                ax.set_xlim(0.1, 10)
                ax.set_xlabel("$\\varepsilon$")
                ax.set_ylabel("Range-Scaled MAE")
                ax.grid(which="major", linestyle="--", linewidth=0.5)

                # make sure the y-ticks are labeled
                ax.yaxis.set_tick_params(which="both", labelleft=True)

                ax.spines[["right", "top"]].set_visible(False)
                ax.label_outer()

        axes[0, 0].set_ylim(1e-4, 1)

        store_dir = output_dir  # / data
        if not store_dir.exists():
            store_dir.mkdir(parents=True, exist_ok=True)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="outside upper center", ncol=len(methods), fontsize=6)

        plt.savefig(store_dir / f"mean_multi_grid_{error}.pdf")


if __name__ == "__main__":
    df_grouped = _read_all_csvs("mean_multi_dim")

    output_dir = Path("plots_paper")
    plot_method_comparison(df_grouped, output_dir)
    plot_mean_dim_3n(df_grouped, output_dir)
    plot_mean_dim_3d(df_grouped, output_dir)
    plot_grid_n_method(df_grouped, output_dir)
