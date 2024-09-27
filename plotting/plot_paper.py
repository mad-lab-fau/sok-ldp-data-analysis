import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd

plt.style.use("paper.mplstyle")


# paper width: 7in for 2 columns
# paper width: 3.3in for 1 column
def _fig_size(width_fraction=1.0, two_columns=False, height=None):
    if two_columns:
        width = 7
    else:
        width = 3.3

    width = width * width_fraction

    if height is None:
        golden_ratio = (5**0.5 - 1) / 2
        height = width * golden_ratio

    return (width, height)


method_mapping_mean = {
    "Laplace1D": "Laplace Mechanism",
    "Duchi2018": "Duchi et al. (2018) - $\\ell_2$",
    "Wang2019Piecewise1D": "Wang et al. (2019) - Piecewise",
    "Wang2019Hybrid1D": "Wang et al. (2019) - Hybrid",
    "Ding2017LargeM": "Bernoulli Mechanisms",
    "Nguyen2016": "Nguyen et al. (2016)",
    "Duchi2018LInf": "Duchi et al. (2018) - $\\ell_{\\infty}$",
    "Wang2019Duchi1D": 'Wang et al. (2019) - "Duchi et al."',
    "WaudbySmith2023Mean": "Waudby-Smith et al. (2023)",
}

method_mapping_mean_short = {
    "Laplace1D": "Laplace",
    "Duchi2018": "Duchi $\\ell_2$",
    "Wang2019Piecewise1D": "Wang Piecewise",
    "Wang2019Hybrid1D": "Wang Hybrid",
    "Nguyen2016": "Bernoulli",
}


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def _read_all_csvs(method_name):
    import glob
    import os

    df = pd.concat(
        map(
            pd.read_csv,
            glob.glob(os.path.join("..", "experiments", "results_grouped", "simulation", method_name, "*.csv")),
        )
    )
    df = df.reset_index(drop=True)
    return df


def _read_all_pkls(method_name):
    import glob
    import os

    df = pd.concat(
        map(
            pd.read_pickle,
            glob.glob(os.path.join("..", "experiments", "results_grouped", "simulation", method_name, "*.pkl")),
        )
    )
    df = df.reset_index(drop=True)
    return df


def _add_subplot_label(ax, fig, label, x=0.0, y=0.25):
    # add a subplot label inside the subplot - see:
    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.text(
        x,
        y,
        label,
        horizontalalignment="left",
        transform=ax.transAxes + trans,
        fontsize="medium",
        verticalalignment="top",
        fontfamily="serif",
        bbox=dict(facecolor="0.9", edgecolor="none", pad=3.0),
    )
