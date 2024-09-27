import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.plot_paper import _fig_size
from sok_ldp_analysis.simulation.data.real import USCensus, Adult, NYCTaxiData
from sok_ldp_analysis.simulation.data.synthetic_continuous import BimodalData1D, UniformLargeData1D, UniformSmallData1D
from sok_ldp_analysis.simulation.data.synthetic_discrete import BinomialData1D, GeometricData1D

dataset_mapping = {
    "USCensus": "US Census",
    "Adult": "Adult",
    "NYCTaxiData": "NYC Taxi",
    "BimodalData1D": "Bimodal",
    "UniformLargeData1D": "Uniform Large",
    "UniformSmallData1D": "Uniform Small",
    "BinomialData1D": "Binomial",
    "GeometricData1D": "Geometric, $k=256$",
}


def plot_datasets(output_path):
    output_path = output_path / "datasets"
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = data_discrete + data_continuous

    fig = plt.figure(figsize=_fig_size(two_columns=True, height=6))
    gs = fig.add_gridspec(
        math.ceil(len(datasets) / 2), 2, left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.4
    )
    axes = gs.subplots(sharex=False, sharey=False).flatten()

    for ax, dataset in zip(axes, datasets):
        data = dataset.data

        if dataset in data_discrete:
            xlim = (0, dataset.domain_size)

            # count unique values as histogram
            unique, unique_counts = np.unique(data, return_counts=True)

            # Calculate Density
            unique_counts = unique_counts / len(data)

            ax.bar(unique, unique_counts, width=1)
        else:
            # optimal histogram bin size - Freedman-Diaconis rule
            q75, q25 = np.percentile(dataset.data, [75, 25])
            iqr = q75 - q25
            h = 2 * iqr / (len(dataset.data) ** (1 / 3))
            maximum = dataset.input_range[1]
            minimum = dataset.input_range[0]
            bins = int((maximum - minimum) / h)
            xlim = dataset.input_range

            # Calculate histogram and bins with numpy
            hist, bin_edges = np.histogram(data, bins=bins, range=xlim, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax.bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0])

        ax.set_xlim(xlim)
        ax.set_title(dataset_mapping[dataset.__class__.__name__])

        # Remove y-axis labels
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_yticks([])
        ax.set_ylabel("Density")

    plt.savefig(output_path / "datasets.pdf")


if __name__ == "__main__":
    data_rng = np.random.default_rng(0)

    n = 1000000

    data_discrete = [
        BinomialData1D(data_rng, n, 100, 0.2),
        GeometricData1D(data_rng, n, 256, None),
        USCensus(data_rng, data_path="../data"),
        Adult(data_rng, data_path="../data"),
    ]

    data_continuous = [
        BimodalData1D(data_rng, n, 1, 0),
        UniformLargeData1D(data_rng, n),
        UniformSmallData1D(data_rng, n),
        NYCTaxiData(data_rng, data_path="../data"),
    ]

    output_dir = Path("plots_paper")
    plot_datasets(output_dir)
