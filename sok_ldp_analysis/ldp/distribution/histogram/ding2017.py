"""
1-dimensional histogram estimator from Ding et al. (2017) [1].

[1] B. Ding, J. Kulkarni, and S. Yekhanin, “Collecting Telemetry Data Privately,” in Advances in Neural Information
Processing Systems, Curran Associates, Inc., 2017. Accessed: Jul. 04, 2023. [Online]. Available:
https://proceedings.neurips.cc/paper_files/paper/2017/hash/253614bbac999b38b5b60cae531c4969-Abstract.html
"""

from typing import Tuple, Optional

import numpy as np

from sok_ldp_analysis.ldp.distribution.histogram.histogram import HistogramEstimator


class Ding2017Histogram(HistogramEstimator):
    """1-dimensional histogram estimator from Ding et al. (2017) [1]"""

    def __init__(
        self,
        eps: float,
        input_range: Tuple[float, float],
        bin_count: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        d: int = 1,
    ):
        """
        Initialize the histogram estimator.

        Args:
            eps: The privacy budget epsilon.
            input_range: The range of the input data.
            bin_count: The number of bins to use.
            rng: A numpy random number Generator. If None, the default rng is used.
            d: The number of buckets to sample for each client.
        """
        super().__init__(eps, input_range, bin_count, rng)
        self.d = d

    def _transform_input(self, t: np.array) -> np.array:
        """Transform the input data to the range [0, 1]."""
        return (t - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

    def _transform_bins(self, bins: np.array) -> np.array:
        """Transform the histogram bins to the original range."""
        return bins * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

    def response(self, t: np.array) -> np.array:
        """
        Run the mechanism and return the response.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The response.
        """
        # map the data to the range [0, 1]
        x = self._transform_input(t)

        # map the input to the corresponding bucket
        v = np.zeros((t.shape[0], self.bin_count), dtype=bool)
        for i in range(self.bin_count):
            v[:, i] = np.logical_and(i / self.bin_count < x, x < (i + 1) / self.bin_count)

        # randomly draw d bucket numbers without replacement for each client
        js = np.vstack([self.rng.permutation(self.bin_count)[: self.d] for _ in range(t.shape[0])])
        js_mask = np.zeros((t.shape[0], self.bin_count), dtype=bool)
        for i in range(self.d):
            js_mask[np.arange(len(x)), js[:, i]] = True

        p = (np.exp(self.eps / 2)) / (np.exp(self.eps / 2) + 1)

        bs = np.zeros((t.shape[0], self.bin_count), dtype=int)
        bs[v] = self.rng.binomial(n=1, p=p, size=(t.shape[0], self.bin_count))[v]
        bs[~v] = self.rng.binomial(n=1, p=1 - p, size=(t.shape[0], self.bin_count))[~v]

        bs[js_mask == 0] = 0

        return js_mask, bs

    def histogram(self, u: Tuple[np.array, np.array]) -> np.array:
        """
        Compute the histogram of the response.

        Args:
            u: Array of size (n, k) containing the response for each client.

        Returns: Tuple of size 2 containing the estimated histogram and the bins.
        """
        indices, resp = u
        n = indices.shape[0]
        k = self.bin_count
        hist = np.zeros(k)
        bins = np.array([j / k for j in range(k)])

        for i in range(k):
            hist[i] = (
                k
                / (self.d * n)
                * np.sum((resp[indices[:, i] == 1, i] * (np.exp(self.eps / 2) + 1) - 1) / (np.exp(self.eps / 2) - 1))
            )

        return hist, self._transform_bins(bins)
