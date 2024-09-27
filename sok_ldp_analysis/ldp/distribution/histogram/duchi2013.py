"""
1-dimensional histogram estimator by Duchi et al. (2013) [1].

[1] J. Duchi, M. J. Wainwright, and M. I. Jordan, “Local Privacy and Minimax Bounds: Sharp Rates for Probability
Estimation,” in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2013. Accessed: Sep. 04,
2023. [Online]. Available: https://papers.nips.cc/paper_files/paper/2013/hash/5807a685d1a9ab3b599035bc566ce2b9
-Abstract.html"""

import numpy as np

from sok_ldp_analysis.ldp.distribution.histogram.histogram import HistogramEstimator
from sok_ldp_analysis.ldp.distribution.util import project_onto_prob_simplex


class Duchi2013Histogram(HistogramEstimator):
    """1-dimensional histogram estimator from Duchi et al. (2013) [1]."""

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
            t: Array of size (n,) containing the data from n clients. The value range is [0, 1].

        Returns: Array of size (n, k) containing the noisy histogram for each client.

        """
        k = self.bin_count
        # Set k optimally as in Duchi et al. (2013)
        if k is None:
            k = int((t.shape[0] * self.eps**2) ** (1 / 4)) + 1

        x = self._transform_input(t)

        # Generate the responses
        resp = self.rng.laplace(scale=2 / self.eps, size=(k, x.shape[0]))
        for i in range(k - 1):
            resp[i][(i / k <= x) & ((i + 1) / k > x)] += 1

        resp[k - 1][((k - 1) / k <= x) & (x <= 1)] += 1

        return resp.T

    def histogram(self, u: np.array) -> np.array:
        """
        Estimate the histogram from the responses.

        Args:
            u: Array of size (n, k) containing the noisy histogram for each client.

        Returns: Tuple of size 2 containing the estimated histogram and the bins.

        """
        k = u.shape[1]
        bins = np.array([j / k for j in range(k)])
        hist_ = np.sum(u, axis=0) / u.shape[0]

        # Project onto the probability simplex
        hist = project_onto_prob_simplex(hist_)

        return hist, self._transform_bins(bins)
