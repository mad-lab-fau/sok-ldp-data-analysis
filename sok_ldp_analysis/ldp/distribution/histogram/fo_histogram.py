from typing import Tuple, Optional, Type

import numpy as np

from sok_ldp_analysis.ldp.distribution.frequency.frequency_oracle import FrequencyOracle
from sok_ldp_analysis.ldp.distribution.histogram.histogram import HistogramEstimator


class FrequencyOracleHistogram(HistogramEstimator):
    """
    Histogram estimator based on a frequency oracle.
    """

    def __init__(
        self,
        eps: float,
        input_range: Tuple[float, float],
        fo: Type[FrequencyOracle],
        bin_count: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the histogram estimator.

        Args:
            eps: The privacy budget epsilon.
            input_range: The range of the input data.
            fo: The frequency oracle class to use.
            bin_count: The number of bins to use.
            rng: A numpy random number Generator. If None, the default rng is used.
        """
        super().__init__(eps, input_range, bin_count, rng)
        self.fo = fo(eps, bin_count, rng)

    def _transform_input(self, t: np.array) -> np.array:
        """
        Transform the input data to the range [0, 1].

        Args:
            t: Array of size (n,) containing the data from n clients.

        Returns: Array of size (n,) containing the transformed data.
        """
        return (t - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

    def _transform_bins(self, bins: np.array) -> np.array:
        """
        Transform the histogram bins to the original range.

        Args:
            bins: Array of size (k,) containing the histogram bins in the range [0, 1].

        Returns: Array of size (k,) containing the transformed bins.
        """
        return bins * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

    def response(self, t: np.array) -> np.array:
        """
        Run the mechanism and return the response.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The frequency oracle response.
        """
        # calculate which bin each element falls into
        x = self._transform_input(t)
        bin_indices = np.floor(x * self.bin_count).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.bin_count - 1)

        # Use the frequency oracle to produce the response
        return self.fo.response(bin_indices)

    def histogram(self, u: np.array) -> np.array:
        """
        Compute the histogram of the response.

        Args:
            u: Array of size (n, k) containing the frequency oracle response for each client.

        Returns: Tuple of size 2 containing the estimated histogram and the bins.
        """
        freqs = self.fo.frequencies(u) / len(u)
        bins = np.array([i / self.bin_count for i in range(self.bin_count)])
        return freqs, self._transform_bins(bins)
