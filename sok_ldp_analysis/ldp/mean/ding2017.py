"""
Ding et al.'s sampling method [1] for 1-dimensional input (of multiple clients).

[1] B. Ding, J. Kulkarni, and S. Yekhanin, “Collecting Telemetry Data Privately,” in Advances in Neural Information
Processing Systems, Curran Associates, Inc., 2017.
https://proceedings.neurips.cc/paper_files/paper/2017/hash/253614bbac999b38b5b60cae531c4969-Abstract.html
"""

import math
from typing import Tuple

import numpy as np

from sok_ldp_analysis.ldp.mean.mean import Mean1D


class Ding2017(Mean1D):
    """Ding et al.'s sampling method for 1-dimensional input (of multiple clients)."""

    def __init__(self, eps: float, input_range: Tuple[float, float], m: int = 1, rng: np.random.Generator = None):
        """Initialize the mean estimator."""
        super().__init__(eps, input_range, rng)
        self.m = m

    def mechanism(self, x: np.array) -> np.array:
        """One-bit sampling method for 1-dimensional input (of multiple clients) from range [0, m].

        Each value is replaced by 1 with probability 1/(exp(eps)+1) + x/m * (exp(eps)-1)/(exp(eps)+1) and 0 otherwise.

        Args:
            x: The private data of n clients (vector of size n)

        Returns: The disturbed data.

        """
        assert x.ndim == 1

        # Transform the data to range [0, m]
        x = (x - self.input_range[0]) / (self.input_range[1] - self.input_range[0]) * self.m

        # Check that the data is in range [0, m]
        assert np.all(x >= 0)
        assert np.all(x <= self.m)

        p = 1 / (math.exp(self.eps) + 1) + x / self.m * ((math.exp(self.eps) - 1) / (math.exp(self.eps) + 1))
        ret = self.rng.binomial(n=1, p=p, size=x.shape)

        return ret

    def mean(self, u: np.array) -> float:
        """Compute the mean of the one-bit sampled data. The private data is assumed to be from range [0, m].

        Args:
            u: The noisy one-bit responses.

        Returns: The mean of the one-bit sampled data.

        """
        x = (u * (np.exp(self.eps) + 1) - 1) / (np.exp(self.eps) - 1)
        mean = self.m * np.mean(x)

        # Return the mean in the original range
        return mean / self.m * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
