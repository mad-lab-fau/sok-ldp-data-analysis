"""Mean estimation using the Harmony method according to Nguyên et al. [1].

[1] T. T. Nguyên, X. Xiao, Y. Yang, S. C. Hui, H. Shin, and J. Shin, “Collecting and Analyzing Data from Smart Device
Users with Local Differential Privacy.” arXiv, Jun. 16, 2016. doi: 10.48550/arXiv.1606.05053.

"""
import math

import numpy as np

from sok_ldp_analysis.ldp.mean.mean import MeanMultiDim


class Nguyen2016(MeanMultiDim):
    """Mean estimation using the Harmony method according to Nguyên et al., 2016."""

    def mechanism(self, t: np.array) -> np.array:
        """
        Eps-LDP mechanism for disturbing multidimensional input (of multiple clients).

        Harmony works by sampling a random index for each client. Only this index is disturbed.

        Args:
            t: The private data of n clients (matrix of size n x d)

        Returns: The disturbed data (matrix of size n x d). Only one entry per client is non-zero.
        """
        n = t.shape[0]
        d = t.shape[1]

        # Scale data to be in range [-1, 1]
        range_size = self.input_range[1] - self.input_range[0]
        x = 2 * (t - self.input_range[0]) / range_size - 1

        assert np.all(np.abs(x) <= 1 + 1e-10)

        # Uniformly sample a random index for each client (using self.rng)
        idx = self.rng.integers(low=0, high=d, size=n)

        # Sample a Bernoulli random variable for each client
        u = self.rng.binomial(
            n=1,
            p=(
                (x[np.arange(n), idx] * (math.exp(self.eps) - 1) + math.exp(self.eps) + 1)
                / (2 * math.exp(self.eps) + 2)
            ),
            size=n,
        )

        ret = np.zeros_like(x)
        ret[u == 1, idx[u == 1]] = d * (math.exp(self.eps) + 1) / (math.exp(self.eps) - 1)
        ret[u == 0, idx[u == 0]] = -d * (math.exp(self.eps) + 1) / (math.exp(self.eps) - 1)

        return ret

    def mean(self, u: np.array) -> np.array:
        """Compute the mean of the disturbed data."""
        mean = np.mean(u, axis=0)

        # Return the mean in the original range
        return (mean + 1) / 2 * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
