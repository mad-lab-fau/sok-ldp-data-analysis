"""
Implementation of the mean and confidence interval estimation from Waudby-Smith et al. 2023 [1].

[1] I. Waudby-Smith, S. Wu, and A. Ramdas, “Nonparametric Extensions of
    Randomized Response for Private Confidence Sets,” in Proceedings of the 40th International Conference on Machine
    Learning, PMLR, Jul. 2023, pp. 36748-36789. Available: https://proceedings.mlr.press/v202/waudby-smith23a.html
"""

from typing import Tuple

import numpy as np

from sok_ldp_analysis.ldp.mean.mean import Mean1D
from sok_ldp_analysis.ldp.mean.mean_ci import MeanAndConfidenceInterval1D


class WaudbySmith2023CI(MeanAndConfidenceInterval1D):
    """Implementation of the mean and confidence interval estimation from Waudby-Smith et al. 2023 [1]."""

    def mechanism(self, x: np.array, g: int = 1) -> np.array:
        """Eps-LDP mechanism "Nonparametric randomized response" (NPRR) for disturbing 1-dimensional input (of multiple
        clients) based on Algorithm 2 in [1].

        We fix epsilon for all clients to be eps.

        Args:
            x: The private data of n clients (vector of size n)
            g: The number of bins to discretize the data into. Default is 1.
               This is the recommended value for Hoeffding confidence intervals implemented in this class.


        Returns: The disturbed data.
        """
        # Only works for 1-dimensional inputs
        assert x.ndim == 1

        # Scale data to be in range [0, 1]
        x = (x - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

        assert np.all(x <= 1 + 1e-10)
        assert np.all(x >= 0 - 1e-10)

        # Step 1: Discretize x into y via stochastic rounding
        x_ceil = np.ceil(g * x) / g
        x_floor = np.floor(g * x) / g

        selection = self.rng.binomial(1, g * (x - x_floor), size=len(x))
        y_ = np.where(selection, x_ceil, x_floor)

        # TODO: this should be superfluous as y_ == x if x_ceil == x_floor
        y = np.where(x_ceil == x_floor, x, y_)

        # Step 2: Privatize y into z via k-RR
        # Note: k = g+1
        # and: the probability is slightly different from k-RR to account for the uniform sampling (which could return
        # the true value); k-RR would sample only non-true values with prob. exp(eps) / (exp(eps) + k - 1)
        r = (np.exp(self.eps) - 1) / (np.exp(self.eps) + g)

        u = self.rng.choice([x / g for x in range(g + 1)], size=len(x), replace=True)

        # for each client return y with probability r and u with probability 1-r
        z = np.where(self.rng.binomial(1, r, size=len(x)), y, u)

        return z

    def mean_and_ci(self, z: np.array, alpha: float = 0.1, g: int = 1) -> Tuple[float, float, float]:
        """Compute the mean and Hoeffding confidence interval according to Waudby-Smith et al. 2023.

        Note that we follow equation (12) which provides a tighter confidence interval than equation (11).

        We assume that epsilons are equal for all clients.
        We assume that G is equal to 1 for all clients.

        Args:
            z: The disturbed data.
            alpha: The confidence level. Default is 0.1. Pr[mean not in CI] <= alpha.
            g: The number of bins to discretize the data into. Default is 1. Needs to be the same as in the mechanism.

        Returns: A tuple containing the mean, the lower bound of the confidence interval and the upper bound of the
            confidence interval.
        """
        # Only works for 1-dimensional inputs
        assert z.ndim == 1

        n = len(z)
        r = (np.exp(self.eps) - 1) / (np.exp(self.eps) + g)

        # Equation (10)
        mean = np.mean(z - (1 - r) / 2) / r

        # Equation (12)
        lambda_n = np.sqrt(8 * np.log(1 / alpha) / n)
        lower = np.max(mean - (np.log(1 / alpha) + np.arange(1, n + 1)) / (lambda_n * n * r))

        # TODO: not in the paper, but Hoeffding confidence intervals should be symmetric
        upper = 2 * mean - lower

        # Rescale to the input range
        mean = mean * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        lower = lower * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        upper = upper * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

        return mean, lower, upper


class WaudbySmith2023Mean(Mean1D):
    """Implementation of the mean estimation from Waudby-Smith et al. 2023 [1]."""

    def __init__(self, eps: float, input_range: Tuple[float, float], g: int = 1, rng: np.random.Generator = None):
        """
        Initialize the mean estimator.

        Args:
            eps: The privacy budget epsilon.
            input_range: The range of the input data.
            g: The number of bins to discretize the data into. Default is 1.
            rng: A numpy random number Generator.
        """
        super().__init__(eps, input_range, rng)
        self.ws = WaudbySmith2023CI(eps=eps, rng=rng, input_range=input_range)
        self.g = g

    def mechanism(self, t: np.array) -> np.array:
        """
        Eps-LDP mechanism "Nonparametric randomized response" (NPRR) for disturbing 1-dimensional input (of multiple
        clients) based on Algorithm 2 in [1].

        We fix epsilon for all clients to be eps.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The disturbed data.
        """
        self.ws.mechanism(t, g=self.g)

    def mean(self, u: np.array) -> float:
        """
        Estimate the mean based on the disturbed data.

        Args:
            u: The disturbed data.

        Returns: The estimated mean.
        """
        return self.ws.mean_and_ci(u, g=self.g)[0]
