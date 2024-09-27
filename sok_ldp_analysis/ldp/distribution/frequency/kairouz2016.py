"""
This module implements the O-RR frequency oracle by Kairouz et al. (2016) [1] for closed alphabets.

[1] P. Kairouz, K. Bonawitz, and D. Ramage, “Discrete Distribution Estimation under Local Privacy,” in Proceedings of
The 33rd International Conference on Machine Learning, PMLR, Jun. 2016, pp. 2436–2444. Accessed: Feb. 09,
2024. [Online]. Available: https://proceedings.mlr.press/v48/kairouz16.html

"""
import math

import numpy as np
import scipy

from sok_ldp_analysis.ldp.distribution.frequency.d_randomized_response import d_rr_response
from sok_ldp_analysis.ldp.distribution.frequency.frequency_oracle import FrequencyOracle
from sok_ldp_analysis.ldp.distribution.util import project_onto_prob_simplex


class Kairouz2016(FrequencyOracle):
    """
    The O-RR frequency oracle by Kairouz et al. (2016) [1] for closed alphabets.

    Note that we implement it using permutations instead of hash functions (see section 5.4 of [1]).
    """

    def __init__(self, eps: float, domain_size: int, num_cohorts: int = 1, rng: np.random.Generator = None):
        """
        Initialize the frequency estimator.

        Args:
            self.eps: The privacy budget self.epsilon.
            domain_size: The domain size of the input data. We assume that the input data are integers in the range
            [0, domain_size).
            num_cohorts: The number of cohorts.
            rng: A numpy random number Generator. If None, the default rng is used.
        """
        super().__init__(eps, domain_size, rng)
        self.num_cohorts = num_cohorts

        # generate the permutations for the cohorts
        perm_rng = np.random.default_rng(self.num_cohorts + 1)
        self.permutations = np.array([perm_rng.permutation(domain_size) for _ in range(num_cohorts)])
        # self.permutations = np.array([np.arange(domain_size) for _ in range(num_cohorts)])

    def _d_rr_response(self, x_: np.array, c) -> np.array:
        """
        Apply the d-ary randomized response mechanism to the permuted input data and shift the values based on the cohort.

        Args:
            x_: The permuted input data.
            c: The cohort of the client.

        Returns: The disturbed data.
        """
        return c * self.domain_size + d_rr_response(x_, self.eps, self.domain_size, self.rng)

    def response(self, x: np.array) -> np.array:
        """
        Run the mechanism and return the response.

        For the hash functions, we use permutations (see section 5.4 of [1]).

        Args:
            x: The private data of n clients (vector of size n)

        Returns: The disturbed data.
        """
        # assign each client to a cohort
        c = self.rng.integers(0, self.num_cohorts, len(x))

        return self._d_rr_response(self.permutations[c, x], c)

    def frequencies(self, z: np.array) -> np.array:
        """
        Estimate the frequencies from the responses.

        Args:
            z: The disturbed data.

        Returns: The estimated frequencies (vector of size domain_size).
        """

        # empirical frequencies over the (k * num_cohorts) options
        m_ = np.bincount(z, minlength=self.num_cohorts * self.domain_size) / len(z)

        # (k * num_cohorts) x k matrix encoding the outputs of the permutations
        h = np.zeros((self.domain_size * self.num_cohorts, self.domain_size))
        for i in range(self.num_cohorts):
            h[i * self.domain_size : (i + 1) * self.domain_size, :] = np.eye(self.domain_size)[self.permutations[i]].T

        rhs = (1 / (math.exp(self.eps) - 1)) * (self.num_cohorts * (math.exp(self.eps) + self.domain_size - 1) * m_ - 1)

        p = scipy.optimize.lsq_linear(h, rhs)

        return project_onto_prob_simplex(p.x)
