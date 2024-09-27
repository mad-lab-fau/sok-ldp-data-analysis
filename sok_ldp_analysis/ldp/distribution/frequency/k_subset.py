"""
This module contains the implementation of the k-subset frequency oracle, independently proposed by
Wang et al. (2016) [1] and Ye and Barg (2017) [2].

[1] S. Wang et al., “Mutual Information Optimally Local Private Discrete Distribution Estimation.”
arXiv, Jul. 27, 2016. doi: 10.48550/arXiv.1607.08025.

[2] M. Ye and A. Barg, “Optimal schemes for discrete distribution estimation under local differential privacy,”
in 2017 IEEE International Symposium on Information Theory (ISIT), Aachen, Germany: IEEE, Jun. 2017,
pp. 759–763. doi: 10.1109/ISIT.2017.8006630.
"""
import math

import numpy as np

from sok_ldp_analysis.ldp.distribution.frequency.frequency_oracle import FrequencyOracle
from sok_ldp_analysis.ldp.distribution.util import project_onto_prob_simplex


class KSubset(FrequencyOracle):
    """
    K-subset frequency oracle based on Wang et al. (2016) [1] and Ye and Barg (2017) [2].
    """

    def __init__(self, eps: float, domain_size: int, k: int = None, rng: np.random.Generator = None):
        super().__init__(eps, domain_size, rng)

        # Set k optimally as in Wang et al. (2016), section 3.3
        if k is None:
            k = int(domain_size / (math.exp(eps) + 1))

            if k > domain_size:
                k = domain_size

            if k < 1:
                k = 1

        assert 1 <= k <= domain_size, "The value of k must be between 1 and the domain size."
        self.k = k

    def response(self, t: np.array) -> np.array:
        """
        Run the mechanism and return the response.

        Algorithm 1 in Wang et al. (2016). Instead of responding with a set, each participant responds with a binary
        vector where indices with a 1 are in the set.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The response.
        """
        n = t.shape[0]

        p = self.k * math.exp(self.eps) / (self.k * math.exp(self.eps) + self.domain_size - self.k)

        # Generate the responses
        resp = np.zeros((n, self.domain_size), dtype=bool)
        for i in range(n):
            rem = self.k
            if self.rng.random() < p:
                resp[i, t[i]] = 1
                rem -= 1

            # sample rem elements but not t[i]
            idx = self.rng.choice(self.domain_size - 1, rem, replace=False)
            idx = (idx + t[i]) % self.domain_size
            resp[i, idx] = 1

        return resp

    def frequencies(self, u: np.array) -> np.array:
        """
        Estimate the frequencies from the responses.

        Algorithm 2 in Wang et al. (2016).

        Args:
            u: The response.

        Returns: The estimated frequencies (vector of size domain_size).
        """
        n = u.shape[0]
        k = self.k
        d = self.domain_size

        frequencies = np.sum(u, axis=0)

        theta_hat = np.zeros(self.domain_size)

        g_k = (k * math.exp(self.eps)) / (k * math.exp(self.eps) + d - k)
        h_k = g_k * ((k - 1) / (d - 1)) + ((d - k) / (k * math.exp(self.eps) + d - k)) * (k / (d - 1))

        for i in range(len(frequencies)):
            theta_hat[i] = (frequencies[i] - n * h_k) / (n * (g_k - h_k))

        # project onto the probability simplex, see Wang et al. (2016), section 5
        theta_hat = project_onto_prob_simplex(theta_hat)

        return theta_hat
