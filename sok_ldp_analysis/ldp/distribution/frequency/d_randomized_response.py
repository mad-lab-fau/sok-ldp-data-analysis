"""
This module contains the implementation of the d-ary randomized response mechanism for frequency estimation (also
known as d-RR, k-RR, generalized randomized response, or direct encoding).

This mechanism was first introduced by [1] and is a generalization of the binary randomized response mechanism.
It was later termed "generalized randomized response" in [2].
The implementation follows the description in [2].

References:
    [1] P. Kairouz, S. Oh, and P. Viswanath, “Extremal Mechanisms for Local Differential Privacy,
    ” in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2014. Accessed: Mar. 18,
    2024. [Online]. Available: https://papers.nips.cc/paper_files/paper/2014/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
    [2] Z. Li, T. Wang, M. Lopuhaä-Zwakenberg, N. Li, and B. Škoric, “Estimating Numerical
    Distributions under Local Differential Privacy,” in Proceedings of the 2020 ACM SIGMOD International Conference
    on Management of Data, Portland OR USA: ACM, Jun. 2020, pp. 621–635. doi: 10.1145/3318464.3389700.
"""

import numpy as np

from sok_ldp_analysis.ldp.distribution.frequency.frequency_oracle import FrequencyOracle


def d_rr_response(t: np.array, eps: float, d: int, rng: np.random.Generator) -> np.array:
    """
    Apply the d-ary randomized response mechanism [1,2] to the input data.

    Args:
        t: The private data of n clients (vector of size n)
        eps: The privacy budget epsilon.
        d: The domain size.
        rng: A numpy random number Generator

    Returns: The disturbed data.
    """
    p = np.exp(eps) / (np.exp(eps) + d - 1)

    # randomly pick "offsets" in the range {1, 2, ..., d-1} for each client
    offset = rng.integers(1, d, size=t.shape)

    # create a mask for the true value
    mask = rng.binomial(1, p, size=t.shape)

    # return the true value if the mask is true, otherwise shift the value by the offset
    return np.where(mask, t, (t + offset) % d)


def d_rr_estimate_frequencies(u: np.array, eps: float, d: int) -> np.array:
    """
    Estimate the frequencies of the input data disturbed by the d-ary randomized response mechanism [1,2].

    Args:
        u: The disturbed data.
        eps: The privacy budget epsilon.
        d: The domain size.

    Returns: The estimated frequencies (vector of size d).
    """
    n = u.shape[0]
    p = np.exp(eps) / (np.exp(eps) + d - 1)
    q = (1 - p) / (d - 1)
    counts = np.bincount(u, minlength=d)
    return (counts / n - q) / (p - q)


class DRandomizedResponse(FrequencyOracle):
    """
    Implementation of the d-ary randomized response mechanism for frequency estimation [1,2].
    """

    def response(self, t: np.array) -> np.array:
        """
        Apply the d-ary randomized response mechanism to the input data.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The disturbed data.
        """
        return d_rr_response(t, self.eps, self.domain_size, self.rng)

    def frequencies(self, u: np.array) -> np.array:
        """
        Estimate the frequencies of the input data.

        Args:
            u: The disturbed data.

        Returns: The estimated frequencies (vector of size domain_size).
        """
        return d_rr_estimate_frequencies(u, self.eps, self.domain_size)
