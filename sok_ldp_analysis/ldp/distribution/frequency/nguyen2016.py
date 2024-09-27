"""
This module contains the implementation of the frequency oracle by Nguyên et al. (2016) [1].

[1] T. T. Nguyên, X. Xiao, Y. Yang, S. C. Hui, H. Shin, and J. Shin, “Collecting and Analyzing Data from Smart Device
Users with Local Differential Privacy.” arXiv, Jun. 16, 2016. doi: 10.48550/arXiv.1606.05053.

"""
import math

import numpy as np

from sok_ldp_analysis.ldp.distribution.frequency.frequency_oracle import FrequencyOracle


def _generate_orthogonal_set(k: int) -> np.array:
    """
    Generate a set of k orthogonal vectors (where k is a power of 2).

    Algorithm 5 from the appendix of [1].

    If k is not a power of 2, the set is generated for the smallest power of 2 greater than k.

    Args:
        k: The number of vectors to generate

    Returns: A set of k orthogonal vectors
    """
    s = np.array([np.array([1, -1]), np.array([1, 1])], dtype=np.int8)
    while len(s) < k:
        s = np.concatenate((np.concatenate((s, s), axis=1), np.concatenate((s, -s), axis=1)))

    return np.array(s)


class Nguyen2016FO(FrequencyOracle):
    """
    Frequency oracle by Nguyên et al. (2016) [1].

    Here we only implement the case for a single categorical attribute.
    """

    def response(self, x: np.array) -> np.array:
        """
        Generate the response for the given data.

        Args:
            x: The data by n participants.

        Returns: The response.
        """
        n = len(x)
        # Generate the orthogonal set
        m = self.domain_size
        psi = _generate_orthogonal_set(m)

        s = self.rng.choice(m, size=n)
        t = self.rng.binomial(1, (math.exp(self.eps)) / (math.exp(self.eps) + 1), size=n)

        c_eps = (math.exp(self.eps) + 1) / (math.exp(self.eps) - 1)

        alpha = c_eps * m * psi[s, x] * 1 / math.sqrt(self.domain_size**2)

        alpha[t] = -alpha[t]

        return s, alpha

    def frequencies(self, z: np.array) -> np.array:
        """
        Estimate the frequencies of the disturbed data.

        Args:
            z: The responses from n participants.

        Returns: The estimated frequencies.
        """
        s, alpha = z
        z_ = np.zeros(self.domain_size)
        for k in range(self.domain_size):
            z_[k] = np.sum(alpha[s == k])

        psi = _generate_orthogonal_set(self.domain_size)

        # add empty bins to have z_ of length of psi
        z_ = np.concatenate((z_, np.zeros(psi.shape[0] - z_.shape[0])))

        return (np.dot(psi.T, z_))[: self.domain_size] / len(s)
