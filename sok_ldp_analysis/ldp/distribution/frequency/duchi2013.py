"""
This module contains the implementation of a frequency oracle by Duchi et al., 2013 [1].

References:
    [1] J. Duchi, M. J. Wainwright, and M. I. Jordan, “Local Privacy and Minimax Bounds: Sharp Rates for
    Probability Estimation,” in Advances in Neural Information Processing Systems, Curran Associates, Inc.,
    2013. Accessed: Sep. 04, 2023. [Online]. Available:
    https://papers.nips.cc/paper_files/paper/2013/hash/5807a685d1a9ab3b599035bc566ce2b9-Abstract.html

"""


import numpy as np

from sok_ldp_analysis.ldp.distribution.frequency.frequency_oracle import FrequencyOracle
from sok_ldp_analysis.ldp.distribution.util import project_onto_prob_simplex


class Duchi2013(FrequencyOracle):
    """
    Duchi et al.'s method for multinomial estimation [1].
    """

    def response(self, x: np.array) -> np.array:
        """
        Perturb the data using Duchi et al.'s method for multinomial estimation.

        Args:
            x: Array of size (n,) containing the data from n clients. The values are integers in [domain_size].

        Returns: Array of size (n, domain_size) containing the perturbed data for each client.
        """
        # One-hot encode the data
        X_one_hot = self._one_hot(x)

        X_one_hot_flipped = np.zeros(X_one_hot.shape, dtype=bool)

        for i in range(X_one_hot.shape[0]):
            flip = self.rng.binomial(1, 1 / (1 + (np.exp(self.eps / 2))), size=X_one_hot.shape[1])

            # Flip the entry with probability 1/(1+exp(eps/2))
            X_one_hot_flipped[i] = np.logical_xor(X_one_hot[i], flip)

        return X_one_hot_flipped

    def frequencies(self, z: np.array) -> np.array:
        """
        Estimate the frequencies from the perturbed data.

        Args:
            z: Array of size (n, domain_size) containing the perturbed data for each client.

        Returns: Array of size (domain_size,) containing the estimated frequencies.
        """
        # Sum the matrix over the rows and subtract 1/(1+exp(eps/2)) from every entry in the response matrix
        sum_ = np.sum(z, axis=0) - z.shape[0] / (np.exp(self.eps / 2) + 1)

        # Correct the sum by multiplying with (exp(eps/2)+1)/(exp(eps/2)-1)
        sum_ = sum_ * (np.exp(self.eps / 2) + 1) / (np.exp(self.eps / 2) - 1)

        # Calculate the frequencies
        freq = sum_ / z.shape[0]

        # Project onto the probability simplex
        freq = project_onto_prob_simplex(freq)

        return freq

    def _one_hot(self, x):
        return np.eye(self.domain_size, dtype=bool)[x]
