"""
This module contains the implementation of the frequency oracle by Murakami et al. (2018) [1].

[1] T. Murakami, H. Hino, and J. Sakuma, “Toward Distribution Estimation under Local Differential Privacy with Small
Samples,” Proceedings on Privacy Enhancing Technologies, 2018, doi: 10.1515/popets-2018-0022."""

import math

import numpy as np

from sok_ldp_analysis.ldp.distribution.frequency.d_randomized_response import d_rr_response
from sok_ldp_analysis.ldp.distribution.frequency.frequency_oracle import FrequencyOracle


def _find_a_hat(p_hat, g_n, n):
    """
    Algorithm 2, step 2, in section 4.2 of Murakami et al. (2018) [1].

    Args:
        p_hat: The estimated frequencies. A vector of size d.
        g_n: The mechanism matrix. A matrix of size d x n.
        n: The number of samples. An integer.

    Returns: The estimated a_hat.
    """
    d = len(p_hat)

    # equation 22
    s_n = (1 / (p_hat.T @ g_n)) * g_n
    # equation 31
    ggT = np.einsum("ij,kj->jik", g_n, g_n)
    s_before_sum = (1 / (p_hat.T @ g_n) ** 2) * ggT.T
    s = 1 / n * np.sum(s_before_sum, axis=2)

    # equation 30
    v_n = s[:, :, np.newaxis] - s_before_sum

    # equation 29 + 32
    lmbda = 1e-10  # Thikonov regularization; TODO size? 1e-20 is too small
    q = -np.linalg.inv(s + lmbda * np.eye(d))

    # equation 56 + 57
    part1 = 1 / n * np.sum(np.einsum("ij,jki->ik", (s_n.T @ q.T), v_n), axis=0)

    # equation 62
    b = np.sum(np.einsum("ik,jk->ijk", q @ s_n, q @ s_n), axis=2)

    # equation 64
    u = np.sum(ggT * b, axis=(1, 2))

    # equation 63
    ab = np.sum(1 / (p_hat.T @ g_n) ** 3 * g_n * u, axis=1)

    # equation 58 with explanations after the equation
    part2 = 1 / (2 * n**2) * ab

    # equation 28
    a_hat_ = 1 / n * q @ (part1 - part2)

    return a_hat_


def _em_reconstruction(p_hat, g_n, n):
    """
    Algorithm 1 of Murakami et al. (2018) [1].

    Args:
        p_hat: The estimated frequencies. A vector of size d.
        g_n: The mechanism matrix. A matrix of size d x n.
        n: The number of samples. An integer.

    Returns: The reconstructed frequencies.
    """
    # Expectation maximization
    max_iter = 1000  # TODO: how many iterations?
    for _ in range(max_iter):
        denominator = g_n * p_hat
        numerator = np.sum(denominator, axis=1)
        p_hat_new = 1 / n * np.sum(denominator / numerator[:, None], axis=0)

        diff = np.abs(p_hat - p_hat_new)

        p_hat = p_hat_new
        if np.all(diff < 1e-5):  # convergence criterion TODO: how small should this be?
            break

    return p_hat


class Murakami2018(FrequencyOracle):
    def _find_alpha(self, p_hat, g_n, n, rng):
        """
        Run the whole algorithm again on data simulated using the estimated p_hat and try a range of alpha values to
        find the one with the smallest mean squared error.

        Args:
            p_hat: The estimated frequencies. A vector of size d.
            g_n: The mechanism matrix. A matrix of size d x n.
            n: The number of samples. An integer.
            rng: A numpy random number Generator.

        Returns: The best alpha value.
        """
        alphas = [math.pow(10, i) for i in range(-10, 10)]

        best_error = np.inf
        best_alpha = None

        # generate data x' with probabilities from p_hat
        x_ = rng.choice(len(p_hat), n, p=p_hat)

        # simulate the mechanism
        u_ = d_rr_response(x_, self.eps, self.domain_size, self.rng)
        q_hat = np.bincount(u_, minlength=self.domain_size) / n

        # find p_hat with EM
        p_hat_ = _em_reconstruction(q_hat, g_n, n)

        # find a_hat
        a_hat_ = _find_a_hat(p_hat_, g_n.T, n)

        for alpha in alphas:
            # calculate p_tilde
            p_tilde = p_hat_ - alpha * a_hat_

            # calculate MSE
            error = np.mean((p_tilde - p_hat) ** 2)

            if error < best_error:
                best_error = error
                best_alpha = alpha

        return best_alpha

    def response(self, t: np.array) -> np.array:
        """
        Run the d-RR mechanism and return the response for all clients.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The disturbed data.
        """
        return d_rr_response(t, self.eps, self.domain_size, self.rng)

    def frequencies(self, z: np.array) -> np.array:
        """
        Estimate the frequencies of the disturbed data.

        Args:
            z: The disturbed data.

        Returns: The estimated frequencies (vector of size domain_size).
        """
        n = len(z)
        q_hat = np.bincount(z, minlength=self.domain_size) / n

        mechanism_matrix = np.ones((self.domain_size, self.domain_size)) * (
            1 / (self.domain_size - 1 + np.exp(self.eps))
        )
        np.fill_diagonal(mechanism_matrix, np.exp(self.eps) / (self.domain_size - 1 + np.exp(self.eps)))

        # construct a d x n matrix by selecting columns from the mechanism matrix based on the indices in u
        g_n = mechanism_matrix[z]

        # Find p_hat with EM
        p_hat = _em_reconstruction(q_hat, g_n, n)

        # Note: g_n in the paper is a vector of size d, but here we store all g_n in a matrix of size d x n
        a_hat_ = _find_a_hat(p_hat, g_n.T, n)

        # Find a value for alpha
        alpha = self._find_alpha(p_hat, g_n, n, self.rng)

        p_tilde = p_hat - alpha * a_hat_

        # Normalized Decoder (section 3.2): truncate negative elements to 0 and renormalize
        p_tilde = np.maximum(p_tilde, 0)
        p_tilde /= np.sum(p_tilde)

        return p_tilde
