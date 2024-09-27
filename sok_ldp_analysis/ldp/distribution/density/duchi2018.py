"""
Density estimation method by Duchi et al. 2018 [1].

References:
[1] J. C. Duchi, M. I. Jordan, and M. J. Wainwright, “Minimax Optimal Procedures for Locally Private
Estimation,” Journal of the American Statistical Association, vol. 113, no. 521, pp. 182–201, Jan. 2018,
doi: 10.1080/01621459.2017.1389735.
"""

import math
from typing import Tuple

import numpy as np

from sok_ldp_analysis.ldp.distribution.density.density import DensityEstimator, Density
from sok_ldp_analysis.ldp.mean.duchi2018 import duchi_linf_mechanism


def _trigonometric_basis(idx: int) -> callable:
    """
    Create a trigonometric basis function.

    Equation (34) from [1].

    Args:
        idx: The index of the basis function.

    Returns: The basis function.
    """
    if idx == 0:
        return lambda x: 1
    elif idx % 2 == 0:
        return lambda x: math.sqrt(2) * np.cos(np.pi * idx * x)
    else:
        return lambda x: math.sqrt(2) * np.sin(np.pi * (idx - 1) * x)


class DuchiDensity(Density):
    """
    Density estimation based on the mechanism response by Duchi et al. 2018 [1].
    """

    def __init__(self, k, input_range: Tuple[float, float], z):
        """
        Initialize the density function.

        Args:
            k: The number of basis functions.
            input_range: The range of the input data.
            z: The mechanism response of n participants.
        """
        super().__init__(input_range, mechanism_range=(0, 1))
        self.k = k
        self.n = z.shape[0]

        self.phi = [_trigonometric_basis(i) for i in range(k)]
        self.z_j = np.sum(z, axis=0)

    def _density(self, x):
        """
        Compute the density function.

        This is equation (39) from [1], but reorganized to allow the pre-computation of the sum of z_j.

        Args:
            x: The input data in the range of the mechanism.

        Returns: The estimated density at x.
        """

        result = 0
        for j in range(self.k):
            result += self.z_j[j] * self.phi[j](x)

        return result / self.n


class Duchi2018(DensityEstimator):
    """
    Density estimation method by Duchi et al. 2018 [1].
    """

    def __init__(self, eps: float, input_range: tuple, k: int, rng: np.random.Generator = None):
        """
        Initialize the density estimator.

        Args:
            eps: The privacy budget epsilon.
            input_range: The range of the input data.
            k: The number of basis functions.
            rng: A numpy random number Generator.
        """
        super().__init__(eps, input_range, mechanism_range=(0, 1), rng=rng)
        self.k = k
        self.phi = [_trigonometric_basis(i) for i in range(k)]

    def response(self, x):
        """
        Apply the mechanism to the data of n participants in the range [0, 1].

        Args:
            x: The private data of n participants.

        Returns: The response of the mechanism of size n x k.
        """

        phi_x = np.array([[f(x_i) for f in self.phi] for x_i in x])

        # Apply the mechanism
        phi_max = math.sqrt(2)
        b = phi_max * math.sqrt(self.k) * (math.exp(1) + 1) / (math.exp(1) - 1)
        phi_z = duchi_linf_mechanism(phi_x, 1, phi_max, np.random.default_rng(), b)

        return phi_z

    def density(self, z) -> Density:
        """
        Estimate the density function based on the mechanism response.

        Args:
            z: The mechanism response of n participants.

        Returns: The estimated density function.
        """
        return DuchiDensity(self.k, self.input_range, z)
