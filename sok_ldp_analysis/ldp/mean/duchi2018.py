"""Implementation of Duchi et al.'s sampling method for d-dimensional input.

More information can be found in section 4.2 of the paper:

J. C. Duchi, M. I. Jordan, and M. J. Wainwright, “Minimax Optimal Procedures for Locally Private Estimation,”
Journal of the American Statistical Association, vol. 113, no. 521, pp. 182-201, Jan. 2018,
doi: 10.1080/01621459.2017.1389735.
"""

import math
from typing import Tuple, Optional

import numpy as np
from scipy.special import binom

from sok_ldp_analysis.ldp.mean.mean import MeanMultiDim


def _sample_from_sphere(n: int, d: int, scale: float, rng: np.random.Generator = None) -> np.array:
    """Sample n points from a d-dimensional sphere with radius scale.

    According to https://mathworld.wolfram.com/HyperspherePointPicking.html

    Args:
        n: The number of points to sample.
        d: The dimension of the sphere.
        scale: The radius of the sphere.
        rng: A numpy random number Generator.

    Returns: The sampled points. An array of size (n,d). Each point fulfills ||x||_2 = scale.

    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample from a d-dimensional Gaussian distribution
    x = rng.normal(size=(n, d))

    # Normalize to unit length
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    # Scale to radius
    x *= scale

    return x


def _sample_from_hypercube(n: int, d: int, scale: float, rng: np.random.Generator = None) -> np.array:
    """Sample n corners from a d-dimensional hypercube with radius scale.

    Args:
        n: The number of points to sample.
        d: The dimension of the hypercube.
        scale: The radius of the hypercube. ||x||_inf <= scale for all x.
        rng: A numpy random number Generator.

    Returns: The sampled points. An array of size (n,d).

    """
    if rng is None:
        rng = np.random.default_rng()

    # Each point is from {-scale, scale}^d

    # Sample from a d-dimensional Bernoulli distribution to fix the signs
    x = rng.binomial(n=1, p=0.5, size=(n, d)) * 2 - 1

    # Scale to radius
    x *= scale

    return x


class Duchi2018(MeanMultiDim):
    """Duchi et al.'s sampling method for d-dimensional input.

    Data is assumed to be in an L_2 ball of radius r, i.e. ||x||_2 <= r for all x.
    """

    def __init__(
        self, eps: float, input_range: Tuple[float, float], r: float = 1, rng: Optional[np.random.Generator] = None
    ):
        """Initialize the Duchi2018 mechanism.

        Args:
            eps: The privacy budget epsilon.
            r: The radius of the ball the data is assumed to be in.
            rng: A numpy random number Generator.
        """
        super().__init__(eps, input_range, rng)
        self.r = r

    def mechanism(self, input_data: np.array) -> np.array:
        """Mean Estimation in d dimensions according to Duchi et al. 2018, section 4.2.

        Args:
            input_data: The private d-dimensional data of n clients (matrix of size n x d)

        Returns: The disturbed data.
        """
        # Check input
        assert input_data.ndim == 2

        d = input_data.shape[1]

        # Project data to [-1, 1]
        range_size = self.input_range[1] - self.input_range[0]
        x = 2 * (input_data - self.input_range[0]) / range_size - 1

        assert np.all(np.abs(x) <= 1 + 1e-10)

        # Scale the unit cube to fit inside the unit ball
        x /= math.sqrt(d)

        # Rescale the ball to the correct radius
        x *= self.r

        # Validate that data is in the ball
        assert np.all(np.linalg.norm(x, axis=1) <= self.r + 1e-10)

        # Bernoulli probability
        p = math.exp(self.eps) / (math.exp(self.eps) + 1)

        # Sample from a Bernoulli distribution (Binomial with n=1)
        t = self.rng.binomial(n=1, p=p, size=x.shape[0])

        norm = np.linalg.norm(x, axis=1, keepdims=True)
        # avoid division by zero in the next step
        norm[norm == 0] = 1e-20

        # Construct a random vector x_tilde
        x_tilde = self.r * x / norm

        # Set x_tilde to -x_tilde with probability 1/2 + ||x||_2 / (2r)
        # Sample from {0,1}
        x_tilde_multiplier = self.rng.binomial(
            n=1, p=1 / 2 + np.linalg.norm(x, axis=1, keepdims=True) / (2 * self.r), size=x.shape
        )
        # Turn {0,1} into {-1,1}
        x_tilde_multiplier = 2 * x_tilde_multiplier - 1
        x_tilde *= x_tilde_multiplier

        # Calculate B
        b = (
            self.r
            * ((math.exp(self.eps) + 1) / (math.exp(self.eps) - 1))
            * ((math.sqrt(math.pi)) / 2)
            * ((d * math.gamma(1 + (d - 1) / 2)) / math.gamma(1 + d / 2))
        )

        # Perform rejection sampling
        z = _rejection_sampling(self.rng, x, t, x_tilde, b)

        return z

    def mean(self, u: np.array) -> np.array:
        """Compute the mean of the disturbed data.

        Args:
            u: The disturbed data.

        Returns: The mean of the disturbed data.
        """
        mean = np.mean(u, axis=0)

        d = u.shape[1]

        range_size = self.input_range[1] - self.input_range[0]

        # Rescale the ball with radius r to the unit ball
        mean /= self.r

        # Rescale to the original unit cube
        mean *= math.sqrt(d)

        # Rescale to the original range
        mean += 1
        mean /= 2
        mean *= range_size
        mean += self.input_range[0]
        return mean


def _rejection_sampling(rng: np.random.Generator, x: np.array, t: np.array, x_tilde: np.array, b: float) -> np.array:
    """
    Perform rejection sampling as suggested by Duchi et al.
    """
    d = x.shape[1]

    # Perform rejection sampling
    z = np.zeros_like(x)

    rejected = np.ones(x.shape[0], dtype=bool)

    while np.any(rejected):
        # Sample one point for each rejected point
        z_ = _sample_from_sphere(n=np.sum(rejected), d=d, scale=b, rng=rng)

        # filter the points for which <z, x_tilde> >= 0 if T = 1 and <z, x_tilde> < 0 if T = 0
        # Create two masks to help with filtering
        # mask has the same size as z
        # sample_mask has the same size as z_
        mask = np.zeros_like(rejected)
        sample_mask = np.logical_and(
            np.einsum("ij,ij->i", z_, x_tilde[rejected]) >= 0, t[rejected] == 1
        ) | np.logical_and(np.einsum("ij,ij->i", z_, x_tilde[rejected]) <= 0, t[rejected] == 0)

        mask[rejected] = sample_mask

        # Store the accepted points in z
        z[rejected & mask] = z_[sample_mask]

        # Update the rejected points
        rejected[rejected & mask] = False

    return z


class Duchi2018LInf(MeanMultiDim):
    """Duchi et al.'s sampling method for d-dimensional input.

    Data is assumed to be in an L_inf ball of radius r, i.e. ||x||_inf <= r for all x.
    """

    def __init__(
        self, eps: float, input_range: Tuple[float, float], r: float = 1, rng: Optional[np.random.Generator] = None
    ):
        """Initialize the Duchi2018 mechanism.

        Args:
            eps: The privacy budget epsilon.
            r: The radius of the L_inf ball the data is assumed to be in.
            rng: A numpy random number Generator.
        """
        super().__init__(eps, input_range, rng)
        self.r = r

    def mechanism(self, input_data: np.array) -> np.array:
        # Project data to [-r, r]
        range_size = self.input_range[1] - self.input_range[0]
        x = 2 * (input_data - self.input_range[0]) / range_size - 1
        x *= self.r

        return duchi_linf_mechanism(x, self.eps, self.r, self.rng)

    def mean(self, u: np.array) -> np.array:
        mean = np.mean(u, axis=0)

        # Project data from [-r, r] to the original range
        range_size = self.input_range[1] - self.input_range[0]
        mean *= range_size / (2 * self.r)
        mean += (self.input_range[0] + self.input_range[1]) / 2
        return mean


def duchi_linf_mechanism(x, eps, r, rng, b_override=None):
    assert x.ndim == 2

    d = x.shape[1]

    # Validate that data is in the ball
    assert np.all(np.abs(x) <= r + 1e-10)

    # Sample from a bernoulli distribution
    x_tilde = (rng.binomial(n=1, p=0.5 + x / (2 * r), size=x.shape) * 2 - 1) * r

    if not b_override:
        # Calculate C
        if d % 2 == 0:
            c = binom(d - 1, d // 2) / (2 ** (d - 1) + 0.5 * binom(d, d // 2))
        else:
            c = binom(d - 1, d // 2) / (2 ** (d - 1))

        b = r * c * (math.exp(eps) + 1) / (math.exp(eps) - 1)
    else:
        b = b_override

    # Bernoulli probability
    p = math.exp(eps) / (math.exp(eps) + 1)

    # Sample from a Bernoulli distribution (Binomial with n=1)
    t = rng.binomial(n=1, p=p, size=x.shape[0])

    # Perform rejection sampling
    z = _rejection_sampling(rng, x, t, x_tilde, b)

    return z
