"""Abstract base classes for mean estimation under local differential privacy."""
import math
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class GaussianMean1D(ABC):
    """Abstract class for 1-dimensional mean estimation of Gaussian data."""

    def __init__(self, eps: float, rng: np.random.Generator = None):
        """Initialize the mean estimator.

        Args:
            eps: The privacy budget epsilon.
            rng: A numpy random number Generator.
        """
        self.eps = eps
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def reseed_rng(self, seed: int):
        """
        Reseed the random number generator.

        Args:
            seed: The seed to use.
        """
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def estimate_mean(self, t: np.array) -> float:
        """
        Run the mechanism and estimate the mean (potentially using multiple rounds and post-processing).

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The estimated mean.
        """


class Mean1D(ABC):
    """Abstract class for 1-dimensional mean estimation."""

    def __init__(self, eps: float, input_range: Tuple[float, float], rng: np.random.Generator = None):
        """Initialize the mean estimator.

        Args:
            eps: The privacy budget epsilon.
            rng: A numpy random number Generator.
        """
        self.eps = eps
        self.input_range = input_range
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def reseed_rng(self, seed: int):
        """
        Reseed the random number generator.

        Args:
            seed: The seed to use.
        """
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def mechanism(self, t: np.array) -> np.array:
        """
        Eps-LDP mechanism for disturbing 1-dimensional input (of multiple clients).

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The disturbed data.
        """

    @abstractmethod
    def mean(self, u: np.array) -> float:
        """
        Compute the mean of the disturbed data.

        Args:
            u: The disturbed data.

        Returns: The mean of the disturbed data.
        """

    def estimate_mean(self, t: np.array) -> float:
        """
        Run the mechanism and estimate the mean (potentially using multiple rounds and post-processing).

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The estimated mean.
        """
        u = self.mechanism(t)
        return self.mean(u)


class MeanMultiDim(ABC):
    """Abstract class for multidimensional mean estimation."""

    def __init__(self, eps: float, input_range: Tuple[float, float], rng: np.random.Generator = None):
        """Initialize the mean estimator.

        Args:
            eps: The privacy budget epsilon.
            rng: A numpy random number Generator.
        """
        self.eps = eps
        self.input_range = input_range
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def reseed_rng(self, seed: int):
        """
        Reseed the random number generator.

        Args:
            seed: The seed to use.
        """
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def mechanism(self, t: np.array) -> np.array:
        """
        Eps-LDP mechanism for disturbing d-dimensional input (of multiple clients).

        Args:
            t: The private data of n clients. An array of size (n,d).

        Returns: The disturbed data.
        """

    @abstractmethod
    def mean(self, u: np.array) -> np.array:
        """
        Compute the mean of the disturbed data.

        Args:
            u: The disturbed data. An array of size (n,d).

        Returns: The mean of the disturbed data.
        """

    def estimate_mean(self, t: np.array) -> np.array:
        """
        Run the mechanism and estimate the mean.

        Args:
            t: The private data of n clients. An array of size (n,d).

        Returns: The estimated mean.
        """
        u = self.mechanism(t)
        return self.mean(u)


class MeanMultiDimWrapperLikeNguyen(MeanMultiDim):
    """Wrapper for 1-dimensional mean estimation of multidimensional data.

    This class wraps a 1-dimensional mean estimator to work on multidimensional data. It selects one dimension at
    random for each client and applies the 1-dimensional mean estimator to the selected dimensions. This is inspired
    by the Harmony method according to Nguyên et al., 2016 [1].

    [1] T. T. Nguyên, X. Xiao, Y. Yang, S. C. Hui, H. Shin, and J. Shin, “Collecting and Analyzing Data from Smart Device
    Users with Local Differential Privacy.” arXiv, Jun. 16, 2016. doi: 10.48550/arXiv.1606.05053.
    """

    def __init__(self, base_method: Mean1D):
        """Initialize the mean estimator.

        Args:
            base_method: The base method to use for the 1-dimensional mean estimation.
        """
        super().__init__(base_method.eps, base_method.input_range, base_method.rng)
        self.base_method = base_method

    def reseed_rng(self, seed: int):
        """
        Reseed the random number generator.

        Args:
            seed: The seed to use.
        """
        self.base_method.reseed_rng(seed)
        self.rng = self.base_method.rng

    def mechanism(self, t: np.array) -> np.array:
        """
        Run the mechanism on each dimension separately.

        Args:
            t: The private data of n clients. An array of size (n,d).

        Returns: The disturbed data.
        """
        n = t.shape[0]
        d = t.shape[1]

        # Uniformly sample a random index for each client (using self.rng)
        idx = self.rng.integers(low=0, high=d, size=n)

        u = np.zeros(t.shape)
        u[np.arange(n), idx] = self.base_method.mechanism(t[np.arange(n), idx]) * d
        # TODO: figure out why we need to multiply by d here

        return u

    def mean(self, u: np.array) -> np.array:
        """
        Compute the mean of the disturbed data.

        Args:
            u: The disturbed data. A tuple of two arrays: the disturbed data and the indices of the selected dimensions.

        Returns: The mean of the disturbed data.
        """
        d = u.shape[1]
        mean = np.zeros(d)
        for i in range(d):
            mean[i] = self.base_method.mean(u[:, i])
        return mean


class MeanMultiDimWrapperLikeWang(MeanMultiDim):
    """Wrapper for 1-dimensional mean estimation of multidimensional data.

    This class wraps a 1-dimensional mean estimator to work on multidimensional data. It selects k dimensions at
    random for each client and applies the 1-dimensional mean estimator to the selected dimensions. This is inspired
    by Wang et al.'s hybrid sampling method for multi-dimensional input (of multiple clients) [1].

    [1] N. Wang et al., “Collecting and Analyzing Multidimensional Data with Local Differential Privacy,” presented
    at the 2019 IEEE 35th International Conference on Data Engineering (ICDE), IEEE Computer Society, Apr. 2019,
    pp. 638-649. doi: 10.1109/ICDE.2019.00063.

    """

    def __init__(self, base_method: Mean1D):
        """Initialize the mean estimator.

        Args:
            base_method: The base method to use for the 1-dimensional mean estimation.
        """
        super().__init__(base_method.eps, base_method.input_range, base_method.rng)
        self.base_method = base_method

    def reseed_rng(self, seed: int):
        """
        Reseed the random number generator.

        Args:
        seed: The seed to use.
        """
        self.base_method.reseed_rng(seed)
        self.rng = self.base_method.rng

    def mechanism(self, t: np.array) -> np.array:
        """
        Run the mechanism on each dimension separately.

        Args:
            t: The private data of n clients. An array of size (n,d).

        Returns: The disturbed data.
        """
        n = t.shape[0]
        d = t.shape[1]

        k = max(1, min(d, math.floor(self.eps / 2.5)))
        print(k)

        # update eps as self.eps / k
        mechanism = self.base_method.__class__(
            eps=self.eps / k, input_range=self.base_method.input_range, rng=self.base_method.rng
        )

        # Sample k values from {1,2...d} without replacement for each client
        idx = np.vstack([self.rng.permutation(d)[:k] for _ in range(t.shape[0])])

        t_ = np.zeros_like(t)
        for i in range(k):
            t_[np.arange(n), idx[:, i]] = d / k * mechanism.mechanism(t[np.arange(n), idx[:, i]])

        return t_

    def mean(self, u: np.array) -> np.array:
        """
        Compute the mean of the disturbed data.

        Args:
            u: The disturbed data. An array of size (n,d).

        Returns: The mean of the disturbed data.
        """
        d = u.shape[1]
        mean = np.zeros(d)
        for i in range(d):
            mean[i] = self.base_method.mean(u[:, i])
        return mean
