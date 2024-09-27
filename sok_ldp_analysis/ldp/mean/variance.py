from abc import ABC, abstractmethod
from typing import Tuple, NewType, Type, Union

import numpy as np

from sok_ldp_analysis.ldp.mean.mean import Mean1D, MeanMultiDim

Mean1DType = NewType("Mean1DType", Mean1D)
MeanMultiDimType = NewType("MeanMultiDimType", MeanMultiDim)


def _get_squared_range(a, b):
    """
    Given range [a, b], return the range the squared values will be in.

    Args:
        a: The lower bound of the input range.
        b: The upper bound of the input range.

    Returns: The range the squared values will be in.
    """
    if a > 0:
        return a**2, b**2
    else:
        max_val = max(a**2, b**2)
        return 0, max_val


class MeanAndVariance(ABC):
    """
    Abstract class for 1-dimensional mean and variance estimation.
    """

    def __init__(
        self,
        base_method: Union[Type[Mean1DType], Type[MeanMultiDimType]],
        epsilon: float = 1,
        split_ratio: float = 0.5,
        input_range: Tuple[float, float] = (0, 1),
        rng: np.random.Generator = None,
    ):
        """
        Initialize the mean and variance estimator.

        This estimator uses a split of the privacy budget to estimate the mean and variance separately.

        Args:
            base_method: The base method to use for the mean estimation.
            epsilon: The total privacy budget epsilon.
            split_ratio: The ratio of the privacy budget/number of participants to use for the mean estimation (the
            rest is used for the variance estimation). Values between 0 and 1 (exclusive) are allowed.
            input_range: The range of the input data.
            rng: A numpy random number Generator. Use None for the default random number generator.
        """
        assert 0 < split_ratio < 1, "The split ratio must be between 0 and 1 (exclusive)."

        self.base_method = base_method
        if issubclass(base_method, MeanMultiDim):
            self.multi_dim_method = True
        else:
            self.multi_dim_method = False

        self.epsilon = epsilon
        self.split_ratio = split_ratio
        self.input_range = input_range
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    @abstractmethod
    def estimate_mean_and_variance(self, t: np.array) -> Tuple[float, float]:
        """
        Run the mechanism on t and t**2 and estimate the mean and variance.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The estimated mean and variance.
        """
        raise NotImplementedError


class MeanAndVarianceSplitEps(MeanAndVariance):
    """
    Mean and variance estimator that splits the privacy budget between the mean and variance estimation.
    """

    def estimate_mean_and_variance(self, t: np.array) -> Tuple[float, float]:
        """
        Run the mechanism on t and t**2 and estimate the mean and variance.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The estimated mean and variance.
        """
        epsilon1 = self.epsilon * self.split_ratio
        epsilon2 = self.epsilon * (1 - self.split_ratio)

        n = len(t)

        if self.multi_dim_method:
            t = t.reshape(-1, 1)

        mean_estimator = self.base_method(eps=epsilon1, input_range=self.input_range, rng=self.rng)

        squared_data = t**2

        squared_range = _get_squared_range(self.input_range[0], self.input_range[1])
        variance_estimator = self.base_method(eps=epsilon2, input_range=squared_range, rng=self.rng)

        mean_x = mean_estimator.estimate_mean(t)
        mean_squared_x = variance_estimator.estimate_mean(squared_data)

        if self.multi_dim_method:
            mean_x = mean_x[0]
            mean_squared_x = mean_squared_x[0]

        variance = (n / (n - 1)) * (mean_squared_x - mean_x**2)

        return mean_x, variance


class MeanAndVarianceSplitN(MeanAndVariance):
    """
    Mean and variance estimator that splits the data between the mean and variance estimation.
    """

    def estimate_mean_and_variance(self, t: np.array) -> Tuple[float, float]:
        """
        Run the mechanism on t and t**2 and estimate the mean and variance.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The estimated mean and variance.
        """
        n = len(t)

        mean_estimator = self.base_method(eps=self.epsilon, input_range=self.input_range, rng=self.rng)

        squared_range = _get_squared_range(self.input_range[0], self.input_range[1])
        variance_estimator = self.base_method(eps=self.epsilon, input_range=squared_range, rng=self.rng)

        n1 = int(n * self.split_ratio)

        data = t[:n1]
        squared_data = t[n1:] ** 2

        if self.multi_dim_method:
            data = data.reshape(-1, 1)
            squared_data = squared_data.reshape(-1, 1)

        mean_x = mean_estimator.estimate_mean(data)
        mean_squared_x = variance_estimator.estimate_mean(squared_data)

        if self.multi_dim_method:
            mean_x = mean_x[0]
            mean_squared_x = mean_squared_x[0]

        variance = (n / (n - 1)) * (mean_squared_x - mean_x**2)

        return mean_x, variance


class MeanAndVarianceSplitNSequential(MeanAndVariance):
    """
    Mean and variance estimator that splits the data between the mean and variance estimation and runs the mechanism
    sequentially (using the mean to estimate the variance).
    """

    def estimate_mean_and_variance(self, t: np.array) -> Tuple[float, float]:
        """
        Run the mechanism on the first part of t to estimate the mean, then subtract the mean and run the mechanism
        on the second part of t to estimate the variance.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The estimated mean and variance.
        """
        n = len(t)

        mean_estimator = self.base_method(eps=self.epsilon, input_range=self.input_range, rng=self.rng)

        n1 = int(n * self.split_ratio)

        data1 = t[:n1]
        data2 = t[n1:]

        if self.multi_dim_method:
            data1 = data1.reshape(-1, 1)
            data2 = data2.reshape(-1, 1)

        mean_x = mean_estimator.estimate_mean(data1)

        if self.multi_dim_method:
            mean_x = mean_x[0]

        data2 = (data2 - mean_x) ** 2
        range2 = _get_squared_range(self.input_range[0] - mean_x, self.input_range[1] - mean_x)

        variance_estimator = self.base_method(eps=self.epsilon, input_range=range2, rng=self.rng)

        mean_squared_diff = variance_estimator.estimate_mean(data2)

        if self.multi_dim_method:
            mean_squared_diff = mean_squared_diff[0]

        # The sample variance is 1/(n-1) * sum((x_i - mean_x)^2) - the mean is 1/n * sum, so we need to correct it
        variance = (n / (n - 1)) * mean_squared_diff

        return mean_x, variance
