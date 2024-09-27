"""Set of LDP mean estimation implementations from Wang et al. 2019 [1].

[1] N. Wang et al., “Collecting and Analyzing Multidimensional Data with Local Differential Privacy,” presented
at the 2019 IEEE 35th International Conference on Data Engineering (ICDE), IEEE Computer Society, Apr. 2019,
pp. 638-649. doi: 10.1109/ICDE.2019.00063.
"""


import math
from typing import Tuple

import numpy as np

from sok_ldp_analysis.ldp.mean.mean import Mean1D, MeanMultiDim


class Wang2019Duchi1D(Mean1D):
    """Mean estimation method for 1-dimensional input attributed to Duchi et al. by Wang et al. [1].

    Based on pseudocode from the full paper of [1]: https://arxiv.org/abs/1907.00782
    """

    def mechanism(self, t: np.array) -> np.array:
        """
        Mean estimation mechanism for 1-dimensional input (of multiple clients).

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The disturbed data.
        """
        return _wang2019_duchi_1d(_scale_input_data(t, self.input_range), self.eps, self.rng)

    def mean(self, u: np.array) -> float:
        """Compute the mean of the disturbed data.

        The data is unbiased, so we can simply calculate the mean without any correction.

        Args:
            u: The disturbed data.

        Returns: The mean of the disturbed data.
        """
        return _scale_output_mean(np.mean(u), self.input_range)


class Wang2019Piecewise1D(Mean1D):
    """Wang et al.'s piecewise sampling method for 1-dimensional input (of multiple clients) [1].

    [1] N. Wang et al., “Collecting and Analyzing Multidimensional Data with Local Differential Privacy,” presented
    at the 2019 IEEE 35th International Conference on Data Engineering (ICDE), IEEE Computer Society, Apr. 2019,
    pp. 638-649. doi: 10.1109/ICDE.2019.00063.
    """

    def mechanism(self, t: np.array) -> np.array:
        """Piecewise sampling method for 1-dimensional input (of multiple clients).

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The disturbed data.
        """
        return _piecewise_1d(_scale_input_data(t, self.input_range), self.eps, self.rng)

    def mean(self, u: np.array) -> float:
        """Compute the mean of the disturbed data.

        The data is unbiased, so we can simply calculate the mean without any correction.

        Args:
            u: The disturbed data.

        Returns: The mean of the disturbed data.
        """
        return _scale_output_mean(np.mean(u), self.input_range)


class Wang2019Hybrid1D(Mean1D):
    """Wang et al.'s hybrid sampling method for 1-dimensional input (of multiple clients) [1].

    [1] N. Wang et al., “Collecting and Analyzing Multidimensional Data with Local Differential Privacy,” presented
    at the 2019 IEEE 35th International Conference on Data Engineering (ICDE), IEEE Computer Society, Apr. 2019,
    pp. 638-649. doi: 10.1109/ICDE.2019.00063.
    """

    def mechanism(self, t: np.array) -> np.array:
        """Hybrid sampling method for 1-dimensional input (of multiple clients).

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The disturbed data.
        """
        return _hybrid_1d(_scale_input_data(t, self.input_range), self.eps, self.rng)

    def mean(self, u: np.array) -> float:
        """Compute the mean of the disturbed data.

        The data is unbiased, so we can simply calculate the mean without any correction.

        Args:
            u: The disturbed data.

        Returns: The mean of the disturbed data.
        """
        return _scale_output_mean(np.mean(u), self.input_range)


class Wang2019(MeanMultiDim):
    """Wang et al.'s hybrid sampling method for multi-dimensional input (of multiple clients) [1].

    [1] N. Wang et al., “Collecting and Analyzing Multidimensional Data with Local Differential Privacy,” presented
    at the 2019 IEEE 35th International Conference on Data Engineering (ICDE), IEEE Computer Society, Apr. 2019,
    pp. 638-649. doi: 10.1109/ICDE.2019.00063.
    """

    def __init__(self, eps: float, input_range: Tuple[float, float], rng: np.random.Generator = None, k: int = None):
        """Initialize the Wang2019 mechanism.

        Args:
            eps: The privacy budget epsilon.
            input_range: The range of the input data.
            rng: A numpy random number Generator.
            k: The number of dimensions to sample from. If None, k is set optimally.
        """
        super().__init__(eps, input_range, rng)
        self.k = k

    def mechanism(self, input_data: np.array) -> np.array:
        """Hybrid sampling method for multi-dimensional input (of multiple clients).

        Args:
            input_data: The private data of n clients (matrix of size n x d)

        Returns: The disturbed data.
        """
        t = _scale_input_data(input_data, self.input_range)

        t_ = np.zeros_like(t)

        n = t.shape[0]
        d = t.shape[1]

        if self.k is not None:
            k = self.k
        else:
            k = max(1, min(d, math.floor(self.eps / 2.5)))

        # Sample k values from {1,2...d} without replacement for each client
        idx = np.vstack([self.rng.permutation(d)[:k] for _ in range(t.shape[0])])

        for i in range(k):
            t_[np.arange(n), idx[:, i]] = d / k * _hybrid_1d(t[np.arange(n), idx[:, i]], self.eps / k, self.rng)

        return t_

    def mean(self, u: np.array) -> np.array:
        """Compute the mean of the disturbed data.

        The data is unbiased, so we can simply calculate the mean without any correction.

        Args:
            u: The disturbed data.

        Returns: The mean of the disturbed data.
        """
        return _scale_output_mean(np.mean(u, axis=0), self.input_range)


def _scale_input_data(input_data: np.array, input_range: Tuple[float, float]) -> np.array:
    """
    Scale the input data to be in range [-1, 1].

    Args:
        input_data: The input data.
        input_range: The range of the input data.

    Returns: The scaled input data.
    """
    # Scale data to be in range [-1, 1]
    range_size = input_range[1] - input_range[0]
    x = 2 * (input_data - input_range[0]) / range_size - 1

    assert np.all(np.abs(x) <= 1 + 1e-10)

    return x


def _scale_output_mean(mean: np.array, input_range: Tuple[float, float]) -> np.array:
    """
    Scale the output mean to be in the original range.

    Args:
        mean: The mean in range [-1, 1].
        input_range: The range of the input data.

    Returns: The scaled mean.
    """
    # Return the mean in the original range
    return (mean + 1) / 2 * (input_range[1] - input_range[0]) + input_range[0]


def _piecewise_1d(t: np.array, eps: float, rng: np.random.Generator = None) -> np.array:
    """Piecewise sampling method for 1-dimensional input (of multiple clients).

    Method introduced by https://arxiv.org/abs/1907.00782

    Args:
        t: The private data of n clients (vector of size n)
        eps: The privacy budget epsilon.
        rng: A numpy random number Generator

    Returns: The disturbed data.

    """
    assert t.ndim == 1

    if rng is None:
        rng = np.random.default_rng()

    c = (math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)

    left = ((c + 1) / 2) * t - (c - 1) / 2
    right = left + c - 1

    x = rng.uniform(size=t.shape)

    ret = np.zeros_like(t)
    thres = (math.exp(eps / 2)) / (math.exp(eps / 2) + 1)

    # if x < thres
    ret[x < thres] = rng.uniform(left[x < thres], right[x < thres])

    # else: sample uniformly from the disjoint intervals [-C, l) and (r, C]

    # Calculate the size of both intervals
    size_l = np.abs(left[x >= thres] + c)
    size_r = np.abs(c - right[x >= thres])

    # Decide which interval to sample from
    u = rng.binomial(n=1, p=size_l / (size_l + size_r), size=size_l.shape)

    # if t == 1 or t == -1, rng.uniform crashes because the sampling interval is empty
    left[t == -1] = -c + 1e-10
    right[t == 1] = c - 1e-10

    # Create samples from both intervals
    samples_l = rng.uniform(-c, left[x >= thres])
    samples_r = rng.uniform(right[x >= thres], c)

    # Select samples from the correct interval
    ret[x >= thres] = samples_l * u + samples_r * (1 - u)

    return ret


def _hybrid_1d(t: np.array, eps: float, rng: np.random.Generator = None) -> np.array:
    """Hybrid sampling method for 1-dimensional input (of multiple clients).

    Method introduced by https://arxiv.org/abs/1907.00782

    Args:
        t: The private data of n clients (vector of size n)
        eps: The privacy budget epsilon.
        rng: A numpy random number Generator

    Returns: The disturbed data.

    """
    assert t.ndim == 1

    if rng is None:
        rng = np.random.default_rng()

    alpha = 0 if eps <= 0.61 else 1 - math.exp(-eps / 2)

    ret = np.zeros_like(t)
    x = rng.uniform(size=t.shape)

    ret[x <= alpha] = _piecewise_1d(t[x <= alpha], eps, rng)
    ret[x > alpha] = _wang2019_duchi_1d(t[x > alpha], eps, rng)

    return ret


def _wang2019_duchi_1d(t: np.array, eps: float, rng: np.random.Generator = None) -> np.array:
    """Mean estimation method for 1-dimensional input (of multiple clients).

    Attributed to Duchi et al. by Wang et al. [1] .

    [1] N. Wang et al., “Collecting and Analyzing Multidimensional Data with Local Differential Privacy,” presented
    at the 2019 IEEE 35th International Conference on Data Engineering (ICDE), IEEE Computer Society, Apr. 2019,
    pp. 638-649. doi: 10.1109/ICDE.2019.00063.

    Based on pseudocode from the full paper of [1]: https://arxiv.org/abs/1907.00782

    Args:
        t: The private data of n clients (vector of size n)
        eps: The privacy budget epsilon.
        rng: A numpy random number Generator

    Returns: The disturbed data.

    """
    # Only works for 1-dimensional inputs
    assert t.ndim == 1

    if rng is None:
        rng = np.random.default_rng()

    p = ((math.exp(eps) - 1) / (2 * math.exp(eps) + 2)) * t + 1 / 2
    # Sample from a Bernoulli distribution (Binomial with n=1)
    u = rng.binomial(n=1, p=p, size=t.shape)

    ret_value = (math.exp(eps) + 1) / (math.exp(eps) - 1)
    ret = np.ones_like(t) * ret_value

    ret[u != 1] *= -1

    return ret
