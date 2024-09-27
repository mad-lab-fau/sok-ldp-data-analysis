"""
Mean and confidence interval estimation for Gaussian distributions according to Gaboardi et al. 2019 [1].

[1] M. Gaboardi, R. Rogers, and O. Sheffet, “Locally Private Mean Estimation: $Z$-test and Tight Confidence
Intervals,” in Proceedings of the Twenty-Second International Conference on Artificial Intelligence and
Statistics, PMLR, Apr. 2019, pp. 2545-2554. Accessed: Jul. 07, 2023. [Online]. Available:
https://proceedings.mlr.press/v89/gaboardi19a.html
"""
import functools
import math
from typing import Tuple

import numpy as np
import scipy

from sok_ldp_analysis.ldp.mean.mean import GaussianMean1D


def _bit_flipping_mechanism(eps: float, x: np.array, partition: np.array, rng: np.random.Generator) -> np.array:
    """
    Bit-flipping mechanism.

    This method takes an array of values and a partition array as input.
    It replaces each value with a one-hot vector of length len(partition) with a 1 at the index of the bin the value
    belongs to and 0 everywhere else. Then, it flips each bit with probability exp(eps)/(exp(eps)+1).

    Args:
        x: input data
        partition: Boundaries of the bins: [-R, -R + sigma, -R + 2*sigma, ..., R - sigma, R]

    Returns
    -------
        Bit-flipped one-hot encoded data
    """
    # Replace each value with a one-hot vector of length len(partition) with a 1 at the index of the bin the value
    # belongs to and 0 everywhere else
    x_one_hot = np.zeros((len(x), len(partition) - 1))
    for i in range(len(partition) - 1):
        x_one_hot[:, i] = np.logical_and(partition[i] <= x, x < partition[i + 1]).astype(int)

    # Keep each bit with prob. exp(eps/2)/(exp(eps/2)+1); flip otherwise
    p = 1 - math.exp(eps / 2) / (math.exp(eps / 2) + 1)
    flipped_bits = rng.binomial(n=1, p=p, size=x_one_hot.shape)

    # Use the flipped bits to flip the bits in x_one_hot
    x_flipped = np.logical_xor(x_one_hot, flipped_bits).astype(int)

    return x_flipped


def _known_var(
    eps: float, delta: float, data: np.array, sigma: float, r: float, beta: float, rng: np.random.Generator
) -> Tuple[float, float, float]:
    """
    KnownVar algorithm from Gaboardi et al. 2019 [1, Algorithm 1].

    Args:
        data: input data
        sigma: known standard deviation of the data
        r: defining the range of the data [-r, r]
        beta: confidence level. The returned confidence interval is a (1-beta)-confidence interval.

    Returns
    -------
        Tuple containing the estimated mean and the lower and upper bound of the confidence interval
    """
    n = len(data)
    d = 2 * math.ceil(r / sigma) + 1
    n1 = round(800 * ((math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)) ** 2 * math.log(8 * d / beta))
    n2 = n - n1

    if n1 > n:
        return math.nan, math.nan, math.nan

    # Partition the input data
    data1 = data[:n1]
    data2 = data[n1:]

    # Partition [-R, R] into into d bins
    bins = np.linspace(-r, r, d)

    # Apply bit-flipping mechanism to data1
    data1_flipped = _bit_flipping_mechanism(eps, data1, bins, rng)

    # Calculate the corrected histogram
    corrector = (math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)
    bf_est = np.sum(corrector * (data1_flipped - 1 / (1 + math.exp(eps / 2))), axis=0)

    # Find the bin with the highest count
    j_star = np.argmax(bf_est) - math.ceil(r / sigma)

    # Calculate the interval around bin j_star
    diff = 2 * sigma + sigma * math.sqrt(2 * math.log(8 * n / beta))
    s1 = j_star * sigma - diff
    s2 = j_star * sigma + diff

    # Run steps 6-8 of KnownVar on data2
    return _steps6to8(eps, delta, data2, s1, s2, diff, sigma, r, beta, rng)


def _rr(eps: float, data: np.array, bool_fn, rng: np.random.Generator) -> np.array:
    """
    Randomized response mechanism.

    This method takes an array of values and a boolean function as input.
    It applies the boolean function to each value and flips the result with probability exp(eps)/(exp(eps)+1).

    Args:
        data: input data
        bool_fn: boolean function

    Returns
    -------
        Randomized response of the boolean function applied to the data
    """
    p = 1 - math.exp(eps) / (math.exp(eps) + 1)
    return np.logical_xor(bool_fn(data), rng.binomial(n=1, p=p, size=len(data))).astype(int)


def _bin_quant(
    eps: float,
    data: np.array,
    target_quantile: float,
    search_interval: Tuple[float, float],
    lmbda: float,
    rounds: int,
    rng: np.random.Generator,
) -> float:
    """
    Binary search for the target quantile; Algorithm 2 from Gaboardi et al. 2019 [1].

    The returned quantile is within lmbda of the target quantile with high probability.

    Args:
        data: input data
        target_quantile: target quantile
        search_interval: interval [q_min, q_max] in which to search for the target quantile
        lmbda: quantile estimation error.
        rounds: number of rounds

    Returns
    -------
        The estimated quantile.
    """
    total_n = len(data)
    group_n = int(total_n / rounds)  # Note: we may ignore up to rounds-1 data points here
    q_min, q_max = search_interval
    s1 = q_min
    s2 = q_max
    t = (s1 + s2) / 2

    for j in range(rounds):
        partial_data = data[j * group_n : (j + 1) * group_n]
        t = (s1 + s2) / 2

        bool_fn = functools.partial(lambda t_, x: x < t_, t)

        z = np.sum(_rr(eps, partial_data, bool_fn, rng)) / group_n
        if z > target_quantile + lmbda / 2:
            s2 = t
        elif z < target_quantile - lmbda / 2:
            s1 = t
        else:
            break

    return t


def _unk_var(
    eps: float,
    delta: float,
    data: np.array,
    sigma_interval: Tuple[float, float],
    r: float,
    beta: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """UnkVar algorithm from Gaboardi et al. 2019 [1, Algorithm 3].

    The algorithm assumes that the data is from a Gaussian distribution, the unknown standard deviation of the data
    is in the range [sigma_min, sigma_max] and the unknown mean of the data is in the range [-r, r].

    Args:
        data: input data
        sigma_interval: interval [sigma_min, sigma_max] in which the standard deviation of the data is assumed to
                        be. sigma_min > 0, sigma_max > sigma_min, sigma_max < 2r.
        r: defining the range of the data [-r, r]
        beta: confidence level. The returned confidence interval is a (1-beta)-confidence interval.

    Returns
    -------
        Tuple containing the estimated mean and the lower and upper bound of the confidence interval
    """
    sigma_min, sigma_max = sigma_interval

    # Calculate the number of rounds for estimating the median and the standard deviation
    t_med = math.ceil(math.log2(8 * r / sigma_min))
    t_sd = math.ceil(math.log2((8 * r + 4 * sigma_max) / sigma_min))

    # Calculate the number of data points for the different steps of the estimation
    n = len(data)
    n1 = int(t_med / (0.098**2) * ((math.exp(eps) + 1) / (math.exp(eps) - 1)) ** 2 * math.log(16 * t_med / beta))
    n2 = int(t_sd / (0.052**2) * ((math.exp(eps) + 1) / (math.exp(eps) - 1)) ** 2 * math.log(16 * t_sd / beta))
    n3 = n - n1 - n2

    if n1 + n2 > n:
        return math.nan, math.nan, math.nan

    # Prepare the data for the different steps of the estimation
    data1 = data[:n1]
    data2 = data[n1 : n1 + n2]
    data3 = data[n1 + n2 :]

    # Estimate the median and the standard deviation using data1 and data2
    t_mu = _bin_quant(eps, data1, 0.5, (-r, r), 0.098, t_med, rng)
    t_sigma = _bin_quant(eps, data2, scipy.stats.norm.cdf(1), (-r, r + sigma_max), 0.052, t_sd, rng)

    diff = (t_sigma - t_mu) * (0.5 + 2 * math.sqrt(2 * math.log(8 * n / beta)))

    s1 = t_mu - diff
    s2 = t_mu + diff

    # Run steps 6-8 of KnownVar on data3
    return _steps6to8(eps, delta, data3, s1, s2, diff, t_sigma, r, beta, rng)


def _steps6to8(eps, delta, data, s1, s2, diff, sigma, r, beta, rng):
    """
    Run steps 6-8 of KnownVar algorithm from Gaboardi et al. 2019 [1, Algorithm 1].

    Args:
        data: input data
        s1: lower bound of the estimated interval
        s2: upper bound of the estimated interval
        diff: diff as calculated by KnownVar or UnkVar
        sigma: (estimated) standard deviation of the data
        r: defining the range of the data [-r, r]
        beta: confidence level. The returned confidence interval is a (1-beta)-confidence interval.

    Returns
    -------
        Tuple containing the estimated mean and the lower and upper bound of the confidence interval
    """
    # Project data onto the interval [s1, s2]
    data_proj = np.clip(data, s1, s2)
    n = len(data)

    # Use additive Gaussian noise to disturb the projected data
    sigma_squared_hat = 8 * diff**2 * math.log(2 / delta) / (eps**2)
    data_noisy = data_proj + rng.normal(0, math.sqrt(sigma_squared_hat), size=len(data_proj))

    # Correct the estimated confidence interval
    mu_tilde = np.sum(data_noisy) / n
    tau = np.sqrt((sigma**2 + sigma_squared_hat) / n) * scipy.stats.norm.cdf(1 - beta / 8)

    ci1 = mu_tilde - tau
    ci2 = mu_tilde + tau

    ci1_clip = min(max(ci1, -r), r)
    ci2_clip = min(max(ci2, -r), r)

    # Return the estimated mean and confidence interval
    return mu_tilde, ci1_clip, ci2_clip


class Gaboardi2019KnownVar(GaussianMean1D):
    """
    Implementation of the mean estimation from Gaboardi et al. 2019 [1] for known variance.

    The method works for 1-dimensional input that is assumed to be from a Gaussian distribution.
    The method is (eps, delta)-locally differentially private.
    """

    def __init__(self, eps: float, delta: float, sigma: float, r: float, beta: float, rng: np.random.Generator = None):
        """
        Initialize the KnownVar mechanism.

        Args:
            eps: The privacy budget epsilon.
            delta: The privacy parameter delta.
            sigma: The known standard deviation of the data.
            r: The range of the data is assumed to be [-r, r].
            beta: The confidence level. The returned confidence interval is a (1-beta)-confidence interval.
            rng: A numpy random number Generator.
        """
        super().__init__(eps, rng)
        self.delta = delta
        self.sigma = sigma
        self.r = r
        self.beta = beta

    def estimate_mean(self, t: np.array) -> float:
        """
        Estimate the mean of the data using the KnownVar mechanism.

        Args:
            t: The private data of n clients (vector of size n)

        Returns
        -------
            The estimated mean.
        """
        return _known_var(self.eps, self.delta, t, self.sigma, self.r, self.beta, self.rng)[0]


class Gaboardi2019UnknownVar(GaussianMean1D):
    """
    Implementation of the mean estimation from Gaboardi et al. 2019 [1] for unknown variance.

    The method works for 1-dimensional input that is assumed to be from a Gaussian distribution.
    The method is (eps, delta)-locally differentially private.
    """

    def __init__(
        self,
        eps: float,
        delta: float,
        sigma_interval: Tuple[float, float],
        r: float,
        beta: float,
        rng: np.random.Generator = None,
    ):
        """
        Initialize the UnkVar mechanism.

        Args:
            eps: The privacy budget epsilon.
            delta: The privacy parameter delta.
            sigma_interval: The interval [sigma_min, sigma_max] in which the standard deviation of the data is assumed
                            to be. 0 < sigma_min < sigma_max < 2r.
            r: The range of the data is assumed to be [-r, r].
            beta: The confidence level. The returned confidence interval is a (1-beta)-confidence interval.
            rng: A numpy random number Generator.
        """
        super().__init__(eps, rng)
        self.delta = delta

        self.sigma_interval = sigma_interval
        assert sigma_interval[0] > 0
        assert sigma_interval[1] > sigma_interval[0]
        assert sigma_interval[1] < 2 * r

        self.r = r
        self.beta = beta

    def estimate_mean(self, t: np.array) -> float:
        """
        Estimate the mean of the data using the UnkVar mechanism.

        Args:
            t: The private data of n clients (vector of size n)

        Returns
        -------
            The estimated mean.
        """
        return _unk_var(self.eps, self.delta, t, self.sigma_interval, self.r, self.beta, self.rng)[0]
