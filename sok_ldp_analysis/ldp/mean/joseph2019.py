"""
Epsilon-LDP mean estimation algorithms from Joseph et al. 2019 [1].

[1] M. Joseph, J. Kulkarni, J. Mao, and S. Z. Wu, “Locally Private Gaussian Estimation,” in Advances in Neural
Information Processing Systems, Curran Associates, Inc., 2019. Available:
https://papers.nips.cc/paper_files/paper/2019/hash/a588a6199feff5ba48402883d9b72700-Abstract.html
"""

import math
from typing import Tuple

import numpy as np
from scipy.special import erfinv

from sok_ldp_analysis.ldp.mean.mean import GaussianMean1D


def _find_closest_point_in_s_uv(data, j1, j2, rho):
    """
    Find the closest point in S(j1, j2) to each point in data.

    This method is used for the unknown variance case.
    S(j1, j2) = {j2 + b * rho * 2 ** j1 | b in Z}

    Args:
        data: input data
        j1: index j1
        j2: index j2
        rho: rho

    Returns
    -------
        The closest point in S(j1, j2) to each point in data.
    """
    # For each point in data, find the closest point z in S(j1, j2)
    # S(j1, j2) = {j2 + b * rho * 2 ** j1 | b in Z}
    data_ = data - j2
    data_ = data_ / (rho * 2**j1)
    b = np.round(data_)
    s = b * rho * 2**j1 + j2

    return s


def _rr1(eps, data, k, Lmin, Lmax, rng):
    """
    Algorithm 1 from the supplement of [1].

    Args:
        eps: privacy budget
        data: input data
        k: size of subgroups
        Lmin: minimum subgroup index
        Lmax: maximum subgroup index
        rng: random number generator

    Returns
    -------
        Array of responses
    """
    idx = np.floor(np.arange(len(data)) / k) + Lmin

    assert np.all(idx >= Lmin)
    assert np.all(idx <= Lmax)

    # data mod 4
    y = np.floor(data / np.power(2, idx)) % 4

    # Sample values from {0,1,2,3} \ y.
    # We add a random sample from {1,2,3} to y and then take the modulo 4.
    y_rnd = (y + rng.integers(1, 4, size=len(data))) % 4

    # Sample n random points uniformly form [0,1]
    p = rng.uniform(size=len(data))

    # If p > exp(eps) / (exp(eps) + 3), use the sample, else use y
    y_resp = np.where(p > math.exp(eps) / (math.exp(eps) + 3), y_rnd, y)

    return y_resp


def _kv_agg1(eps, responses, k, Lmin, Lmax):
    """
    Algorithm 2 from [1].

    Args:
        eps: privacy budget
        responses: array of responses
        k: size of subgroups
        Lmin: minimum subgroup index
        Lmax: maximum subgroup index

    Returns
    -------
        The raw histogram counts and the randomized-response-adjusted histogram counts
    """
    # Note: the paper only uses a \in {0,1}, but this does not fit to the other pseudocodes. We use a \in {0,1,2,3}
    C = np.zeros((Lmax - Lmin + 1, 4))

    idx = np.floor(np.arange(len(responses)) / k) + Lmin

    L = list(range(Lmin, Lmax + 1))
    for j in L:
        for a in [0, 1, 2, 3]:
            # Note that Lmin can be non-zero (or even negative), so we need to subtract Lmin from the index to
            # get a zero-based index for C
            C[j - Lmin, a] = np.count_nonzero(responses[idx == j] == a)

    H = (math.exp(eps) + 3) / (math.exp(eps) - 1) * (C - k / (math.exp(eps) + 3))

    return C, H


def _est_mean(eps, hist, k, Lmin, Lmax, beta):
    """
    Algorithm 3 from [1].

    Args:
        eps: privacy budget
        hist: histogram
        k: size of subgroups
        Lmin: minimum subgroup index
        Lmax: maximum subgroup index
        beta: probability for the estimated mean to be more than 2 * sigma away from the true mean

    Returns
    -------
        The estimated mean
    """
    psi = (eps + 4) / (eps * math.sqrt(2)) * math.sqrt(k * math.log(8 * (Lmax - Lmin + 1) / beta))

    # Note: M1, M2 and hist use a zero-based index, so we need to subtract Lmin from j to get the correct index
    M1 = np.argsort(hist)[:, -1]  # largest element
    M2 = np.argsort(hist)[:, -2]  # second-largest element

    search_min = -(2**Lmax)  # TODO: this is not in the paper, but it works - check if it's in the proof!
    search_max = 2**Lmax

    j = Lmax
    while j >= Lmin and np.max(hist[j - Lmin, :]) >= 0.52 * k + psi:
        # Find integer c such that c*2^j is in the interval [search_min, search_max] and c mod 4 is M1[j]
        c_candidate = search_min // 2**j
        while search_min <= c_candidate * 2**j <= search_max:
            if c_candidate % 4 == M1[j - Lmin]:
                c = c_candidate
                break
            c_candidate += 1
        else:
            raise ValueError("No c found")

        # c = M1[j - Lmin]
        # while c * 2**j < search_min:
        #     c += 4

        assert c * 2**j >= search_min
        assert c * 2**j <= search_max

        # Update the search interval
        search_min = c * 2**j
        search_max = (c + 1) * 2**j

        j -= 1

    j = max(j, Lmin)

    # Find the largest integer c such that c*2^j is in the search interval and c mod 4 is M1[j] or M2[j]
    c = search_max // 2**j
    while search_min <= c * 2**j <= search_max:
        if c % 4 == M1[j - Lmin] or c % 4 == M2[j - Lmin]:
            break
        c -= 1

    assert c * 2**j >= search_min
    assert c * 2**j <= search_max

    return c * 2**j


def _kv_rr2(eps, data, mean, sigma, rng):
    """
    Algorithm 4 from [1].

    Args:
        eps: privacy budget
        data: input data
        mean: estimated mean
        sigma: standard deviation
        rng: random number generator

    Returns
    -------
        Array of responses (based on randomized response)
    """
    x = (data - mean) / sigma
    y = np.sign(x)
    c = rng.uniform(size=len(data))
    z = np.where(c <= (math.exp(eps)) / (math.exp(eps) + 1), y, -y)

    return z


def _kv_agg2(eps, responses, k):
    """
    Algorithm 5 from [1].

    Args:
        eps: privacy budget
        responses: array of responses
        k: size of subgroups

    Returns
    -------
        The raw histogram counts and the randomized-response-adjusted histogram counts
    """
    C = np.zeros(2)
    for i, a in enumerate([-1, 1]):
        C[i] = np.count_nonzero(responses == a)

    H = (math.exp(eps) + 1) / (math.exp(eps) - 1) * (C - k / (math.exp(eps) + 1))

    return C, H


def kv_gausstimate(eps, data, sigma, beta, rng):
    """
    Algorithm 1 from [1] for the known variance case with two rounds.

    Args:
        eps: privacy budget
        data: input data
        sigma: standard deviation
        beta: probability for the estimated mean to be more than 2 * sigma away from the true mean
        rng: random number generator

    Returns
    -------
        The estimated mean
    """
    n = len(data)
    k, L, Lmin, Lmax = _calc_k_and_L(n, beta, eps, sigma)

    # use the first half of the data (actually L*k <= n/2) to estimate the mean using _rr1, _kv_agg1 and _est_mean
    responses = _rr1(eps, data[: L * k], k, Lmin, Lmax, rng)
    hist = _kv_agg1(eps, responses, k, Lmin, Lmax)[1]
    mean1 = _est_mean(eps, hist, k, Lmin, Lmax, beta)

    # Us the seconds half of the data to estimate the deviation using _rr2 and _kv_agg2
    responses = _kv_rr2(eps, data[L * k :], mean1, sigma, rng)

    # Note: the paper uses n/2 here, but in reality we use slightly more data
    hist2 = _kv_agg2(eps, responses, len(responses))[0]

    # Note: hist2[0] is the count for -1, hist2[1] is the count for 1
    t = math.sqrt(2) * erfinv((2 * (-hist2[0] + hist2[1])) / n)

    return mean1 + t * sigma


def _one_round_kv_rr2(eps, data, j, rho, sigma, rng):
    """
    Algorithm 3 from the supplement of [1].

    Args:
        eps: privacy budget
        data: input data
        j: index j
        rho: rho
        sigma: standard deviation
        rng: random number generator

    Returns
    -------
        Array of responses (based on randomized response)
    """
    # For each point in data, find the closest point z in S(j)
    # S(j) = {j + b * rho * sigma | b in Z}
    data_ = data - j
    data_ = data_ / (rho * sigma)
    b = np.round(data_)
    z = b * rho * sigma + j
    y = np.sign((data - z) / sigma)

    # Flip each bit with probability exp(eps) / (exp(eps) + 1)
    p = math.exp(eps) / (math.exp(eps) + 1)
    y_flipped = rng.binomial(n=1, p=p, size=len(data))
    y_ = np.where(y_flipped, y, -y)
    return y_


def _calc_k_and_L(n, beta, eps, sigma, sigma_max=None):
    Lmin = math.floor(math.log(sigma))

    # in the paper, k is given in Omega-notation - we don't know the constant
    # Note: The Proof of Lemma 6.3 has some assumptions on the size of k
    # TODO: check if we can enforce the assumptions to set k (see kv_gausstimate)
    k = math.ceil(math.log(n / beta) / (eps**2)) * 1000
    k = min(k, int(n / 2 / 10))  # Make sure that we have at least 10 subgroups
    k = max(k, int(n / (2 * (60 - Lmin))))  # Limit Lmax to 60 (to avoid int64 overflow)

    if sigma_max is not None:
        # Ensure Lmax >= math.floor(math.log(sigma_max))
        Lmax = math.floor(math.log(sigma_max)) + 1
        k = int(n / (2 * (Lmax - Lmin + 1)))

    # TODO: Multiplying k by 1000 worked well enough empirically;
    #  but we should check if we can enforce the assumptions to set k!
    #  Blindly trying to enforce the assumptions may lead to a very small L and therefore bad results or nans
    #  (especially for small n)
    # # Try to enforce the strictest assumption on k
    # while k < 5000 * math.log(L / beta):
    #     k *= 2
    #     L = math.floor(n / (2 * k))

    # print(k)
    # print(5000 * math.log(L / beta))
    # print(5000 * math.log(5 / beta))
    # print(625 * ((eps + 4) / (eps * math.sqrt(2))) ** 2 * math.log(4 * L / beta))
    # print()
    # print(n)
    # print(20000 * math.log(4 / beta))
    # print(20000 * ((eps + 2) / eps) ** 2 * math.log(4 / beta))

    L = math.floor(n / (2 * k))

    Lmax = Lmin - 1 + L

    # 2k (2**63 - Lmin) = n. Lmax should be smaller than 2**63

    return k, L, Lmin, Lmax


def one_round_kv_gausstimate(eps, data, sigma, beta, rng):
    """
    Algorithm 2 from the supplement of [1] for the known variance case with one round.

    Args:
        eps: privacy budget
        data: input data
        sigma: standard deviation
        beta: probability for the estimated mean to be more than 2 * sigma away from the true mean
        rng: random number generator

    Returns
    -------
        The estimated mean
    """
    n = len(data)
    k, L, Lmin, Lmax = _calc_k_and_L(n, beta, eps, sigma)

    # Split the data
    u1 = data[: L * k]
    u2 = data[L * k :]

    # Split the second half of the data into several subgroups
    rho = math.ceil(2 * math.sqrt(math.log(4 * n)))
    r = [0.2 * sigma * i for i in range(1, 5 * rho + 1)]

    subgroups = np.array_split(u2, len(r))

    # Use the first half of the data to run _rr1
    responses1 = _rr1(eps, u1, k, Lmin, Lmax, rng)

    # Use the second half of the data to run _one_round_kv_rr2 for each subgroup
    responses2 = []
    for j, subgroup in zip(r, subgroups):
        res2 = _one_round_kv_rr2(eps, subgroup, j, rho, sigma, rng)
        responses2.append(res2)

    # Analyze the first half of the data
    hist = _kv_agg1(eps, responses1, k, Lmin, Lmax)[1]
    mean1 = _est_mean(eps, hist, k, Lmin, Lmax, beta)

    # Analyze the second half of the data
    # Find j_star for which the difference between any point in S(j_star) and mean1 is smallest
    # Each group S(j) is defined as S(j) = {j + b * rho * sigma | b in Z}
    j_star_idx = -1
    min_diff = math.inf
    for idx, j in enumerate(r):
        b = np.round((mean1 - j) / (rho * sigma))
        diff = np.abs(mean1 - (j + b * rho * sigma))
        if diff < min_diff:
            min_diff = diff
            j_star_idx = idx

    assert j_star_idx != -1
    j_star = r[j_star_idx]

    # Aggregate the responses for subgroup j_star
    k2 = len(subgroups[j_star_idx])
    hist2 = _kv_agg2(eps, responses2[j_star_idx], k2)[0]

    # Note: hist2[0] is the count for -1, hist2[1] is the count for 1
    t = math.sqrt(2) * erfinv((-hist2[0] + hist2[1]) / k2)

    # Find the closest point to mean1 in S(j_star)
    b = np.round((mean1 - j_star) / (rho * sigma))
    closest_point = j_star + b * rho * sigma

    return sigma * t + closest_point


def uv_gausstimate(eps, data, sigma_range, beta, rng):
    """
    Algorithm 6 from [1] for the unknown variance case with two rounds.

    Args:
        eps: privacy budget
        data: input data
        sigma_range: range of possible standard deviations
        beta: probability for the estimated mean to be more than 2 * sigma away from the true mean
        rng: random number generator

    Returns
    -------
        The estimated mean
    """
    sigma_min, sigma_max = sigma_range
    n = len(data)

    k1, L, Lmin, Lmax = _calc_k_and_L(n, beta, eps, sigma_min, sigma_max)

    assert Lmax >= math.ceil(math.log(sigma_max))

    # Split the data
    u1 = data[: L * k1]
    u2 = data[L * k1 :]

    # Use the first half of the data to run _rr1
    responses1 = _rr1(eps, u1, k1, Lmin, Lmax, rng)

    # Analyze the first half of the data
    hist1 = _agg1(eps, k1, Lmin, Lmax, u1)
    sigma_hat = _est_var(eps, hist1, k1, Lmin, Lmax, beta)

    hist2 = _kv_agg1(eps, responses1, k1, Lmin, Lmax)[1]
    mean1 = _est_mean(eps, hist2, k1, Lmin, Lmax, beta)

    interval_len = sigma_hat * (2 + math.sqrt(math.log(4 * n)))
    interval_min = mean1 - interval_len
    interval_max = mean1 + interval_len

    # Split the second half of the data into several subgroups
    responses2 = _uv_rr2(eps, u2, interval_min, interval_max, rng)

    # Analyze the second half of the data
    mean2 = 2 / n * np.sum(responses2)

    return mean2


def _agg1(eps, k, Lmin, Lmax, data):
    """
    Algorithm 4 from the supplement of [1].

    Args:
        eps: privacy budget
        k: size of subgroups
        Lmin: minimum subgroup index
        Lmax: maximum subgroup index
        data: input data

    Returns
    -------
        The randomized-response-adjusted histogram counts
    """
    # We reuse the code from _kv_agg1 since it is the same as the first part of _agg1
    H = _kv_agg1(eps, data, k, Lmin, Lmax)[1]
    H1 = np.zeros_like(H)
    for a in [0, 1, 2, 3]:
        H1[:, a] = H[:, a] + H[:, (a + 1) % 4]
    return H1


def _est_var(eps, hist, k, Lmin, Lmax, beta):
    """
    Algorithm 5 from the supplement of [1].

    Args:
        eps: privacy budget
        hist: histogram
        k: size of subgroups
        Lmin: minimum subgroup index
        Lmax: maximum subgroup index
        beta: probability for the estimated mean to be more than 2 * sigma away from the true mean

    Returns
    -------
        The estimated variance
    """
    # Find smallest index j for such that for all j' > j, hist[j'] is concentrated

    L = Lmax - Lmin + 1
    tau = math.sqrt(2 * k * math.log(2 * L / beta)) + (1 + 4 / eps) * math.sqrt(2 * k * math.log(8 * L / beta))

    j = Lmin
    while j <= Lmax:
        found_unconcentrated = False
        for j_ in range(j + 1, Lmax + 1):
            # if not concentrated break
            if np.min(hist[j_, :]) > 0.03 * k + tau:
                found_unconcentrated = True
                break

        if not found_unconcentrated:
            return 2**j

        j += 1

    return 2**Lmax


def _uv_rr2(eps, data, interval_min, interval_max, rng):
    """
    Algorithm 6 from the supplement of [1].

    Args:
        eps: privacy budget
        data: input data
        interval_min: minimum value of the interval
        interval_max: maximum value of the interval
        rng: random number generator

    Returns
    -------
        Array of responses (based on randomized response)
    """
    # Algorithm 12
    data_clipped = np.clip(data, interval_min, interval_max)
    # Add Laplace noise to each point
    noise = rng.laplace(scale=(interval_max - interval_min) / eps, size=len(data))
    return data_clipped + noise


def one_round_uv_gausstimate(eps, data, sigma_range, beta, rng):
    """
    Algorithm 7 from [1] for the unknown variance case with one round.

    Args:
        eps: privacy budget
        data: input data
        sigma_range: range of possible standard deviations
        beta: probability for the estimated mean to be more than 2 * sigma away from the true mean
        rng: random number generator

    Returns
    -------
        The estimated mean
    """
    sigma_min, sigma_max = sigma_range
    n = len(data)

    k1, L, Lmin, Lmax = _calc_k_and_L(n, beta, eps, sigma_min, sigma_max)

    assert Lmax >= math.ceil(math.log(sigma_max))

    # Split the data
    u1 = data[: L * k1]
    u2 = data[L * k1 :]

    # Split the second half of the data into several subgroups
    rho = math.ceil(math.sqrt(math.log(4 * n)) + 6)
    j1s = list(range(Lmin, Lmax + 1))
    r = [[j2 * 2**j1 for j2 in range(1, rho + 1)] for j1 in j1s]

    subgroups = np.array_split(u2, L * rho)

    # Use the first half of the data to run _rr1
    responses1 = _rr1(eps, u1, k1, Lmin, Lmax, rng)

    # Use the second half of the data to run _one_round_uv_rr2 for each subgroup
    responses2 = []
    for idx1, j1 in enumerate(j1s):
        responses2.append([])
        for idx2, j2 in enumerate(r[idx1]):
            res2 = _one_round_uv_rr2(eps, subgroups[idx1 * rho + idx2], j1, j2, rho, rng)
            responses2[idx1].append(res2)

    # Analyze the first half of the data
    hist1 = _agg1(eps, k1, Lmin, Lmax, u1)
    sigma_hat = _est_var(eps, hist1, k1, Lmin, Lmax, beta)  # TODO: test this function, sigma_hat is very small!

    j1 = int(math.log(sigma_hat))

    hist2 = _kv_agg1(eps, responses1, k1, Lmin, Lmax)[1]
    mean1 = _est_mean(eps, hist2, k1, Lmin, Lmax, beta)

    s_star = math.inf
    j2_idx = None
    for idx, j2 in enumerate(r[j1 - Lmin]):
        s = _find_closest_point_in_s_uv(mean1, j1, j2, rho)
        if s < s_star:
            s_star = s - mean1
            j2_idx = idx

    final_subgroup = subgroups[(j1 - Lmin) * rho + j2_idx]
    mean2 = s_star + np.sum(final_subgroup) / len(final_subgroup)  # TODO: invalid value encountered in scalar divide

    return mean2


def _one_round_uv_rr2(eps, data, j1, j2, rho, rng):
    """
    Algorithm 8 from the supplement of [1].

    Args:
        eps: privacy budget
        data: input data
        j1: index j1
        j2: index j2
        rho: rho
        rng: random number generator

    Returns
    -------
        Array of responses (based on randomized response)
    """
    s = _find_closest_point_in_s_uv(data, j1, j2, rho)

    y = data - s

    # Add Laplace noise
    noise = rng.laplace(scale=2 * rho * 2**j1 / eps, size=len(data))
    return y + noise


class Joseph2019KnownVar(GaussianMean1D):
    """Gaussian mean estimation algorithm from Joseph et al. 2019 [1] for the known variance case with two rounds."""

    def __init__(self, eps: float, sigma: float, beta: float = 0.05, rng: np.random.Generator = None):
        """
        Initialize the mechanism.

        Args:
            eps: The privacy budget epsilon.
            sigma: The known standard deviation of the data.
            beta: The probability for the estimated mean to be more than 2 * sigma away from the true mean.
            rng: A numpy random number Generator.
        """
        super().__init__(eps, rng)
        self.sigma = sigma
        self.beta = beta

    def estimate_mean(self, t: np.array) -> float:
        """
        Run the mechanism and estimate the mean.

        Args:
            t: The private data of n clients (vector of size n)

        Returns
        -------
            The estimated mean.
        """
        return kv_gausstimate(self.eps, t, self.sigma, self.beta, self.rng)


class Joseph2019KnownVarOneRound(GaussianMean1D):
    """Gaussian mean estimation algorithm from Joseph et al. 2019 [1] for the known variance case with one round."""

    def __init__(self, eps: float, sigma: float, beta: float = 0.05, rng: np.random.Generator = None):
        """
        Initialize the mechanism.

        Args:
            eps: The privacy budget epsilon.
            sigma: The known standard deviation of the data.
            beta: The probability for the estimated mean to be more than 2 * sigma away from the true mean.
            rng: A numpy random number Generator.
        """
        super().__init__(eps, rng)
        self.sigma = sigma
        self.beta = beta

    def estimate_mean(self, t: np.array) -> float:
        """
        Run the mechanism and estimate the mean.

        Args:
            t: The private data of n clients (vector of size n)

        Returns
        -------
            The estimated mean.
        """
        return one_round_kv_gausstimate(self.eps, t, self.sigma, self.beta, self.rng)


class Joseph2019UnknownVar(GaussianMean1D):
    """Gaussian mean estimation algorithm from Joseph et al. 2019 [1] for the unknown variance case with two rounds."""

    def __init__(
        self, eps: float, sigma_interval: Tuple[float, float], beta: float = 0.05, rng: np.random.Generator = None
    ):
        """
        Initialize the mechanism.

        Args:
            eps: The privacy budget epsilon.
            sigma_range: The range of possible standard deviations of the data.
            beta: The probability for the estimated mean to be more than 2 * sigma away from the true mean.
            rng: A numpy random number Generator.
        """
        super().__init__(eps, rng)
        self.sigma_interval = sigma_interval
        self.beta = beta

    def estimate_mean(self, t: np.array) -> float:
        """
        Run the mechanism and estimate the mean.

        Args:
            t: The private data of n clients (vector of size n)

        Returns
        -------
            The estimated mean.
        """
        return uv_gausstimate(self.eps, t, self.sigma_interval, self.beta, self.rng)


class Joseph2019UnknownVarOneRound(GaussianMean1D):
    """Gaussian mean estimation algorithm from Joseph et al. 2019 [1] for the unknown variance case with one round."""

    def __init__(
        self, eps: float, sigma_interval: Tuple[float, float], beta: float = 0.05, rng: np.random.Generator = None
    ):
        """
        Initialize the mechanism.

        Args:
            eps: The privacy budget epsilon.
            sigma_interval: The range of possible standard deviations of the data.
            beta: The probability for the estimated mean to be more than 2 * sigma away from the true mean.
            rng: A numpy random number Generator.
        """
        super().__init__(eps, rng)
        self.sigma_interval = sigma_interval
        self.beta = beta

    def estimate_mean(self, t: np.array) -> float:
        """
        Run the mechanism and estimate the mean.

        Args:
            t: The private data of n clients (vector of size n)

        Returns
        -------
            The estimated mean.
        """
        return one_round_uv_gausstimate(self.eps, t, self.sigma_interval, self.beta, self.rng)
