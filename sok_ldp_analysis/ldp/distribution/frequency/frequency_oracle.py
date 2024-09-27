from abc import ABC, abstractmethod

import numpy as np


class FrequencyOracle(ABC):
    """
    Abstract class for frequency estimation of data.
    """

    def __init__(self, eps: float, domain_size: int, rng: np.random.Generator = None):
        """
        Initialize the frequency estimator.

        Args:
            eps: The privacy budget epsilon.
            domain_size: The domain size of the input data. We assume that the input data are integers in the range
            [0, domain_size).
            rng: A numpy random number Generator. If None, the default rng is used.
        """
        self.eps = eps
        self.domain_size = domain_size
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def estimate_frequencies(self, t: np.array) -> np.array:
        """
        Run the mechanism and estimate the frequencies.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The estimated frequencies (vector of size domain_size).
        """
        return self.frequencies(self.response(t))

    def reseed_rng(self, seed: int):
        """
        Reseed the random number generator.

        Args:
            seed: The seed to use.
        """
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def response(self, t: np.array) -> np.array:
        """
        Run the mechanism and return the response.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The disturbed data.
        """
        pass

    @abstractmethod
    def frequencies(self, u: np.array) -> np.array:
        """
        Estimate the frequencies from the responses.

        Args:
            u: The disturbed data.

        Returns: The estimated frequencies (vector of size domain_size).
        """
        pass
