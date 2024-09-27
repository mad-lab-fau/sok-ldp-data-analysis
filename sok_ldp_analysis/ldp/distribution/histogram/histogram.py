from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np


class HistogramEstimator(ABC):
    """Abstract class for 1-dimensional histogram estimation."""

    def __init__(
        self,
        eps: float,
        input_range: Tuple[float, float],
        bin_count: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the histogram estimator.

        Args:
            eps: The privacy budget epsilon.
            input_range: The range of the input data.
            bin_count: The number of bins to use.
            rng: A numpy random number Generator. If None, the default rng is used.
        """
        self.eps = eps
        self.input_range = input_range
        self.bin_count = bin_count
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

    def estimate_histogram(self, t: np.array) -> np.array:
        """
        Run the mechanism and estimate the histogram (potentially using multiple rounds and post-processing).

        Args:
            t: The private data of n clients (vector of size n)

        Returns: Tuple of size 2 containing the estimated histogram and the bins.
        """
        return self.histogram(self.response(t))

    @abstractmethod
    def response(self, t: np.array) -> np.array:
        """
        Run the mechanism and return the response.

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The response.
        """

    @abstractmethod
    def histogram(self, u: np.array) -> np.array:
        """
        Compute the histogram of the response.

        Args:
            u: The noisy responses.

        Returns: Tuple of size 2 containing the estimated histogram and the bins.
        """
