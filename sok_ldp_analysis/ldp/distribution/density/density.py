from abc import abstractmethod, ABC
from typing import Tuple

import numpy as np
import scipy


class Density(ABC):
    def __init__(
        self,
        input_range: Tuple[float, float],
        mechanism_range: Tuple[float, float] = (0, 1),
    ):
        """
        Initialize the density function.

        Args:
            input_range: The range of the input data.
            mechanism_range: The input range required by the mechanism.
        """
        self.input_range = input_range
        self.mechanism_range = mechanism_range

        self.output_transform_factor = (self.input_range[1] - self.input_range[0]) / (
            self.mechanism_range[1] - self.mechanism_range[0]
        )

        self.correction_factor_ = None

    def _transform_input(self, x: float) -> float:
        """
        Transform the input data to the range required by the mechanism.

        Args:
            x: The input data.

        Returns: The transformed input data.
        """
        return (x - self.input_range[0]) / self.output_transform_factor + self.mechanism_range[0]

    @abstractmethod
    def _density(self, x: float) -> float:
        """
        Compute the density function.

        Args:
            x: The input data in the range of the mechanism.

        Returns: The density at x.
        """
        raise NotImplementedError

    @property
    def correction_factor(self):
        if not self.correction_factor_:
            self.correction_factor_ = scipy.integrate.quad(
                self._density, self.mechanism_range[0], self.mechanism_range[1]
            )[0]

        return self.correction_factor_

    def __call__(self, x: float, correct_density: bool = True) -> float:
        """
        Compute the density function.

        Args:
            x: The input data.
            correct_density: Whether to correct the density to have an integral of 1.

        Returns: The density at x.
        """
        density = self._density(self._transform_input(x))

        if correct_density:
            density /= self.correction_factor

        return density / self.output_transform_factor


class DensityEstimator(ABC):
    def __init__(
        self,
        eps: float,
        input_range: Tuple[float, float],
        mechanism_range: Tuple[float, float] = (0, 1),
        rng: np.random.Generator = None,
    ):
        self.eps = eps
        self.mechanism_range = mechanism_range
        self.input_range = input_range
        self.rng = rng

    def reseed_rng(self, seed: int):
        """
        Reseed the random number generator.

        Args:
            seed: The seed to use.
        """
        self.rng = np.random.default_rng(seed)

    def _transform_input(self, x: float) -> float:
        """
        Transform the input data to the range required by the mechanism.

        Args:
            x: The input data.

        Returns: The transformed input data.
        """
        input_range_size = self.input_range[1] - self.input_range[0]
        mechanism_range_size = self.mechanism_range[1] - self.mechanism_range[0]
        return (x - self.input_range[0]) / input_range_size * mechanism_range_size + self.mechanism_range[0]

    @abstractmethod
    def response(self, x):
        raise NotImplementedError

    @abstractmethod
    def density(self, z) -> Density:
        raise NotImplementedError

    def estimate_density(self, x: np.array) -> Density:
        """
        Apply the mechanism and estimate the density on the data of n participants.

        Args:
            x: The private data of n participants.

        Returns: The estimated density function.
        """
        return self.density(self.response(self._transform_input(x)))


class HistogramDensity(Density):
    def __init__(self, theta, input_range: Tuple[float, float]):
        super().__init__(input_range, mechanism_range=(0, 1))
        self.theta = theta

    def _density(self, x: float) -> float:
        """
        Compute the density function based on the estimated histogram.

        Args:
            x: The input data in the range of the mechanism [0, 1].

        Returns: The estimated density at x.
        """

        domain_bins = len(self.theta)

        # check which bin x falls into
        bin_idx = int(x * domain_bins)

        if bin_idx == domain_bins:
            bin_idx -= 1

        return self.theta[bin_idx]

    @property
    def correction_factor(self):
        if not self.correction_factor_:
            bin_width = 1 / len(self.theta)
            self.correction_factor_ = np.sum(self.theta * bin_width)

            print("correction", self.correction_factor_)

        return self.correction_factor_
