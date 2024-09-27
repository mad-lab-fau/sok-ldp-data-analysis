"""Abstract base class for the estimation of the mean and confidence interval under local differential privacy."""

from abc import abstractmethod
from typing import Tuple

import numpy as np

from sok_ldp_analysis.ldp.mean.mean import Mean1D


class MeanAndConfidenceInterval1D(Mean1D):
    """Abstract class for 1-dimensional mean and confidence interval estimation."""

    def __init__(self, eps: float, input_range: Tuple[float, float], rng: np.random.Generator = None):
        """Initialize the estimator.

        Args:
            eps: The privacy budget epsilon.
            input_range: The range of the input data.
            rng: A numpy random number Generator.
        """
        super().__init__(eps, input_range, rng)

    @abstractmethod
    def mean_and_ci(self, z: np.array) -> Tuple[float, float, float]:
        """
        Compute the mean and confidence interval of the disturbed data.

        Args:
            z: The disturbed data.

        Returns: A tuple containing the mean, the lower bound of the confidence interval and the upper bound of the
        confidence interval.
        """

    def mean(self, z: np.array) -> float:
        """Compute the mean.

        Args:
            z: The disturbed data.

        Returns: The mean.
        """
        return self.mean_and_ci(z)[0]

    def ci(self, z: np.array) -> Tuple[float, float]:
        """Compute the confidence interval.

        Args:
            z: The disturbed data.

        Returns: The confidence interval.
        """
        return self.mean_and_ci(z)[1], self.mean_and_ci(z)[2]
