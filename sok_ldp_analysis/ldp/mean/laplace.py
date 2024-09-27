"""Laplace mechanism for local differential privacy.

Introduced by Dwork et al., 2006 for central DP [1]. Optimality for 1-dim. LDP proven by Duchi et al., 2018 [2].
We use the scale parameter as defined by Duchi et al., 2018 and restrict the input to be in the range [-1, 1].

[1] C. Dwork, F. McSherry, K. Nissim, and A. Smith, “Calibrating Noise to Sensitivity in Private Data
Analysis,” in Theory of Cryptography, S. Halevi and T. Rabin, Eds., in Lecture Notes in Computer Science.
Berlin, Heidelberg: Springer, 2006, pp. 265-284. doi: 10.1007/11681878_14.

[2] J. C. Duchi, M. I. Jordan, and M. J. Wainwright, “Minimax Optimal Procedures for Locally Private
Estimation,” Journal of the American Statistical Association, vol. 113, no. 521, pp. 182-201, Jan. 2018,
doi: 10.1080/01621459.2017.1389735.
"""

import numpy as np

from sok_ldp_analysis.ldp.mean.mean import Mean1D, MeanMultiDim


class Laplace1D(Mean1D):
    """Laplace mechanism for 1-dimensional input (of multiple clients).

    Introduced by Dwork et al., 2006 for central DP [1]. Optimality for LDP proven by Duchi et al., 2018 [2].
    We use the scale parameter as defined by Duchi et al., 2018 and restrict the input to be in the range [-1, 1].
    """

    def mechanism(self, t: np.array) -> np.array:
        """Apply the Laplace mechanism to a 1-dimensional input (of multiple clients).

        Args:
            t: The private data of n clients (vector of size n)

        Returns: The disturbed data.

        """
        # Only works for 1-dimensional inputs
        assert t.ndim == 1

        # Scale data to be in range [-1, 1]
        range_size = self.input_range[1] - self.input_range[0]
        x = 2 * (t - self.input_range[0]) / range_size - 1

        assert np.all(np.abs(x) <= 1 + 1e-10)

        # Note: Due to the different definition of the Laplace distribution by Duchi et al., 2018, we need to use the
        # inverse scale parameter.
        return x + self.rng.laplace(loc=0, scale=2 / self.eps, size=x.shape)

    def mean(self, u: np.array) -> float:
        """Compute the mean of the disturbed data."""
        mean = np.mean(u)

        # Return the mean in the original range
        return float((mean + 1) / 2 * (self.input_range[1] - self.input_range[0]) + self.input_range[0])


class Laplace(MeanMultiDim):
    """Laplace mechanism for d-dimensional input (of multiple clients).

    We use the 1-D version in combination with the composition theorem and split eps equally between the dimensions.
    Following the special case in Duchi et al., 2018 [2], we restrict the input to be in the range [-1, 1].
    """

    def mechanism(self, t: np.array) -> np.array:
        """Apply the Laplacian mechanism to a d-dimensional input (of multiple clients).

        Args:
            t: The private data of n clients (matrix of size n x d)

        Returns: The disturbed data.

        """
        assert t.ndim == 2

        d = t.shape[1]

        # Scale data to be in range [-1, 1]
        range_size = self.input_range[1] - self.input_range[0]
        x = 2 * (t - self.input_range[0]) / range_size - 1

        assert np.all(np.abs(x) <= 1 + 1e-10)

        return x + self.rng.laplace(loc=0, scale=d * 2 / self.eps, size=x.shape)

    def mean(self, u: np.array) -> np.array:
        """Compute the mean of the disturbed data."""
        mean = np.mean(u, axis=0)

        # Return the mean in the original range
        return (mean + 1) / 2 * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
