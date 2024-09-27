from sok_ldp_analysis.ldp.mean.gaboardi2019 import Gaboardi2019UnknownVar, Gaboardi2019KnownVar
from sok_ldp_analysis.ldp.mean.joseph2019 import (
    Joseph2019KnownVarOneRound,
    Joseph2019KnownVar,
    Joseph2019UnknownVar,
    Joseph2019UnknownVarOneRound,
)

one_dim_mean_gaussian_methods = [
    Joseph2019KnownVar,
    Joseph2019KnownVarOneRound,
    Joseph2019UnknownVar,
    Joseph2019UnknownVarOneRound,
]

one_dim_mean_gaussian_eps_delta_methods = [Gaboardi2019KnownVar, Gaboardi2019UnknownVar]


def _init_gaussian_method(method, rng, eps, beta, known_sigma, sigma_interval=None, r=50, delta=1e-10):
    """
    Helper function to initialize the Gaussian methods.

    Args:
        method: The method to initialize.
        rng: The random number generator.
        eps: The privacy budget.
        beta: The failure probability.
        known_sigma: The known sigma value. Only for KnownVar methods.
        sigma_interval: The interval to search for the sigma value. Only for UnknownVar methods. Default: [1e-10, 20]
        r: The range [-r, r] to estimate the mean in. Only for Gaboardi. Default: 50
        delta: The failure probability for the algorithm. Only for epsilon-delta algorithms (Gaboardi). Default: 1e-10

    Returns:
        The initialized method.
    """
    if sigma_interval is None:
        sigma_interval = [1e-10, 20]

    kwargs = {"eps": eps, "rng": rng, "beta": beta}

    if "Unknown" in method.__name__:
        kwargs["sigma_interval"] = sigma_interval
    else:
        kwargs["sigma"] = known_sigma

    if "Gaboardi" in method.__name__:
        kwargs["r"] = r
        kwargs["delta"] = delta

    mechanism = method(**kwargs)

    return mechanism
