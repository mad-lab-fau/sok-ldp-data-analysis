import math

import numpy as np

from sok_ldp_analysis.ldp.distribution.frequency.d_randomized_response import DRandomizedResponse
from sok_ldp_analysis.ldp.distribution.frequency.duchi2013 import Duchi2013
from sok_ldp_analysis.ldp.distribution.frequency.k_subset import KSubset
from sok_ldp_analysis.ldp.distribution.frequency.kairouz2016 import Kairouz2016
from sok_ldp_analysis.ldp.distribution.frequency.murakami2018 import Murakami2018
from sok_ldp_analysis.ldp.distribution.frequency.nguyen2016 import Nguyen2016FO
from sok_ldp_analysis.ldp.distribution.frequency.pure_fo import PureFO, PureProtocol


class DirectEncoding(PureFO):
    def __init__(self, eps: float, domain_size: int, rng: np.random.Generator = None):
        super().__init__(eps, domain_size, PureProtocol.DE, None, None, rng)


class SymmetricUnaryEncoding(PureFO):
    def __init__(self, eps: float, domain_size: int, rng: np.random.Generator = None):
        super().__init__(eps, domain_size, PureProtocol.UE, None, None, rng)


class OptimizedUnaryEncoding(PureFO):
    def __init__(self, eps: float, domain_size: int, rng: np.random.Generator = None):
        super().__init__(eps, domain_size, PureProtocol.OUE, None, None, rng)


class OptimizedLocalHashing(PureFO):
    def __init__(self, eps: float, domain_size: int, rng: np.random.Generator = None):
        g = 2  # set by OLH as e^eps + 1
        client_params = {"g": g}
        server_params = {"g": g}

        super().__init__(eps, domain_size, PureProtocol.OLH, client_params, server_params, rng)


class FastLocalHashing(PureFO):
    def __init__(self, eps: float, domain_size: int, rng: np.random.Generator = None):
        # TODO: how to set k and g?
        k = int(domain_size / 2)
        g = 2

        client_params = {"k": k, "g": g}
        server_params = {"k": k, "g": g}

        super().__init__(eps, domain_size, PureProtocol.FLH, client_params, server_params, rng)


class HadamardMechanism(PureFO):
    def __init__(self, eps: float, domain_size: int, rng: np.random.Generator = None):
        server_params = {"use_optimal_t": True, "t": 0}
        client_params = {"use_optimal_t": True, "t": 0}

        super().__init__(eps, domain_size, PureProtocol.HM, client_params, server_params, rng)


class HadamardResponse(PureFO):
    def __init__(self, eps: float, domain_size: int, rng: np.random.Generator = None):
        super().__init__(eps, domain_size, PureProtocol.HR, None, None, rng)


class RAPPOR(PureFO):
    def __init__(self, eps: float, domain_size: int, rng: np.random.Generator = None):
        f = round(1 / (0.5 * math.exp(eps / 2) + 0.5), 2)  # transformation taken from the pure_ldp package
        m = 128  # number of bloom bits
        k = 2  # number of hash functions

        server_params = {"f": f, "m": m, "k": k}
        client_params = {"f": f, "m": m}  # client does not need k, as it gets the hash functions from the server

        super().__init__(eps, domain_size, PureProtocol.RAPPOR, client_params, server_params, rng)


non_pure_frequency_oracles = [
    Kairouz2016,
    DRandomizedResponse,
    Duchi2013,
    KSubset,
    Nguyen2016FO,
    Murakami2018,
]

pure_frequency_oracles = [
    OptimizedUnaryEncoding,
    OptimizedLocalHashing,
    HadamardMechanism,
    HadamardResponse,
    RAPPOR,
    SymmetricUnaryEncoding,
]
