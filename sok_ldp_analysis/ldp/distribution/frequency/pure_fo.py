"""
This module provides a wrapper for the pure frequency oracles as implemented by the pure_ldp package [1].

[1] https://github.com/Samuel-Maddock/pure-LDP
"""
from enum import Enum

import numpy as np
from pure_ldp.core.fo_creator import create_fo_client_instance, create_fo_server_instance

from sok_ldp_analysis.ldp.distribution.frequency.frequency_oracle import FrequencyOracle


class PureProtocol(Enum):
    """
    Options for Pure Protocols as implemented by the pure_ldp package [1].
    """

    RAPPOR = "RAPPOR"  # RAPPOR
    DE = "DE"  # Direct Encoding
    UE = "UE"  # Unary Encoding
    OUE = "oue"  # Optimal Unary Encoding
    HE = "HE"  # Histogram Encoding
    SHE = "she"  # Histogram Encoding with Summation
    THE = "the"  # Histogram Encoding with Thresholding
    LH = "LH"  # Local Hashing
    OLH = "olh"  # Optimal Local Hashing
    FLH = "FastLH"  # Fast Local Hashing
    HR = "HadamardResponse"  # Hadamard Response
    HM = "HadamardMech"  # Hadamard Mechanism


class PureFO(FrequencyOracle):
    """
    Wrapper for pure frequency oracles as implemented by the pure_ldp package [1].
    """

    def __init__(
        self,
        eps: float,
        domain_size: int,
        protocol: PureProtocol,
        client_params,
        server_params,
        rng: np.random.Generator = None,
    ):
        """
        Initialize the frequency estimator.

        Args:
            eps: The privacy budget epsilon.
            domain_size: The domain size of the input data. We assume that the input data are integers in the range
            [0, domain_size).
            protocol: The pure protocol to use.
            rng: A numpy random number Generator. If None, the default rng is used.
        """
        super().__init__(eps, domain_size, rng)

        # The pure FO implementations do not support using a NumPy Generator
        np.random.seed(self.rng.bit_generator._seed_seq.entropy % (2**32 - 1))

        if client_params is None:
            client_params = {}

        if server_params is None:
            server_params = {}

        if protocol == PureProtocol.OUE:
            protocol = PureProtocol.UE
            server_params["use_oue"] = True

        if protocol == PureProtocol.SHE:
            protocol = PureProtocol.HE

        if protocol == PureProtocol.THE:
            protocol = PureProtocol.HE
            server_params["use_the"] = True

        if protocol == PureProtocol.OLH:
            protocol = PureProtocol.LH
            server_params["use_olh"] = True

        if protocol == PureProtocol.FLH and "k" not in server_params:
            server_params["k"] = int(domain_size / 2)
            client_params["k"] = int(domain_size / 2)

        if protocol == PureProtocol.HM:
            assert (
                domain_size & (domain_size - 1) == 0
            ), "The domain size must be a power of 2 for the Hadamard Mechanism."

        client_params["epsilon"] = eps
        server_params["epsilon"] = eps

        if client_params.get("d") is None and server_params.get("d") is None:
            client_params["d"] = domain_size
            server_params["d"] = domain_size

        # for our simulations, we only have domains of the type {0, 1, 2, ..., domain_size-1}
        index_mapper = lambda x: x

        client_params["index_mapper"] = index_mapper
        server_params["index_mapper"] = index_mapper

        self.server = create_fo_server_instance(protocol.value, server_params)

        # Some clients need hash functions as parameters
        try:
            hash_funcs = self.server.get_hash_funcs()
            client_params["hash_funcs"] = hash_funcs
        except AttributeError:
            pass

        try:
            server_fo_hash_funcs = self.server.server_fo_hash_funcs
            client_params["server_fo_hash_funcs"] = server_fo_hash_funcs
        except AttributeError:
            pass

        self.client = create_fo_client_instance(protocol.value, client_params)

        self.normalization = 2  # use projection on the probability simplex

    def estimate_frequencies(self, t: np.array) -> np.array:
        """
        Estimate the frequencies of the elements in t.

        Args:
            t: The input data.

        Returns:
            The estimated frequencies of the elements in t.
        """
        for i, item in enumerate(t):
            self.server.aggregate(self.client.privatise(item))

        return self.server.estimate_all(np.arange(self.domain_size), normalization=self.normalization) / len(t)

    def response(self, t: np.array) -> np.array:
        pass

    def frequencies(self, u: np.array) -> np.array:
        pass
