import numpy as np

from sok_ldp_analysis.simulation.data.synthetic import SynthDataset


class BinomialData1D(SynthDataset):
    def __init__(self, rng, size, domain_size, p):
        self.size = size

        self.domain_size = domain_size
        self.p = p

        n = domain_size - 1

        self.pop_mean = n * p
        self.pop_sigma = np.sqrt(n * p * (1 - p))

        super().__init__(rng, (0, domain_size), domain_size)

    def __str__(self):
        return f"{self.__class__.__name__}({self.domain_size}, {self.p})"

    def load_data(self):
        self.data_ = self.rng.binomial(self.domain_size - 1, self.p, self.size)


class GeometricData1D(SynthDataset):
    def __init__(self, rng, size, domain_size, p):
        self.size = size

        # This is the distribution used in Kairouz et al. 2016
        if p is None:
            p = 5 / domain_size

        self.p = p
        super().__init__(rng, (0, domain_size), domain_size)

    def __str__(self):
        return f"{self.__class__.__name__}({self.domain_size, self.p})"

    def load_data(self):
        # The geometric distribution is defined over the integers 0, 1, 2, ... but we want a finite domain
        # so we need to truncate the distribution at domain_size
        probabilities = [self.p * (1 - self.p) ** (i - 1) for i in range(1, self.domain_size + 1)]

        # normalize the probabilities to sum to 1
        probabilities = np.array(probabilities) / sum(probabilities)

        self.data_ = self.rng.choice(np.arange(0, self.domain_size), self.size, p=probabilities)
