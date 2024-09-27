from abc import abstractmethod, ABC


class SynthGaussianDataset(ABC):
    def __init__(self, rng, pop_mean, pop_sigma):
        self.data_ = None
        self.rng = rng
        self.pop_mean = pop_mean
        self.pop_sigma = pop_sigma

    def __str__(self):
        return f"{self.__class__.__name__}({len(self.data)}, {self.pop_mean}, {self.pop_sigma})"

    @property
    def data(self):
        if self.data_ is None:
            self.load_data()

        return self.data_

    @abstractmethod
    def load_data(self):
        pass


class GaussianData1D(SynthGaussianDataset):
    def __init__(self, rng, size, pop_mean, pop_sigma):
        self.size = size
        super().__init__(rng, pop_mean, pop_sigma)

    def __str__(self):
        return f"{self.__class__.__name__}({self.pop_mean}, {self.pop_sigma})"

    def load_data(self):
        self.rng.normal(self.pop_mean, self.pop_sigma, self.size)
