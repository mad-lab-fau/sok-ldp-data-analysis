import numpy as np

from sok_ldp_analysis.simulation.data.synthetic import SynthDataset


class UniformSmallData1D(SynthDataset):
    def __init__(self, rng, size):
        self.size = size
        input_range = (0, 1)

        self.pop_mean = 0.5
        self.pop_sigma = (input_range[1] - input_range[0]) / np.sqrt(12)

        super().__init__(rng, input_range)

    def __str__(self):
        return f"{self.__class__.__name__}"

    def load_data(self):
        self.data_ = self.rng.uniform(self.input_range[0], self.input_range[1], self.size)


class UniformLargeData1D(SynthDataset):
    def __init__(self, rng, size):
        self.size = size
        input_range = (-100, 100)

        self.pop_mean = 0
        self.pop_sigma = (input_range[1] - input_range[0]) / np.sqrt(12)

        super().__init__(rng, input_range)

    def __str__(self):
        return f"{self.__class__.__name__}"

    def load_data(self):
        self.data_ = self.rng.uniform(self.input_range[0], self.input_range[1], self.size)


class BimodalData1D(SynthDataset):
    def __init__(self, rng, size, scale, shift):
        self.size = size
        self.scale = scale
        self.shift = shift

        pop_mean = (0.3 + 0.6) / 2
        pop_sigma = np.sqrt(0.5 * (0.1**2 + 0.3**2) + 0.5 * (0.6**2 + 0.2**2) - pop_mean**2)
        self.pop_mean = scale * (pop_mean + 0.5) / 2 + shift
        self.pop_sigma = scale * pop_sigma / 2

        super().__init__(rng, (shift, shift + scale))

    def __str__(self):
        return f"{self.__class__.__name__}({self.scale}, {self.shift})"

    def load_data(self):
        sample1 = self.rng.normal(0.3, 0.1, self.size)
        sample2 = self.rng.normal(0.6, 0.2, self.size)

        samples = np.concatenate((sample1, sample2))

        sample = self.rng.choice(samples, size=self.size)

        # scale the data to ensure that it fits inside [0,1] without having to clip too much
        sample = (sample + 0.5) / 2

        self.data_ = np.clip(sample, 0, 1) * self.scale + self.shift
