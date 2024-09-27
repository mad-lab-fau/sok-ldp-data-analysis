from abc import abstractmethod, ABC

import numpy as np


class SynthDataset(ABC):
    def __init__(self, rng, input_range, domain_size=None):
        self.data_ = None
        self.rng = rng
        self.input_range = input_range
        self.domain_size = domain_size

    def __str__(self):
        return f"{self.__class__.__name__}({len(self.data)}, {self.input_range})"

    @property
    def data(self):
        if self.data_ is None:
            self.load_data()

        return self.data_

    @abstractmethod
    def load_data(self):
        pass


class MixedDataMultiDim(SynthDataset):
    def __init__(self, rng, size):
        from sok_ldp_analysis.simulation.data.synthetic_continuous import UniformLargeData1D, BimodalData1D
        from sok_ldp_analysis.simulation.data.synthetic_discrete import BinomialData1D

        self.datasets = [
            UniformLargeData1D(rng, size),
            BinomialData1D(rng, size, 100, 0.5),
            BimodalData1D(rng, size, 1, 0),
        ]

        input_range = (
            min([dataset.input_range[0] for dataset in self.datasets]),
            max([dataset.input_range[1] for dataset in self.datasets]),
        )

        super().__init__(rng, input_range)

    def __str__(self):
        return f"{self.__class__.__name__}"

    def load_data(self):
        self.data_ = np.vstack([dataset.data for dataset in self.datasets]).T


class MultiDatasetSynth(SynthDataset):
    def load_data(self):
        self.base_dataset.load_data()
        self.data_ = np.tile(self.base_dataset.data[: self.n], (self.d, 1)).T

    def __init__(self, n, base_dataset, d):
        self.base_dataset = base_dataset
        self.d = d
        self.n = n

        super().__init__(base_dataset.rng, base_dataset.input_range, base_dataset.domain_size)

    def __str__(self):
        return f"{self.__class__.__name__}({self.base_dataset}, {self.d})"
