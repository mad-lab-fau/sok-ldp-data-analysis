import itertools
import os
import pathlib
import posixpath
from abc import abstractmethod
from urllib.parse import urlparse, urlsplit
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


class RealDataset:
    def __init__(self, data_rng, raw_file_path, filename, data_path, input_range=(0, 1), domain_size=None):
        self.data_rng = data_rng
        self.raw_file_path = raw_file_path
        self.data_path = pathlib.Path(data_path)
        self.filename = filename
        if filename is None:
            self.file_path = None
        else:
            self.file_path = self.data_path / self.filename

        self.domain_size = domain_size
        if domain_size is not None:
            self.input_range = (0, domain_size)

        self.data_df = None
        self.input_range = input_range

        self._prepare_data()

    def __str__(self):
        return f"{self.__class__.__name__}"

    def _prepare_data(self):
        # check if the file was already prepared
        if self.file_path is not None and os.path.exists(self.file_path):
            return

        if self.raw_file_path is None:
            return

        is_url = urlparse(self.raw_file_path).scheme != ""
        if is_url:
            tmp_file = self._download(self.raw_file_path)
        else:
            tmp_file = self.raw_file_path

        print(f"({self.__class__.__name__}) Preprocessing {tmp_file}...")
        self._preprocess(tmp_file)

        # remove the temporary file
        if is_url:
            os.remove(tmp_file)

    def _download(self, url):
        if not url:
            return None

        url_path = urlsplit(url).path
        filename = posixpath.basename(url_path)

        download_path = self.data_path / "tmp" / filename

        print(f"({self.__class__.__name__}) Downloading {url} to {download_path}...")
        if not os.path.exists(self.data_path / "tmp"):
            os.makedirs(self.data_path / "tmp")

        tmp_file = download_path
        urlretrieve(url, tmp_file)
        return tmp_file

    @abstractmethod
    def _preprocess(self, tmp_file):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @property
    def data(self) -> np.array:
        # if data is already loaded, return it
        if self.data_df is not None:
            return self.data_df

        # if file path is not set, load the data (using some Dataset specific method)
        if self.file_path is None:
            self.load_data()
            return self.data_df

        # if file path does not exist, prepare the data
        if not os.path.exists(self.file_path):
            self._prepare_data()

        # load the data
        self.load_data()
        return self.data_df


class MultiDatasetReal(RealDataset):
    def _preprocess(self, tmp_file):
        pass

    def load_data(self):
        self.base_dataset.load_data()
        self.data_df = np.tile(self.base_dataset.data[: self.n], (self.d, 1)).T

    def __init__(self, n, base_dataset, d):
        self.base_dataset = base_dataset
        self.d = d
        self.n = n

        super().__init__(
            None,
            None,
            base_dataset.data_path,
            input_range=base_dataset.input_range,
            domain_size=base_dataset.domain_size,
        )
# US census data (1990) https://archive.ics.uci.edu/dataset/116/us+census+data+1990
class USCensus(RealDataset):
    """
    Dataset containing US census data from 1990. Available at https://archive.ics.uci.edu/dataset/116/us+census+data+1990.

    This dataset was used by [1]. The paper uses age, sex, income, and marital status and combines them to create a
    single categorical variable (with dimension 400 = 8 x 2 x 5 x 5). We use the same data.

    [1] T. Murakami, H. Hino, and J. Sakuma, “Toward Distribution Estimation under Local Differential Privacy with Small Samples,” Proceedings on Privacy Enhancing Technologies, 2018, doi: 10.1515/popets-2018-0022.
    """

    def __init__(self, data_rng, data_path="data"):
        url = "../data/tmp/USCensus1990.data.txt"
        filename = "us_census_1990.csv"

        super().__init__(data_rng, url, filename, data_path, domain_size=400)

    def _preprocess(self, tmp_file):
        df = pd.read_csv(tmp_file)
        df = df[
            [
                "dAge",
                "iSex",
                "dIncome1",
                "iMarital",
            ]
        ]

        # add a column with the combined categorical variable
        df["combined"] = df["dAge"] * 1000 + df["iSex"] * 100 + df["dIncome1"] * 10 + df["iMarital"]

        # map the combined variable to [400]
        all_options = itertools.product(range(8), range(2), range(5), range(5))
        mapping = {
            option[0] * 1000 + option[1] * 100 + option[2] * 10 + option[3]: i for i, option in enumerate(all_options)
        }
        df["combined"] = df["combined"].map(mapping)

        df.to_csv(self.file_path, index=False)

    def load_data(self):
        if not os.path.exists(self.file_path):
            self._prepare_data()

        self.data_df = pd.read_csv(self.file_path)["combined"].to_numpy()
        self.data_df = self.data_rng.permutation(self.data_df)


# UCI machine learning repository
# - Adult
class Adult(RealDataset):
    """
    Dataset containing census data from the UCI machine learning repository.
    https://archive.ics.uci.edu/dataset/2/adult

    This dataset was used by [1]. We use the same data and focus on the age column only. The paper used it for
    regression and kernel density estimation, but we want to use it for mean or frequency estimation.

    [1] F. Farokhi, “Deconvoluting kernel density estimation and regression for locally differentially private data,”
    Sci Rep, vol. 10, no. 1, Art. no. 1, Dec. 2020, doi: 10.1038/s41598-020-78323-0.
    """

    def __init__(self, data_rng, data_path="data"):
        url = None
        filename = "adult.csv"

        super().__init__(data_rng, url, filename, data_path, input_range=(16, 100), domain_size=100)

    def _preprocess(self, tmp_file):
        adult = fetch_ucirepo(id=2)
        df = adult.data.features
        df.to_csv(self.file_path, index=False)

    def load_data(self):
        adult = fetch_ucirepo(id=2)
        self.data_df = adult.data.features["age"]
        if not os.path.exists(self.file_path):
            self._preprocess(None)

        df = pd.read_csv(self.file_path)
        self.data_df = df["age"].values
        self.data_df = self.data_rng.permutation(self.data_df)




# New York City Taxi Data
class NYCTaxiData(RealDataset):
    """
    Dataset containing taxi data from New York City.

    This dataset was used by [1]. They use data from January 2018 and focus on the pick-up time (time of day in seconds).
    We use the same data and focus on the pick-up time as well, but we do not map it to [0,1].

    Data is available at https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page.


    [1] Z. Li, T. Wang, M. Lopuhaä-Zwakenberg, N. Li, and B. Škoric, “Estimating Numerical Distributions under Local Differential Privacy,” in Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data, in SIGMOD ’20. New York, NY, USA: Association for Computing Machinery, May 2020, pp. 621–635. doi: 10.1145/3318464.3389700.
    """

    def __init__(self, data_rng, data_path="data"):
        url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2018-01.parquet"
        filename = "nyc_taxi_yellow_tripdata_2018-01.csv"

        super().__init__(data_rng, url, filename, data_path, input_range=(0, 24 * 60 * 60))

    def _preprocess(self, tmp_file):
        df = pd.read_parquet(tmp_file)
        df = df[["tpep_pickup_datetime"]]
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_pickup_datetime"] = (
            df["tpep_pickup_datetime"].dt.hour * 60 * 60
            + df["tpep_pickup_datetime"].dt.minute * 60
            + df["tpep_pickup_datetime"].dt.second
        )
        df.to_csv(self.file_path, index=False)

    def load_data(self):
        self.data_df = pd.read_csv(self.file_path)["tpep_pickup_datetime"].values
        self.data_df = self.data_rng.permutation(self.data_df)


# San Francisco retirement data (Kaggle)
class SFRetirementData(RealDataset):
    """
    Dataset containing employee compensation data from San Francisco. Available at https://www.kaggle.com/datasets/san-francisco/sf-employee-compensation.

    This dataset was used by [1]. Similar to [1], we focus on the retirement compensation column only and remove values
    < 0 and > 60000.

    [1] Z. Li, T. Wang, M. Lopuhaä-Zwakenberg, N. Li, and B. Škoric, “Estimating Numerical Distributions under Local Differential Privacy,” in Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data, in SIGMOD ’20. New York, NY, USA: Association for Computing Machinery, May 2020, pp. 621–635. doi: 10.1145/3318464.3389700.
    """

    def __init__(self, data_path="data"):
        tmp_path = "data/tmp/employee-compensation.csv"
        filename = "sf_retirement.csv"

        # Check if the tmp file exists
        if not os.path.exists(tmp_path):
            raise FileNotFoundError(
                f"File {tmp_path} does not exist. Please download the file from Kaggle at"
                f"https://www.kaggle.com/datasets/san-francisco/sf-employee-compensation and place it in {tmp_path}."
            )

        super().__init__(tmp_path, filename, data_path, input_range=(0, 60000))

    def _preprocess(self, tmp_file):
        df = pd.read_csv(tmp_file)
        df = df[["Retirement"]]
        # Remove values < 0 and > 60000
        df = df[df["Retirement"] > 0]
        df = df[df["Retirement"] < 60000]
        df.to_csv(self.file_path, index=False)

    def load_data(self):
        if not os.path.exists(self.file_path):
            self._prepare_data()

        self.data_df = pd.read_csv(self.file_path)["Retirement"].values
