from sok_ldp_analysis.ldp.mean.ding2017 import Ding2017
from sok_ldp_analysis.ldp.mean.duchi2018 import Duchi2018, Duchi2018LInf
from sok_ldp_analysis.ldp.mean.laplace import Laplace1D
from sok_ldp_analysis.ldp.mean.nguyen2016harmony import Nguyen2016
from sok_ldp_analysis.ldp.mean.wang2019 import Wang2019Piecewise1D, Wang2019Hybrid1D, Wang2019Duchi1D


class Ding2017LargeM(Ding2017):
    def __init__(self, eps, rng, input_range=(0, 1)):
        super().__init__(eps=eps, rng=rng, input_range=input_range, m=input_range[1] - input_range[0])


class Ding2017FixedM(Ding2017):
    def __init__(self, eps, rng, input_range=(0, 1)):
        super().__init__(eps=eps, rng=rng, input_range=input_range, m=1)


one_dim_mean_methods = [Laplace1D, Ding2017LargeM, Wang2019Piecewise1D, Wang2019Duchi1D, Wang2019Hybrid1D]
multi_dim_mean_methods = [Nguyen2016, Duchi2018, Duchi2018LInf]
