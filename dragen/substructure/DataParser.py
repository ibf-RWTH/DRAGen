import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from dragen.utilities.InputInfo import RveInfo
from scipy.stats import lognorm, truncnorm, uniform
import matplotlib.pyplot as plt
from typing import Tuple
import time


class SubsDistribution:
    def __init__(self, estimator):
        self.estimator = estimator

    def pdf(self, x: [float, np.ndarray]) -> np.ndarray:
        x = np.array(x).reshape(-1, 1)
        if isinstance(self.estimator, KernelDensity):
            result = np.exp(self.estimator.score_samples(x))
        else:
            result = self.estimator.pdf(x)

        return result.reshape(-1, 1)


class DataParser:
    def __init__(self):
        kFold = KFold(n_splits=10)  # k-folder cross-validation split data into 10 folds
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(estimator=KernelDensity(kernel="gaussian"),
                            param_grid={"bandwidth": bandwidths},
                            cv=kFold)
        self.grid = grid
        self.num_packets_list = []

    def parse_block_data(self):
        if RveInfo.block_file is not None:
            block_data = pd.read_csv(RveInfo.block_file)
            RveInfo.bt_min = block_data['block_thickness'].min()
            RveInfo.bt_max = block_data['block_thickness'].max()
            print("min bt is ", RveInfo.bt_min)
            print("max bt is ", RveInfo.bt_max)
            self.grid.fit(block_data['block_thickness'].to_numpy().reshape(-1, 1))
            return SubsDistribution(self.grid.best_estimator_)
        else:
            return SubsDistribution(lognorm(s=RveInfo.b_sigma, scale=RveInfo.t_mu))  # check later

    def parse_packet_data(self) -> SubsDistribution:
        """
        get the statistical distribution of packet volume
        :return: SubsDistribution of packet volume
        """
        files = RveInfo.file_dict
        file_idx = RveInfo.PHASENUM["Martensite"]
        pag_file = files[file_idx]
        if pag_file is not None:
            pag_data = pd.read_csv(pag_file)
            # get mean packet volumes
            # 1 count number of packets in each grain
            subs_data = pd.read_csv(RveInfo.block_file)
            self.num_packets_list = [len(subs_data.loc[subs_data['grain_id'] == i, 'packet_id'].unique()) for i in
                                     range(1, len(pag_data) + 1)]
            # 2 compute mean packet volumes
            pag_data['grain_id'] = [i for i in range(1, len(pag_data) + 1)]
            mean_pv = pag_data.apply(lambda grain: grain['volume'] / self.num_packets_list[int(grain['grain_id']) - 1],
                                     axis=1)
            RveInfo.pv_max = pag_data['volume'].max()
            RveInfo.pv_min = pag_data['volume'].min() / 2
            print("min pv is ", RveInfo.pv_min)
            print("max pv is ", RveInfo.pv_max)
            self.grid.fit(mean_pv.to_numpy().reshape(-1, 1))
            return SubsDistribution(self.grid.best_estimator_)
        else:
            r = RveInfo.equiv_d / 2
            average_pv = 4 / 3 * np.pi * r ** 3 * RveInfo.circularity ** (1.5)
            return SubsDistribution(lognorm(s=RveInfo.p_sigma, scale=average_pv))


def subs_sampler(subs_distribution: SubsDistribution, y: float, interval: list) -> list:  # check later
    """
    :param subs_distribution: distribution of packet volume or block thickness
    :param y: grain volume or packet thickness to be divided
    :param interval: the minimum and maximum during sampling
    :return: a list of packet volume or block thickness that follows subs_distribution
    """
    # set to be minimal block thickness at the beginning
    xs = [interval[0]]
    n = 0
    while y - sum(xs) > interval[0]:
        n = n + 1
        x_star = truncnorm.rvs(a=(interval[0] - xs[n - 1]) / 0.5, b=(interval[1] - xs[n - 1]) / 0.5, loc=xs[n - 1],
                               scale=0.5,
                               size=1)  # trial sample from proposal distribution
        alpha = min(1, subs_distribution.pdf(x=x_star[0])[0, 0] / subs_distribution.pdf(x=xs[n - 1])[
            0, 0])  # accept probability
        u = uniform().rvs(1)[0]
        if u < alpha:  # accept sample
            if sum(xs) + x_star[0] > y:  # ensure the sum of bts equal total bt
                new_x = y - sum(xs)
            else:
                new_x = x_star[0]
        else:
            if sum(xs) + xs[n - 1] > y:  # ensure the sum of bts equal total bt
                new_x = y - sum(xs)
            else:
                new_x = xs[n - 1]
        xs.append(new_x)
    print("get bts successfully!")
    return xs


if __name__ == "__main__":
    RveInfo.block_file = r"/ExampleInput/example_block_inp.csv"
    start = time.time()
    # block_data_parser()
    end = time.time()
    print("running time is ", end - start)
