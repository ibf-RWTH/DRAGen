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


def block_data_parser() -> SubsDistribution:
    if RveInfo.block_file is not None:
        block_data = pd.read_csv(RveInfo.block_file)
        kFold = KFold(n_splits=10)  # k-folder cross-validation split data into 10 folds
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(estimator=KernelDensity(kernel="gaussian"),
                            param_grid={"bandwidth": bandwidths},
                            cv=kFold)
        grid.fit(np.array(block_data['block_thickness']).reshape(-1, 1))
        return SubsDistribution(grid.best_estimator_)
    else:
        return SubsDistribution(lognorm(s=RveInfo.b_sigma, scale=RveInfo.t_mu))  # check later


def bt_sampler(bt_distribution: SubsDistribution, total_bt: float, interval: list) -> list: # check later
    # set to be minimal block thickness at the beginning
    bts = [interval[0]]
    n = 0
    while total_bt - sum(bts) > interval[0]:
        n = n + 1
        bt_star = truncnorm.rvs(a=(interval[0] - bts[n - 1]) / 0.5, b=(interval[1] - bts[n - 1]) / 0.5, loc=bts[n - 1],
                                scale=0.5,
                                size=1)  # trial sample from proposal distribution
        alpha = min(1, bt_distribution.pdf(x=bt_star[0])[0, 0] / bt_distribution.pdf(x=bts[n - 1])[
            0, 0])  # accept probability
        u = uniform().rvs(1)[0]
        if u < alpha:  # accept sample
            if sum(bts) + bt_star[0] > total_bt:  # ensure the sum of bts equal total bt
                new_bt = total_bt - sum(bts)
            else:
                new_bt = bt_star[0]
        else:
            if sum(bts) + bts[n - 1] > total_bt:  # ensure the sum of bts equal total bt
                new_bt = total_bt - sum(bts)
            else:
                new_bt = bts[n - 1]
        bts.append(new_bt)
    print("get bts successfully!")
    return bts


if __name__ == "__main__":
    RveInfo.block_file = r"/ExampleInput/example_block_inp.csv"
    start = time.time()
    block_data_parser()
    end = time.time()
    print("running time is ", end - start)
