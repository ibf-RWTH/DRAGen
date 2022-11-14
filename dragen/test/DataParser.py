import sys
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from dragen.utilities.InputInfo import RveInfo
from scipy.stats import lognorm, truncnorm, uniform
import matplotlib.pyplot as plt
from typing import Tuple


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