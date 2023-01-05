# _*_ coding: utf-8 _*_
"""
Time:     2022/5/3 10:22
Author:   Linghao Kong
Version:  V 0.1
File:     sample
Describe: Writing for DRAGen"""

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from dragen.utilities.InputInfo import RveInfo

class Sampler:
    '''
    base Sampler class: markov_matrix may be introduced in the future to increase sampling efficiency,
    so far it is set to be None in childclass.
    '''

    def __init__(self, markov_matrix):
        self.markov_matrix = markov_matrix

    @classmethod
    def rejection_sample(cls, intervals: list, pdf, c: float):
        """
        use acceptance-rejection sampling to achieve complex sampling
        """
        if RveInfo.debug:
            assert len(intervals) == 2  # start and end
            assert c > 0
        while True:
            x = np.random.uniform(intervals[0], intervals[1])
            y = np.random.uniform(0, c)
            # if accpet x?
            if y <= pdf(np.array([x]).reshape((-1, 1))):
                return x


class InputDataSampler(Sampler):
    """
    fit the real distribution in the data using KDE and sample from fitted distribution
    """

    def __init__(self, data):
        super(InputDataSampler, self).__init__(None)
        if RveInfo.debug:
            assert len(data) > 0
        self.data = data
        self.kde = self._train()
        self.intervals = [0, 0]  # avoid future warning

    def _train(self):
        """
        fit the real distribution in the data using KDE. Use GridSearch method to find the most optimal
        bandwidth for KDE
        """
        kFold = KFold(n_splits=10)  # k-folder cross-validation split data into 10 folds
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(estimator=KernelDensity(kernel="gaussian"),
                            param_grid={"bandwidth": bandwidths},
                            cv=kFold)
        grid.fit(self.data)
        kde = grid.best_estimator_
        return kde

    def pdf(self, x):
        """
        probability density function for fitted distribution, return the density of the input x
        """
        if RveInfo.debug:
            assert x > 0
        return np.exp(self.kde.score_samples(np.array([x]).reshape(-1, 1)))

    def sample(self, intervals: list):
        """
        intervals: sample data from this invertal in the fitted distribution
        """

        if intervals[0] != self.intervals[0] or intervals[1] != self.intervals[1]:
            u = np.linspace(intervals[0], intervals[1], 1000)
            self.intervals = u
        else:
            u = self.intervals
        log_dens = self.kde.score_samples(u.reshape(-1, 1))
        c = np.max(np.exp(log_dens)) * 1.1
        x = super().rejection_sample(intervals=intervals,
                                     pdf=self.pdf,
                                     c=c)
        return x


class UserBlockThicknessSampler(Sampler):
    '''
    produce lognorm distribution based on user input: average_bt(average block thickness), sigma(standard variance)
    '''

    def __init__(self, average_bt, sigma=0.1):
        if RveInfo.debug:
            assert average_bt > 0 and sigma > 0
        super(UserBlockThicknessSampler, self).__init__(None)
        self.average_bt = average_bt
        self.sigma = sigma
        self.intervals = [0, 0]

    def pdf(self, x):
        """
        probability density function of produced lognorm distribution
        """
        if RveInfo.debug:
            assert x > 0
        return lognorm.pdf(x, s=self.sigma, scale=self.average_bt)

    def sample(self, intervals: list):
        """
        intervals: sample data from this invertal in the produced distribution
        """
        if intervals[0] != self.intervals[0] or intervals[1] != self.intervals[1]:
            u = np.linspace(intervals[0], intervals[1], 1000)
            self.intervals = u
        else:
            u = self.intervals
        dens = lognorm.pdf(u, s=self.sigma, scale=self.average_bt)
        c = np.max(dens) * 1.1
        x = super().rejection_sample(intervals=intervals,
                                     pdf=self.pdf,
                                     c=c)
        return x


class UserPakVolumeSampler(Sampler):
    """
    produce lognorm distribution
    """

    def __init__(self, equiv_d, circularity=1, sigma=0.1):
        super(UserPakVolumeSampler, self).__init__(None)
        if RveInfo.debug:
            assert equiv_d > 0 and circularity > 0 and sigma > 0
        self.equiv_d = equiv_d
        self.circularity = circularity
        self.sigma = sigma
        self.average_volume = 4 / 3 * np.pi * (equiv_d / 2) ** 3 * circularity ** (1.5)
        self.intervals = [0, 0]

    def pdf(self, x):
        """
        probability density function of produced lognorm distribution
        """
        if RveInfo.debug:
            assert x > 0
        return lognorm.pdf(x, s=self.sigma, scale=self.average_volume)

    def sample(self, intervals: list):
        """
        intervals: sample data from this invertal in the produced distribution
        """
        if intervals[0] != self.intervals[0] or intervals[1] != self.intervals[1]:
            u = np.linspace(intervals[0], intervals[1], 1000)
            self.intervals = u
        else:
            u = self.intervals
        dens = lognorm.pdf(u, s=self.sigma, scale=self.average_volume)
        c = np.max(dens) * 1.1
        x = super().rejection_sample(intervals=intervals,
                                     pdf=self.pdf,
                                     c=c)
        return x


class SamplerFactory:
    """
    Factory Design Pattern: avoid tons of if-else, produce sampler objects depending on passed parameters
    """

    def __init__(self, type):
        """
        type: the type of samplers that being produced
        """
        self.type = type

    @staticmethod
    def create_sampler(sampler: [InputDataSampler, UserPakVolumeSampler, UserBlockThicknessSampler],
                       **kwargs) -> [InputDataSampler, UserPakVolumeSampler, UserBlockThicknessSampler]:
        """
        sampler: must be one of the following samplers: InputDataSampler,UserPakVolumeSampler,UserBlockThicknessSampler.
        kwargs: the needed parameters for above mentioned 3 samplers. For example, the kwargs for UserBlockThicknessSampler
        should be average_bt and sigma(defaulted 0.1)
        """
        if RveInfo.debug:
            assert len(kwargs) > 0
        return sampler(**kwargs)


def test_inputsampler():
    """
    test function: test the InputDataSampler, the user usually needs to change the file path
    """
    df = pd.read_csv("F:/pycharm/dragen/ExampleInput/example_pag_inp2.csv")
    data = df["volume"].to_numpy().reshape((-1, 1))
    data = np.sort(data, axis=0)
    kFold = KFold(n_splits=10)  # k-folder cross-validation split data into 10 folds
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(estimator=KernelDensity(kernel="gaussian"),
                        param_grid={"bandwidth": bandwidths},
                        cv=kFold)
    grid.fit(data)
    kde = grid.best_estimator_
    log_dens = kde.score_samples(data)
    plt.plot(data, np.exp(log_dens), label="real")

    sampler = InputDataSampler(data)
    intervals = [np.min(data), np.max(data)]
    samples = []
    for i in range(100):
        samples.append(sampler.sample(intervals))

    grid.fit(np.array(samples).reshape((-1, 1)))
    kde2 = grid.best_estimator_
    samples = np.sort(np.array(samples).reshape((-1, 1)), axis=0)
    log_dens = kde2.score_samples(samples)
    plt.plot(samples, np.exp(log_dens), label="estimated")
    plt.legend()
    plt.show()


def test_usersampler():
    """
    test function: test the UserBlockThicknessSampler or UserPakVolumeSampler, the user usually needs to change the file path and the kind of sampler
    """
    pak_sampler = UserPakVolumeSampler(2)
    data = []
    for i in range(100):
        bt = pak_sampler.sample([0.5, 10])
        data.append(bt)

    data = np.array(data).reshape((-1, 1))
    data = np.sort(data, axis=0)
    kFold = KFold(n_splits=10)  # k-folder cross-validation split data into 10 folds
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(estimator=KernelDensity(kernel="gaussian"),
                        param_grid={"bandwidth": bandwidths},
                        cv=kFold)
    grid.fit(data)
    kde = grid.best_estimator_
    log_dens = kde.score_samples(data)
    plt.plot(data, np.exp(log_dens), label="sampled")
    plt.show()


def slice_to_distribution(num: [int, float],
                          intervals: list,
                          distribution: [InputDataSampler, UserPakVolumeSampler, UserBlockThicknessSampler]) -> list:
    """
    slice a passed number(block thickness or packet volume) into a series of number that follow tha passed distribution,
    the sum of which equals the passed number
    """
    if RveInfo.debug:
        assert num > 0 and len(intervals) == 2
    samples = []
    while True:
        sampled_num = distribution.sample(intervals=intervals)
        if num < sampled_num or num <= intervals[0]:
            break
        samples.append(sampled_num)
        num -= sampled_num
    samples.append(num)
    return samples


if __name__ == "__main__":
    # df = pd.read_csv("F:/pycharm/dragen/ExampleInput/example_pag_inp2.csv")
    # data = df["volume"].to_numpy().reshape((-1, 1))
    # data = np.sort(data, axis=0)
    # input_sampler_creater = SamplerFactory("InputSampler")
    # user_bt_sampler_creater = SamplerFactory("UserBtSampler")
    # user_pv_sampler_creater = SamplerFactory("UserPvSampler")
    # input_sampler = input_sampler_creater.create_sampler(InputDataSampler, data=data)
    # user_bt_sampler = user_bt_sampler_creater.create_sampler(UserBlockThicknessSampler, average_bt=1.5, sigma=0.1)
    # user_pv_sampler = input_sampler_creater.create_sampler(UserPakVolumeSampler, equiv_d=4, circularity=1, sigma=0.1)
    test_sampler_creater = SamplerFactory("TestSampler")
    test_sampler = test_sampler_creater.create_sampler(UserBlockThicknessSampler, average_bt=1.5, sigma=0.1)
    samples = slice_to_distribution(100, [0.5, 2], test_sampler)
    print(samples)
    data = np.array(samples).reshape((-1, 1))
    data = np.sort(data, axis=0)
    kFold = KFold(n_splits=10)  # k-folder cross-validation split data into 10 folds
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(estimator=KernelDensity(kernel="gaussian"),
                        param_grid={"bandwidth": bandwidths},
                        cv=kFold)
    grid.fit(data)
    kde = grid.best_estimator_
    log_dens = kde.score_samples(data)
    plt.plot(data, np.exp(log_dens), label="test_sample")
    plt.show()


