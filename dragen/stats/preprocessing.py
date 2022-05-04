# _*_ coding: utf-8 _*_
"""
Time:     2022/5/3 10:22
Author:   Linghao Kong
Version:  V 0.1
File:     sample
Describe: Writing at home"""
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold,GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

class Sampler:
    def __init__(self,markov_matrix):
        self.markov_matrix = markov_matrix

    '''use acceptance-rejection sampling to achieve complex sampling'''
    @classmethod
    def rejection_sample(cls,intervals:list,pdf,c):
        while True:
            x = np.random.uniform(intervals[0], intervals[1])
            y = np.random.uniform(0,c)
            #if accpet x?
            if y <= pdf(np.array([x]).reshape((-1,1))):
                return x

class InputDataSampler(Sampler):
    '''fit the real distribution using KDE and sample from fitted distribution'''
    def __init__(self,data):
        super(InputDataSampler,self).__init__(None)
        self.data = data
        self.kde = self._train()
        self.intervals = [0, 0]#avoid future warning

    def _train(self):
        kFold = KFold(n_splits=10)  # k-folder cross-validation split data into 10 folds
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(estimator=KernelDensity(kernel="gaussian"),
                            param_grid={"bandwidth": bandwidths},
                            cv=kFold)
        grid.fit(self.data)
        kde = grid.best_estimator_
        return kde

    def pdf(self, x):
        return np.exp(self.kde.score_samples(np.array([x]).reshape(-1, 1)))

    def sample(self,intervals:list):
        if intervals[0] != self.intervals[0] or intervals[1] != self.intervals[1]:
            u = np.linspace(intervals[0], intervals[1], 1000)
            self.intervals = u
        else:
            u = self.intervals
        #transfer uniform distribution to the needed one by acceptance-rejection sampling
        #1.compute c
        log_dens = self.kde.score_samples(u.reshape(-1,1))
        c = np.max(np.exp(log_dens))*1.1
        #sample from uniform distribution in intervals
        x = super().rejection_sample(intervals = intervals,
                                     pdf = self.pdf,
                                     c = c)
        return x

class UserBlockThicknessSampler(Sampler):
    '''produce lognorm distribution'''
    def __init__(self, average_bt, sigma = 0.1):
        super(UserBlockThicknessSampler,self).__init__(None)
        self.average_bt = average_bt
        self.sigma = sigma
        self.intervals = [0, 0]

    def pdf(self, x):
        return lognorm.pdf(x, s=self.sigma, scale=self.average_bt)

    def sample(self,intervals:list):
        if intervals[0] != self.intervals[0] or intervals[1] != self.intervals[1]:
            u = np.linspace(intervals[0], intervals[1], 1000)
            self.intervals = u
        else:
            u = self.intervals
        u = np.linspace(intervals[0], intervals[1], 1000)
        dens = lognorm.pdf(u,s = self.sigma, scale = self.average_bt)
        c = np.max(dens)*1.1
        x = super().rejection_sample(intervals = intervals,
                                     pdf = self.pdf,
                                     c = c)
        return x

class UserPakVolumeSampler(Sampler):
    '''produce lognorm distribution'''
    def __init__(self,equiv_d,circularity = 1,sigma = 0.1):
        super(UserPakVolumeSampler,self).__init__(None)
        self.equiv_d = equiv_d
        self.circularity = circularity
        self.sigma = sigma
        self.average_volume = 4 / 3 * np.pi * (equiv_d/2) ** 3 * circularity ** (1.5)
        self.intervals = [0 ,0]

    def pdf(self,x):
        return lognorm.pdf(x, s=self.sigma, scale=self.average_volume)

    def sample(self, intervals:list):
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
    def __init__(self, type):
        self.type = type
    def create_sampler(self,sampler:[InputDataSampler,UserPakVolumeSampler,UserBlockThicknessSampler], **kwargs):
        return sampler(**kwargs)

def test_inputsampler():
    df = pd.read_csv("F:/pycharm/dragen/ExampleInput/example_pag_inp2.csv")
    data = df["volume"].to_numpy().reshape((-1,1))
    data = np.sort(data,axis=0)
    kFold = KFold(n_splits=10)  # k-folder cross-validation split data into 10 folds
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(estimator=KernelDensity(kernel="gaussian"),
                        param_grid={"bandwidth": bandwidths},
                        cv=kFold)
    grid.fit(data)
    kde = grid.best_estimator_
    log_dens = kde.score_samples(data)
    plt.plot(data,np.exp(log_dens),label="real")

    sampler = InputDataSampler(data)
    intervals = [np.min(data),np.max(data)]
    samples = []
    for i in range(100):
        samples.append(sampler.sample(intervals))

    grid.fit(np.array(samples).reshape((-1,1)))
    kde2 = grid.best_estimator_
    samples = np.sort(np.array(samples).reshape((-1,1)),axis=0)
    log_dens = kde2.score_samples(samples)
    plt.plot(samples,np.exp(log_dens),label="estimated")
    plt.legend()
    plt.show()

def test_usersampler():
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

if __name__ == "__main__":
    df = pd.read_csv("F:/pycharm/dragen/ExampleInput/example_pag_inp2.csv")
    data = df["volume"].to_numpy().reshape((-1, 1))
    data = np.sort(data, axis=0)
    input_sampler_creater = SamplerFactory("InputSampler")
    user_bt_sampler_creater = SamplerFactory("UserBtSampler")
    user_pv_sampler_creater = SamplerFactory("UserPvSampler")
    input_sampler = input_sampler_creater.create_sampler(InputDataSampler,data = data)
    user_bt_sampler = user_bt_sampler_creater.create_sampler(UserBlockThicknessSampler,average_bt = 1.5, sigma = 0.1)
    user_pv_sampler = input_sampler_creater.create_sampler(UserPakVolumeSampler,equiv_d = 4,circularity = 1,sigma = 0.1)










