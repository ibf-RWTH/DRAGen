import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

'''
Part 1: Class for sampling band data (filling + width), both not normalized
TODO: Erstmal zurückstellen.
'''


class BandSampler:

    def __init__(self, path, fit=False):
        np.random.seed(0)
        self.data = pd.read_csv(path).to_numpy()
        self.filling = self.data['Filling'].to_numpy()
        self.width = self.data['Width'].to_numpy()
        self.fitted = fit
        if self.fitted:
            self.fit_distribution()

    def sample(self, size):
        if self.fitted:
            # Implement later
            pass
        else:
            idx = np.random.randint(0, self.data.__len__(), size=size)
            sample_f = np.expand_dims(self.filling[idx], axis=1)
            sample_w = np.expand_dims(self.width[idx], axis=1)
            sample = np.concatenate([sample_f, sample_w], axis=1)
        return sample

    def fit_distribution(self):
        params = stats.lognorm.fit(self.filling)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        x = np.arange(self.filling.__len__())
        if arg:
            pdf_fitted = stats.lognorm.pdf(x, *arg, loc=loc, scale=scale) * self.filling.__len__()
        else:
            pdf_fitted = stats.lognorm.pdf(x, loc=loc, scale=loc) * self.filling.__len__()
        # g = sns.histplot(x=self.filling)
        plt.plot(pdf_fitted)