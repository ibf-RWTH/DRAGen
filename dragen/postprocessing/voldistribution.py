import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from sklearn.neighbors import KernelDensity


class PostProcVol:
    def __init__(self, output_path, dim_flag):

        self.dimension = dim_flag
        self.output_path = output_path
        self.store_path = output_path+'/Postprocessing'
        self.discrete_input_list, self.discrete_output_list, self.conti_input_list, self.conti_output_list = \
            self.gen_in_out_lists()

        if not os.path.isdir(output_path+'/Postprocessing'):
            os.mkdir(output_path+'/Postprocessing')

    def gen_in_out_lists(self) -> tuple:

        discrete_input_df = pd.read_csv(self.output_path+'/Generation_Data/discrete_input_vol.csv')
        discrete_input_list =  discrete_input_df.to_numpy().flatten().tolist()

        discrete_output_df = pd.read_csv(self.output_path + '/Generation_Data/discrete_output_vol.csv')
        discrete_output_list = discrete_output_df.to_numpy().flatten().tolist()

        conti_input_df = pd.read_csv(self.output_path + '/Generation_Data/conti_input_vol.csv')
        conti_input_list = conti_input_df.to_numpy().flatten().tolist()

        conti_output_df = pd.read_csv(self.output_path + '/Generation_Data/conti_output_vol.csv')
        conti_output_list = conti_output_df.to_numpy().flatten().tolist()

        return discrete_input_list, discrete_output_list, conti_input_list, conti_output_list

    def gen_plots(self) -> None:

        kde_disc_in = KernelDensity(bandwidth=10.0, kernel='gaussian')
        kde_disc_out = KernelDensity(bandwidth=10.0, kernel='gaussian')
        kde_conti_in = KernelDensity(bandwidth=10.0, kernel='gaussian')
        kde_conti_out = KernelDensity(bandwidth=10.0, kernel='gaussian')

        kde_disc_in.fit(np.reshape(self.discrete_input_list, (-1, 1)))
        kde_disc_out.fit(np.reshape(self.discrete_output_list, (-1, 1)))
        kde_conti_in.fit(np.reshape(self.conti_input_list, (-1, 1)))
        kde_conti_out.fit(np.reshape(self.conti_input_list, (-1, 1)))

        # capture x-lim
        max_vol = int(max(self.conti_input_list))
        max_vol = 500
        x_d = np.linspace(0, max_vol, 50000)

        # score_samples returns the log of the probability density
        logdens_d_in = kde_disc_in.score_samples(x_d[:, None])
        logdens_d_out = kde_disc_out.score_samples(x_d[:, None])
        logdens_c_in = kde_conti_in.score_samples(x_d[:, None])
        logdens_c_out = kde_conti_out.score_samples(x_d[:, None])

        ##
        plt.plot(x_d, np.exp(logdens_c_in), 'r', label='continuous volume input data')
        plt.plot(x_d, np.exp(logdens_d_in), '--r', label='discrete volume input data')

        plt.plot(x_d, np.exp(logdens_c_out), 'k', label='continuous volume output data')
        plt.plot(x_d, np.exp(logdens_d_out), '--k', label='discrete volume output data')
        if self.dimension == 2:
            plt.xlabel('Grain Volume [µm²]', fontsize=20)
        if self.dimension == 3:
            plt.xlabel('Grain Volume [µm³]', fontsize=20)
        plt.ylabel('Normalized Density [-] ', fontsize=20)

        # caputure max y-lim
        max_log_dens = [max(np.exp(logdens_c_in)), max(np.exp(logdens_c_out)),
                        max(np.exp(logdens_d_in)), max(np.exp(logdens_d_out))]
        max_y = max(max_log_dens)

        plt.xlim(0, max_vol)
        plt.ylim(0, max_y)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=12)
        plt.show()
        plt.savefig(self.store_path+'/vol_distribution.png', dpi=1200)

