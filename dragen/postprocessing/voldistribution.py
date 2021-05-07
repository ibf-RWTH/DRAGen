import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
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
        if self.dimension == 3:
            discrete_input_df['r_ref'] = np.cbrt(3*discrete_input_df['final_discrete_volume']/(4*np.pi))
        discrete_input_list = discrete_input_df['r_ref'].to_numpy().flatten().tolist()

        discrete_output_df = pd.read_csv(self.output_path + '/Generation_Data/discrete_output_vol.csv')
        if self.dimension == 3:
            discrete_output_df['r_ref'] = np.cbrt(3*discrete_output_df['final_discrete_volume']/(4*np.pi))
        discrete_output_list = discrete_output_df['r_ref'].to_numpy().flatten().tolist()

        conti_input_df = pd.read_csv(self.output_path + '/Generation_Data/conti_input_vol.csv')
        if self.dimension == 3:
            conti_input_df['r_ref'] = np.cbrt(3*conti_input_df['final_conti_volume']/(4*np.pi))
        conti_input_list = conti_input_df['r_ref'].to_numpy().flatten().tolist()

        conti_output_df = pd.read_csv(self.output_path + '/Generation_Data/conti_output_vol.csv')
        if self.dimension == 3:
            conti_output_df['r_ref'] = np.cbrt(3*conti_output_df['final_conti_vol']/(4*np.pi))
        print(conti_output_df)
        conti_output_list = conti_output_df['r_ref'].to_numpy().flatten().tolist()


        return discrete_input_list, discrete_output_list, conti_input_list, conti_output_list

    def gen_plots(self) -> None:

        kde_disc_in = KernelDensity(bandwidth=.25, kernel='gaussian')
        kde_disc_out = KernelDensity(bandwidth=.25, kernel='gaussian')
        kde_conti_in = KernelDensity(bandwidth=.25, kernel='gaussian')
        kde_conti_out = KernelDensity(bandwidth=.25, kernel='gaussian')

        kde_disc_in.fit(np.reshape(self.discrete_input_list, (-1, 1)))
        kde_disc_out.fit(np.reshape(self.discrete_output_list, (-1, 1)))
        kde_conti_in.fit(np.reshape(self.conti_input_list, (-1, 1)))
        kde_conti_out.fit(np.reshape(self.conti_output_list, (-1, 1)))

        # capture x-lim
        max_x = int(max(self.conti_input_list))
        max_x = max_x+0.25*max_x
        x_d = np.linspace(0, max_x, 50000)

        # score_samples returns the log of the probability density
        logdens_d_in = kde_disc_in.score_samples(x_d[:, None])
        logdens_d_out = kde_disc_out.score_samples(x_d[:, None])
        logdens_c_in = kde_conti_in.score_samples(x_d[:, None])
        logdens_c_out = kde_conti_out.score_samples(x_d[:, None])

        ##
        fig = plt.figure(figsize=(11, 8))

        ax = fig.add_subplot(1,1,1)

        ax.set_xlabel('time [s]')
        ax.set_ylabel('signal')

        ax.plot(x_d, np.exp(logdens_d_in), '--r', label='discrete volume input data')
        ax.plot(x_d, np.exp(logdens_c_in), 'r', label='continuous volume input data')

        ax.plot(x_d, np.exp(logdens_d_out), '--k', label='discrete volume output data')
        ax.plot(x_d, np.exp(logdens_c_out), 'k', label='continuous volume output data')

        plt.xlabel('Grain Radius of Reference Sphere (µm)', fontsize=20)

        plt.ylabel('Normalized Density ( - ) ', fontsize=20)

        # caputure max y-lim
        max_log_dens = [max(np.exp(logdens_c_in)), max(np.exp(logdens_c_out)),
                        max(np.exp(logdens_d_in)), max(np.exp(logdens_d_out))]
        max_y = max(max_log_dens)
        max_y = max_y + 0.25*max_y

        plt.xlim(0, max_x)
        plt.ylim(0, max_y)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=12)
        plt.savefig(self.store_path+'/vol_distribution.png')

if __name__ == "__main__":

    outputPath = 'C:/temp/OutputData/2021-05-06_0'
    dim_flag = 3
    obj = PostProcVol(outputPath, dim_flag)
    obj.gen_in_out_lists()
    obj.gen_plots()