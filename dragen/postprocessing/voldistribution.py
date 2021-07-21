import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.neighbors import KernelDensity


class PostProcVol:
    def __init__(self, output_path, dim_flag):

        self.dimension = dim_flag
        self.output_path = output_path
        self.store_path = output_path + '/Postprocessing'

        if not os.path.isdir(output_path + '/Postprocessing'):
            os.mkdir(output_path + '/Postprocessing')

    def gen_in_out_lists(self) -> tuple:

        input_df = pd.read_csv(self.output_path + '/Generation_Data/grain_data_input.csv')

        # process total input data
        if self.dimension == 2:
            input_df['r_ref_conti'] = np.sqrt(input_df['final_conti_volume'] / np.pi)
            input_df['r_ref_discrete'] = np.sqrt(input_df['final_discrete_volume'] / np.pi)
        if self.dimension == 3:
            input_df['r_ref_conti'] = np.cbrt(3 * input_df['final_conti_volume'] / (4 * np.pi))
            input_df['r_ref_discrete'] = np.cbrt(3 * input_df['final_discrete_volume'] / (4 * np.pi))
        total_vol_conti_in = sum(input_df['final_conti_volume'].to_numpy().flatten().tolist())
        total_vol_discrete_in = sum(input_df['final_discrete_volume'].to_numpy().flatten().tolist())

        # process phase1 input data
        phase1_input_df = input_df.loc[input_df['phaseID'] == 1]
        # conti data
        phase1_input_list_conti = phase1_input_df['final_conti_volume'].to_numpy().flatten().tolist()
        phase1_vol_conti_in = sum(phase1_input_list_conti)
        phase1_ratio_conti_in = phase1_vol_conti_in / total_vol_conti_in
        phase1_ref_r_conti_in = phase1_input_df['r_ref_conti'].to_numpy().flatten().tolist()

        # discrete data
        phase1_input_list_discrete = phase1_input_df['final_discrete_volume'].to_numpy().flatten().tolist()
        phase1_vol_discrete_in = sum(phase1_input_list_discrete)
        phase1_ratio_discrete_in = phase1_vol_discrete_in / total_vol_discrete_in
        phase1_ref_r_discrete_in = phase1_input_df['r_ref_discrete'].to_numpy().flatten().tolist()

        # process phase2 input data
        phase2_input_df = input_df.loc[input_df['phaseID'] == 2]
        # conti data
        phase2_input_list_conti = phase2_input_df['final_conti_volume'].to_numpy().flatten().tolist()
        phase2_vol_conti_in = sum(phase2_input_list_conti)
        phase2_ratio_conti_in = phase2_vol_conti_in / total_vol_conti_in
        phase2_ref_r_conti_in = phase2_input_df['r_ref_conti'].to_numpy().flatten().tolist()

        # discrete data
        phase2_input_list_discrete = phase2_input_df['final_discrete_volume'].to_numpy().flatten().tolist()
        phase2_vol_discrete_in = sum(phase2_input_list_discrete)
        phase2_ratio_discrete_in = phase2_vol_discrete_in / total_vol_discrete_in
        phase2_ref_r_discrete_in = phase2_input_df['r_ref_discrete'].to_numpy().flatten().tolist()

        # process conti output
        output_df_conti = pd.read_csv(self.output_path + '/Generation_Data/grain_data_output_conti.csv')
        # process discrete output
        output_df_discrete = pd.read_csv(self.output_path + '/Generation_Data/grain_data_output_discrete.csv')

        if self.dimension == 2:
            output_df_conti['r_ref_conti'] = np.sqrt(output_df_conti['meshed_conti_volume'] / np.pi)
            output_df_discrete['r_ref_discrete'] = np.sqrt(output_df_discrete['final_discrete_volume'] / np.pi)
        if self.dimension == 3:
            output_df_conti['r_ref_conti'] = np.cbrt(3 * output_df_conti['meshed_conti_volume'] / (4 * np.pi))
            output_df_discrete['r_ref_discrete'] = np.cbrt(
                3 * output_df_discrete['final_discrete_volume'] / (4 * np.pi))
        total_vol_conti_out = sum(output_df_conti['meshed_conti_volume'].to_numpy().flatten().tolist())
        print(output_df_conti[['meshed_conti_volume', 'phaseID']].head())
        print(total_vol_conti_out)
        total_vol_discrete_out = sum(output_df_discrete['final_discrete_volume'].to_numpy().flatten().tolist())

        # process phase1 output data
        phase1_output_df_conti = output_df_conti.loc[output_df_conti['phaseID'] == 1]
        phase1_output_df_discrete = output_df_discrete.loc[output_df_conti['phaseID'] == 1]
        # conti data
        phase1_output_list_conti = phase1_output_df_conti['meshed_conti_volume'].to_numpy().flatten().tolist()
        phase1_vol_conti_out = sum(phase1_output_list_conti)
        print(phase1_vol_conti_out)
        phase1_ratio_conti_out = phase1_vol_conti_out / total_vol_conti_out
        phase1_ref_r_conti_out = phase1_output_df_conti['r_ref_conti'].to_numpy().flatten().tolist()
        # discrete data

        phase1_output_list_discrete = phase1_output_df_discrete['final_discrete_volume'].to_numpy().flatten().tolist()
        phase1_vol_discrete_out = sum(phase1_output_list_discrete)
        phase1_ratio_discrete_out = phase1_vol_discrete_out / total_vol_discrete_out
        phase1_ref_r_discrete_out = phase1_output_df_discrete['r_ref_discrete'].to_numpy().flatten().tolist()

        # process phase2 input data
        phase2_output_df_conti = output_df_conti.loc[output_df_conti['phaseID'] == 2]
        phase2_output_df_discrete = output_df_discrete.loc[output_df_conti['phaseID'] == 2]
        # conti data
        phase2_output_list_conti = phase2_output_df_conti['meshed_conti_volume'].to_numpy().flatten().tolist()
        phase2_vol_conti_out = sum(phase2_output_list_conti)
        print(phase2_vol_conti_out)
        phase2_ratio_conti_out = phase2_vol_conti_out / total_vol_conti_out
        phase2_ref_r_conti_out = phase2_output_df_conti['r_ref_conti'].to_numpy().flatten().tolist()
        # discrete data
        phase2_output_list_discrete = phase2_output_df_discrete['final_discrete_volume'].to_numpy().flatten().tolist()
        phase2_vol_discrete_out = sum(phase2_output_list_discrete)
        phase2_ratio_discrete_out = phase2_vol_discrete_out / total_vol_discrete_out
        phase2_ref_r_discrete_out = phase2_output_df_discrete['r_ref_discrete'].to_numpy().flatten().tolist()

        return phase1_ratio_conti_in, phase1_ref_r_conti_in, phase1_ratio_discrete_in, phase1_ref_r_discrete_in, \
               phase2_ratio_conti_in, phase2_ref_r_conti_in, phase2_ratio_discrete_in, phase2_ref_r_discrete_in, \
               phase1_ratio_conti_out, phase1_ref_r_conti_out, phase1_ratio_discrete_out, phase1_ref_r_discrete_out, \
               phase2_ratio_conti_out, phase2_ref_r_conti_out, phase2_ratio_discrete_out, phase2_ref_r_discrete_out

    def gen_pie_chart_phases(self, phase1_ratio, phase2_ratio, title):
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'Phase1', 'Phase2'
        sizes = [phase1_ratio, phase2_ratio]
        explode = (0.1, 0)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.set_title(title)

        fig1.savefig(self.store_path + '/phase_distribution_{}.png'.format(title))
        plt.close()

    def gen_plots(self, input, output, title, filename, input_opt=None, output_opt=None, title_opt=None, ) -> None:

        kde_1_in = KernelDensity(bandwidth=.25, kernel='gaussian')
        kde_1_out = KernelDensity(bandwidth=.25, kernel='gaussian')
        kde_1_in.fit(np.reshape(input, (-1, 1)))
        kde_1_out.fit(np.reshape(output, (-1, 1)))

        # capture x-lim
        max_x = 1.25 * 1.25 * max([int(max(input)), int(max(output))])
        if input_opt is not None and output_opt is not None:
            max_x = 1.25 * max([int(max(input)), int(max(output)), int(max(input_opt)), int(max(output_opt))])

        x_d = np.linspace(0, max_x, 50000)
        # score_samples returns the log of the probability density
        logdens_1_in = kde_1_in.score_samples(x_d[:, None])
        logdens_1_out = kde_1_out.score_samples(x_d[:, None])

        ##
        fig = plt.figure(figsize=(11, 8))

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_d, np.exp(logdens_1_in), 'k', label='input data ' + title)
        ax.plot(x_d, np.exp(logdens_1_out), '--k', label='output data ' + title)
        plt.xlabel('Grain Radius of Reference Sphere (µm)', fontsize=20)
        plt.ylabel('Normalized Density ( - ) ', fontsize=20)
        # caputure max y-lim
        max_log_dens = [max(np.exp(logdens_1_in)), max(np.exp(logdens_1_out))]

        if input_opt is not None and output_opt is not None:
            kde_2_in = KernelDensity(bandwidth=.25, kernel='gaussian')
            kde_2_out = KernelDensity(bandwidth=.25, kernel='gaussian')
            kde_2_in.fit(np.reshape(input_opt, (-1, 1)))
            kde_2_out.fit(np.reshape(output_opt, (-1, 1)))
            logdens_2_in = kde_2_in.score_samples(x_d[:, None])
            logdens_2_out = kde_2_out.score_samples(x_d[:, None])
            ax.plot(x_d, np.exp(logdens_2_in), 'r', label='input data ' + title_opt)
            ax.plot(x_d, np.exp(logdens_2_out), '--r', label='output data ' + title_opt)
            # caputure max new y-lim
            max_log_dens = [max(np.exp(logdens_1_in)), max(np.exp(logdens_1_out)),
                            max(np.exp(logdens_2_in)), max(np.exp(logdens_2_out))]

        max_y = max(max_log_dens)
        max_y = 1.25 * max_y
        plt.xlim(0, max_x)
        plt.ylim(0, max_y)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=12)
        plt.savefig(self.store_path + '/vol_distribution_{}.png'.format(filename))
        plt.close()


if __name__ == "__main__":
    outputPath = 'C:/temp/OutputData/2021-05-10_0'
    dim_flag = 3
    obj = PostProcVol(outputPath, dim_flag)
    obj.gen_stack_plot()
