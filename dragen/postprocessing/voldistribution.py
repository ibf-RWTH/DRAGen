import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
from dragen.utilities.InputInfo import RveInfo


class PostProcVol:
    def __init__(self):

        if not os.path.isdir(RveInfo.store_path + '/Postprocessing'):
            os.mkdir(RveInfo.store_path + '/Postprocessing')

    def gen_in_out_lists(self, phaseID) -> tuple:

        input_df = pd.read_csv(RveInfo.store_path + '/Generation_Data/experimental_data.csv')

        # process total input data
        if RveInfo.dimension == 2:
            input_df['r_ref'] = np.sqrt(input_df['volume'] / np.pi)
        if RveInfo.dimension == 3:
            input_df['r_ref'] = np.cbrt(3 * input_df['volume'] / (4 * np.pi))

        # process current_phase input data
        current_phase_input_df = input_df.loc[input_df['phaseID'] == phaseID]
        current_phase_ref_r_in = current_phase_input_df['r_ref'].to_numpy().flatten().tolist()


        # process output
        output_df = pd.read_csv(RveInfo.store_path + '/Generation_Data/grain_data_output.csv')

        if RveInfo.dimension == 2:
            output_df['r_ref'] = np.sqrt(output_df['final_discrete_volume'] / np.pi)
        if RveInfo.dimension == 3:
            output_df['r_ref'] = np.cbrt(3 * output_df['final_discrete_volume'] / (4 * np.pi))
        total_vol_out = sum(output_df['final_discrete_volume'].to_numpy().flatten().tolist())
        print(output_df[['final_discrete_volume', 'phaseID']].head())
        print(total_vol_out)

        # process current_phase output data
        current_phase_output_df_conti = output_df.loc[output_df['phaseID'] == phaseID]

        current_phase_output_list = current_phase_output_df_conti['final_discrete_volume'].to_numpy().flatten().tolist()
        current_phase_vol_out = sum(current_phase_output_list)
        current_phase_ratio_out = current_phase_vol_out / total_vol_out
        current_phase_ref_r_out = current_phase_output_df_conti['r_ref'].to_numpy().flatten().tolist()

        return current_phase_ref_r_in, current_phase_ratio_out, current_phase_ref_r_out


    def gen_pie_chart_phases(self, sizes, labels, title):
        explode = [0.05]*len(sizes)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90, normalize=False, colors=RveInfo.rwth_colors)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.set_title(title)

        fig1.savefig(RveInfo.store_path + '/Postprocessing/phase_distribution_{}.png'.format(title))
        plt.close()

    def gen_plots(self, input, output, label) -> None:

        kernel_in = stats.gaussian_kde(input, bw_method='scott')
        kernel_out = stats.gaussian_kde(output, bw_method='scott')

        # capture x-lim
        max_x = 1.25 * 1.25 * max([int(max(input)), int(max(output))])

        x_d = np.linspace(0, max_x, 5000)
        # score_samples returns the log of the probability density
        log_pdf_in = kernel_in.logpdf(x_d)
        log_pdf_out = kernel_out.logpdf(x_d)

        ##
        fig = plt.figure(figsize=(11, 8))

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_d, np.exp(log_pdf_in), label='input ' + label, c=RveInfo.rwth_colors[1])
        ax.plot(x_d, np.exp(log_pdf_out), label='output ' + label, c=RveInfo.rwth_colors[5])
        plt.xlabel('Grain Radius of Reference Sphere (µm)', fontsize=20)
        plt.ylabel('Normalized Density ( - ) ', fontsize=20)
        # caputure max y-lim
        max_log_dens = [max(np.exp(log_pdf_in)), max(np.exp(log_pdf_out))]

        max_y = max(max_log_dens)
        max_y = 1.25 * max_y
        plt.xlim(0, max_x)
        plt.ylim(0, max_y)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=12)
        plt.savefig(RveInfo.store_path + '/Postprocessing/size_distribution_{}.png'.format(label))
        plt.close()
