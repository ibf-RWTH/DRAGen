import sys
import math
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dragen.utilities.Helpers import HelperFunctions
from dragen.generation.DiscreteRsa2D import DiscreteRsa2D
from dragen.generation.DescreteTesselation2D import Tesselation2D

from dragen.generation.Mesher2D import Mesher_2D, BuildAbaqus2D
from dragen.utilities.InputInfo import RveInfo
from dragen.postprocessing.voldistribution import PostProcVol
from dragen.postprocessing.Shape_analysis import shape


# TODO insert new rve utils like array gen and grid gen etc.

class DataTask2D(HelperFunctions):

    def __init__(self):
        super().__init__()

    def grain_sampling(self):
        """
        In this function the correct number of grains for the chosen Volume is sampled from given csv files
        """
        files = RveInfo.file_dict
        RveInfo.logger.info("RVE generation process has started...")
        total_df = pd.DataFrame()

        # TODO: Generiere Bandwidths hier!
        if RveInfo.number_of_bands > 0:
            low = RveInfo.lower_band_bound
            high = RveInfo.upper_band_bound
            RveInfo.bandwidths = np.random.uniform(low=low, high=high, size=RveInfo.number_of_bands)
            print(RveInfo.bandwidths)
            sum_bw = RveInfo.bandwidths.sum()
        else:
            sum_bw = 0
        experimental_data = pd.DataFrame()
        for phase in RveInfo.phases:
            file_idx = RveInfo.PHASENUM[phase]

            print('current phase is', phase, ';phase input file is', files[file_idx])
            print('current phase is', phase, ';phase ratio file is', RveInfo.phase_ratio[file_idx])

            # Check file ending:
            if files[file_idx].endswith('.csv'):
                phase_input_df = super().read_input(files[file_idx], RveInfo.dimension)
            elif files[file_idx].endswith('.pkl'):
                phase_input_df = super().read_input_gan(files[file_idx], RveInfo.dimension, size=1000)

            if phase != 'Bands':

                adjusted_size = np.sqrt((RveInfo.box_size ** 2 -
                                         (RveInfo.box_size * sum_bw))
                                        * RveInfo.phase_ratio[file_idx])
                grains_df = super().sample_input_2D(phase_input_df, bs=adjusted_size)

                grains_df['phaseID'] = RveInfo.PHASENUM[phase]
                total_df = pd.concat([total_df, grains_df])
                phase_input_df['phaseID'] = RveInfo.PHASENUM[phase]
                experimental_data = pd.concat([experimental_data, phase_input_df])

            else:
                grains_df = phase_input_df.copy()
                grains_df['phaseID'] = RveInfo.PHASENUM[phase]
                total_df = pd.concat([total_df, grains_df])

        print('Processing now')


        grains_df = super().process_df_2D(total_df, RveInfo.shrink_factor)
        total_volume = sum(
            grains_df[grains_df['phaseID'] <= 6]['final_conti_volume'].values)  # Inclusions and bands dont influence filling
        estimated_boxsize = np.cbrt(total_volume)
        RveInfo.logger.info("the total volume of your dataframe is {}. A boxsize of {} is recommended.".
                            format(total_volume, estimated_boxsize))

        experimental_data.to_csv(RveInfo.gen_path + '/experimental_data.csv', index=False)

        print(grains_df)
        return grains_df

    def rve_generation(self, grains_df):

        discrete_RSA_obj = DiscreteRsa2D(grains_df['a'].tolist(), grains_df['b'].tolist(), grains_df['alpha'].tolist())

        rsa, x_0_list, y_0_list, rsa_status = discrete_RSA_obj.run_rsa()

        if rsa_status:
            grains_df['x_0'] = x_0_list
            grains_df['y_0'] = y_0_list
            discrete_tesselation_obj = Tesselation2D(grains_df)
            rve, rve_status = discrete_tesselation_obj.run_tesselation(rsa)

        else:
            RveInfo.logger.info("The rsa did not succeed...")
            sys.exit()

        if rve_status:

            periodic_rve_df = super().repair_periodicity_2D(rve)
            periodic_rve_df['phaseID'] = 0
            # An den NaN-Werten in dem DF liegt es nicht!

            grains_df.sort_values(by=['GrainID'])
            # debug_df = grains_df.copy()
            for i in range(len(grains_df)):
                # Set grain-ID to number of the grain
                # Denn Grain-ID ist entweder >0 oder -200 oder >-200
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == i + 1, 'phaseID'] = grains_df['phaseID'][i]
                # debug_df.loc[debug_df.index == i, 'vol_rve_df'] = \
                #    len(periodic_rve_df.loc[periodic_rve_df['GrainID'] == i + 1])*self.bin_size**3

            if RveInfo.number_of_bands > 0:
                # Set the points where == -200 to phase 2 and to grain ID i + 2
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == -200, 'GrainID'] = (i + 2)
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == (i + 2), 'phaseID'] = 2


            # Write out Volumes
            grains_df = super().get_final_disc_vol_2D(grains_df, rve)
            grains_df.to_csv(RveInfo.store_path + '/Generation_Data/grain_data_output.csv', index=False)

            mesher_obj = Mesher_2D(periodic_rve_df, grains_df, store_path=RveInfo.store_path)
            mesh = mesher_obj.run_mesher_2D()
            BuildAbaqus2D(mesh, periodic_rve_df, grains_df).run()

        RveInfo.logger.info("2D RVE generation process has successfully completed...")
        return rve

    def post_processing(self, rve):
        slice_ID = 0
        # the rve array still contains the boundarys in order to get every 4th slice we need to devide by 8
        phase_ratios = list()
        ref_r_in = dict()
        ref_r_out = dict()
        grain_shapes_in = shape().get_input_ellipses()
        for phase in RveInfo.phases:
            id = RveInfo.PHASENUM[phase]
            print(id)
            # generate pair plots for shape comparison for each phase
            grain_shapes = pd.DataFrame()
            grain_shapes_slice = shape().get_ellipses(rve, slice_ID, id)
            grain_shapes = pd.concat([grain_shapes, grain_shapes_slice])
            grain_shapes['inout'] = 'out'
            grain_shapes_in_thisPhase = grain_shapes_in.loc[grain_shapes_in['phaseID'] == id, ['AR', 'slope', 'inout']]
            if id == 2:
                print(grain_shapes_in_thisPhase)
            grain_shapes = pd.concat([grain_shapes, grain_shapes_in_thisPhase])
            grain_shapes = grain_shapes.sort_values(by=['inout'])
            grain_shapes.reset_index(inplace=True, drop=True)

            plot_kws = {"s": 2}
            sns.pairplot(data=grain_shapes, hue='inout', plot_kws=plot_kws)
            grain_shapes.to_csv('{}/Postprocessing/shape_control_{}.csv'.format(RveInfo.store_path, phase))
            plt.subplots_adjust(top=.95)
            plt.suptitle(phase)
            plt.savefig('{}/Postprocessing/shape_control_{}.png'.format(RveInfo.store_path, phase))


            current_phase_ref_r_in, current_phase_ratio_out, current_phase_ref_r_out = \
                PostProcVol().gen_in_out_lists(phaseID=id)
            phase_ratios.append(current_phase_ratio_out)
            ref_r_in[phase] = current_phase_ref_r_in
            ref_r_out[phase] = current_phase_ref_r_out

        if len(RveInfo.phases) > 1:
            input_ratio = [RveInfo.phase_ratio[key] for key in RveInfo.phase_ratio.keys()]
            labels = [label for label in RveInfo.phases]
            PostProcVol().gen_pie_chart_phases(input_ratio, labels, 'input')
            PostProcVol().gen_pie_chart_phases(phase_ratios, labels, 'output')

        for phase in RveInfo.phases:
            PostProcVol().gen_plots(ref_r_in[phase], ref_r_out[phase], phase)
            if RveInfo.gui_flag:
                RveInfo.infobox_obj.emit('checkout the evaluation report of the rve stored at:\n'
                                         '{}/Postprocessing'.format(RveInfo.store_path))

        if RveInfo.subs_flag:
            RveInfo.sub_run.post_processing(k=3)
        RveInfo.logger.info("RVE generation process has successfully completed...")