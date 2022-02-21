import os
import sys
import datetime
import numpy as np
import csv
import logging
import logging.handlers
import pandas as pd

from dragen.generation.DiscreteRsa2D import DiscreteRsa2D
from dragen.generation.DescreteTesselation2D import Tesselation2D
from dragen.utilities.Helpers import HelperFunctions
from dragen.generation.Mesher2D import Mesher_2D, BuildAbaqus2D
from dragen.utilities.InputInfo import RveInfo


# TODO insert new rve utils like array gen and grid gen etc.

class DataTask2D(HelperFunctions):

    def __init__(self):

        super().__init__()
        self.x_grid, self.y_grid, = super().gen_grid2d()
        RveInfo.logger.info('the exe_flag is: ' + str(RveInfo.exe_flag))
        RveInfo.logger.info('root was set to: ' + RveInfo.root)

    def initializations(self, dimension, epoch):

        phase1_csv = RveInfo.file1
        phase2_csv = RveInfo.file2

        RveInfo.logger.info("RVE generation process has started...")
        phase1_input_df = super().read_input(phase1_csv, dimension)
        # Phase Ratio calculation

        adjusted_size = np.sqrt((RveInfo.box_size ** 2 -
                                 (RveInfo.box_size * RveInfo.number_of_bands * RveInfo.band_width))
                                * RveInfo.phase_ratio)
        grains_df = super().sample_input_2D(phase1_input_df, bs=adjusted_size)
        grains_df['phaseID'] = 1

        if phase2_csv is not None:
            phase2_input_df = super().read_input(phase2_csv, dimension)
            adjusted_size = np.sqrt((RveInfo.box_size ** 2 -
                                     (RveInfo.box_size * RveInfo.number_of_bands * RveInfo.band_width))
                                    * RveInfo.phase_ratio)
            phase2_df = super().sample_input_2D(phase2_input_df, bs=adjusted_size)
            phase2_df['phaseID'] = 2
            grains_df = pd.concat([grains_df, phase2_df])

        grains_df = super().process_df_2D(grains_df, RveInfo.shrink_factor)
        total_volume = sum(grains_df['final_conti_volume'].values)
        estimated_boxsize = np.sqrt(total_volume)
        RveInfo.logger.info("the total volume of your dataframe is {}. A"
                         " boxsize of {} is recommended.".format(total_volume, estimated_boxsize))

        grains_df.to_csv(RveInfo.gen_path + '/grain_data_input.csv', index=False)
        return grains_df, RveInfo.store_path

    def rve_generation(self, grains_df):

        discrete_RSA_obj = DiscreteRsa2D(grains_df['a'].tolist(), grains_df['b'].tolist(), grains_df['alpha'].tolist())

        with open(RveInfo.store_path + '/discrete_input_vol.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(grains_df)
        rsa, x_0_list, y_0_list, rsa_status = discrete_RSA_obj.run_rsa()

        if rsa_status:
            discrete_tesselation_obj = Tesselation2D(grains_df['a'].tolist(), grains_df['b'].tolist(),
                                                     grains_df['alpha'].tolist(), x_0_list, y_0_list)
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

            # Start the Mesher
            # debug_df.to_csv('debug_grains_df.csv', index=False)
            # periodic_rve_df.to_csv('./test_rve.csv') lines are kept for debugging purposes
            # grains_df.to_csv('./grains_df.csv')
            mesher_obj = Mesher_2D(periodic_rve_df, grains_df, store_path=RveInfo.store_path)
            mesh = mesher_obj.run_mesher_2D()
            BuildAbaqus2D(mesh, periodic_rve_df, grains_df).run()

        RveInfo.logger.info("2D RVE generation process has successfully completed...")
