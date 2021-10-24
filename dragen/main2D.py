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
from dragen.utilities.RVE_Utils import RVEUtils
from dragen.generation.Mesher2D import Mesher_2D, BuildAbaqus2D

# TODO insert new rve utils like array gen and grid gen etc.

class DataTask2D(RVEUtils):

    def __init__(self, box_size: int, n_pts: int, number_of_bands: int, bandwidth: float, shrink_factor: float = 0.5,
                 band_ratio_rsa: float = 0.95, band_ratio_final: float = 0.95, phase_ratio: float = None, file1=None, file2=None, store_path=None,
                 gui_flag=False, anim_flag=False, gan_flag=False, exe_flag=False,phases:list =['pearlite','ferrite']):

        self.logger = logging.getLogger("RVE-Gen")
        self.box_size = box_size
        self.n_pts = n_pts  # has to be even
        self.bin_size = self.box_size / self.n_pts
        self.step_half = self.bin_size / 2
        self.number_of_bands = number_of_bands
        self.bandwidth = bandwidth
        self.shrink_factor = np.sqrt(shrink_factor)
        self.band_ratio_rsa = band_ratio_rsa  # Band Ratio for RSA
        self.band_ratio_final = band_ratio_final  # Band ratio for Tesselator - final is br1 * br2
        self.phase_ratio = phase_ratio
        self.gui_flag = gui_flag
        self.exe_flag = exe_flag

        self.root_dir = './'

        if exe_flag:
            self.root_dir = store_path
            print('1',self.root_dir)
        if not gui_flag:
            self.root_dir = sys.argv[0][:-14]  # setting root_dir to root_dir by checking path of current file
            print('2',self.root_dir)
        elif gui_flag and not exe_flag:
            self.root_dir = store_path
            print('3',self.root_dir)

        self.logger.info('the exe_flag is: ' + str(exe_flag))
        self.logger.info('root was set to: ' + self.root_dir)
        self.animation = anim_flag
        self.file1 = file1
        self.file2 = file2

        self.x_grid, self.y_grid, = super().gen_grid2D()
        print(self.bandwidth)
        super().__init__(box_size, n_pts, self.x_grid, self.y_grid, bandwidth=self.bandwidth, debug=False)
        print(self.bandwidth)
    def setup_logging(self):
        LOGS_DIR = self.root_dir + '/Logs/'
        if not os.path.isdir(LOGS_DIR):
            os.makedirs(LOGS_DIR)
        f_handler = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(LOGS_DIR, 'dragen-logs'), when='midnight')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        f_handler.setFormatter(formatter)
        self.logger.addHandler(f_handler)
        self.logger.setLevel(level=logging.DEBUG)

    def initializations(self, dimension, epoch):

        self.setup_logging()

        '''        if not self.gui_flag:
            phase1_csv = self.root_dir + '/ExampleInput/ferrite_54_grains.csv'
            phase2_csv = self.root_dir + '/ExampleInput/pearlite_21_grains.csv'
        else:'''
        phase1_csv = self.file1
        phase2_csv = self.file2

        self.logger.info("RVE generation process has started...")
        phase1_input_df = super().read_input(phase1_csv, dimension)
        # Phase Ratio calculation
        print(self.box_size, self.number_of_bands, self.bandwidth)
        adjusted_size = np.sqrt((self.box_size ** 2 -
                                 (self.box_size * self.number_of_bands * self.bandwidth))
                                * self.phase_ratio)
        grains_df = super().sample_input_2D(phase1_input_df, bs=adjusted_size)
        grains_df['phaseID'] = 1

        if phase2_csv is not None:
            phase2_input_df = super().read_input(phase2_csv, dimension)
            adjusted_size = np.sqrt((self.box_size ** 2 -
                                 (self.box_size * self.number_of_bands * self.bandwidth))
                                * self.phase_ratio)
            phase2_df = super().sample_input_2D(phase2_input_df, bs=adjusted_size)
            phase2_df['phaseID'] = 2
            grains_df = pd.concat([grains_df, phase2_df])

        grains_df = super().process_df_2D(grains_df, self.shrink_factor)
        total_volume = sum(grains_df['final_conti_volume'].values)
        estimated_boxsize = np.sqrt(total_volume)
        self.logger.info("the total volume of your dataframe is {}. A"
                         " boxsize of {} is recommended.".format(total_volume, estimated_boxsize) )

        self.store_path = self.root_dir + '/OutputData/' + str(datetime.datetime.now())[:10] + '_' + str(epoch)
        self.fig_path = self.store_path + '/Figs'
        self.gen_path = self.store_path + '/Generation_Data'

        if not os.path.isdir(self.store_path):
            os.makedirs(self.store_path)
        if not os.path.isdir(self.fig_path):
            os.makedirs(self.fig_path)  # Second if needed
        if not os.path.isdir(self.gen_path):
            os.makedirs(self.gen_path)  # Second if needed

        grains_df.to_csv(self.gen_path + '/grain_data_input.csv', index=False)
        return grains_df, self.store_path

    def rve_generation(self, grains_df, store_path):

        discrete_RSA_obj = DiscreteRsa2D(self.box_size, self.n_pts,
                                         grains_df['a'].tolist(),
                                         grains_df['b'].tolist(),
                                         grains_df['alpha'].tolist(),
                                         store_path=store_path)

        with open(store_path + '/discrete_input_vol.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(grains_df)
        rsa, x_0_list, y_0_list, rsa_status = discrete_RSA_obj.run_rsa(animation=True)

        if rsa_status:
            discrete_tesselation_obj = Tesselation2D(self.box_size, self.n_pts, grains_df['a'].tolist(),
                                                     grains_df['b'].tolist(), grains_df['alpha'].tolist(),
                                                     x_0_list, y_0_list, self.shrink_factor, store_path)
            rve, rve_status = discrete_tesselation_obj.run_tesselation(rsa)

        else:
            self.logger.info("The rsa did not succeed...")
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

            if self.number_of_bands > 0:
                # Set the points where == -200 to phase 2 and to grain ID i + 2
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == -200, 'GrainID'] = (i + 2)
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == (i + 2), 'phaseID'] = 2

            # Start the Mesher
            # debug_df.to_csv('debug_grains_df.csv', index=False)
            # periodic_rve_df.to_csv('./test_rve.csv') lines are kept for debugging purposes
            # grains_df.to_csv('./grains_df.csv')
            mesher_obj = Mesher_2D(periodic_rve_df, grains_df, store_path=store_path)
            mesh = mesher_obj.run_mesher_2D()
            BuildAbaqus2D(mesh, periodic_rve_df, grains_df, store_path).run()


        return store_path
        # self.logger.info("RVE generation process has successfully completed...")
