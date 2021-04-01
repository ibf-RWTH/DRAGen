import os
import sys
import datetime
import numpy as np
import csv
import logging
import logging.handlers
import pandas as pd

from tqdm import tqdm
from dragen.generation.DiscreteRsa3D import DiscreteRsa3D
from dragen.generation.DiscreteTesselation3D import Tesselation3D
from dragen.utilities.RVE_Utils import RVEUtils
from dragen.generation.mesher import Mesher


class DataTask3D:

    def __init__(self, box_size=22, n_pts=30, number_of_bands=0, bandwidth=3, shrink_factor=0.5, file1=None, file2=None, gui_flag=False):
        self.logger = logging.getLogger("RVE-Gen")
        self.box_size = box_size
        self.n_pts = n_pts  # has to be even
        self.bin_size = self.box_size / self.n_pts
        self.step_half = self.bin_size / 2
        self.number_of_bands = number_of_bands
        self.bandwidth = bandwidth
        self.shrink_factor = np.cbrt(shrink_factor)
        self.gui_flag = gui_flag
        if not gui_flag:
            main_dir = sys.argv[0][:-14]
            os.chdir(main_dir)
            self.animation = True
        else:
            main_dir = sys.argv[0][:-31]
            os.chdir(main_dir)
            self.animation = True
        self.file1 = file1
        self.file2 = file2
        self.utils_obj = RVEUtils(self.box_size, self.n_pts, self.bandwidth)

    def setup_logging(self):
        LOGS_DIR = 'Logs/'
        if not os.path.isdir(LOGS_DIR):
            os.makedirs(LOGS_DIR)
        f_handler = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(LOGS_DIR, 'dragen-logs'), when='midnight')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        f_handler.setFormatter(formatter)
        self.logger.addHandler(f_handler)
        self.logger.setLevel(level=logging.DEBUG)

    def initializations(self, dimension):
        self.setup_logging()
        if not self.gui_flag:
            phase1 = './ExampleInput/ferrite_54_grains.csv'
            phase2 = './ExampleInput/pearlite_21_grains.csv'
        else:
            phase1 = self.file1
            phase2 = self.file2

        self.logger.info("RVE generation process has started...")
        phase1_a, phase1_b, phase1_c, phase1_slope = self.utils_obj.read_input(phase1, dimension)
        final_volume_phase1 = [(4/3*phase1_a[i] * phase1_b[i] * phase1_c[i] * np.pi) for i in range(len(phase1_a))]
        phase1_a_shrinked = [phase1_a_i * self.shrink_factor for phase1_a_i in phase1_a]
        phase1_b_shrinked = [phase1_b_i * self.shrink_factor for phase1_b_i in phase1_b]
        phase1_c_shrinked = [phase1_c_i * self.shrink_factor for phase1_c_i in phase1_c]

        phase1_dict = {'a': phase1_a_shrinked,
                       'b': phase1_b_shrinked,
                       'c': phase1_c_shrinked,
                       'slope': phase1_slope,
                       'final_volume': final_volume_phase1}
        grains_df = pd.DataFrame(phase1_dict)
        grains_df['phaseID'] = 1

        if phase2 is not None:
            phase2_a, phase2_b, phase2_c, phase2_slope = self.utils_obj.read_input(phase2, dimension)
            final_volume_phase2 = [(4/3*phase2_a[i] * phase2_b[i] * phase2_c[i] * np.pi) for i in range(len(phase2_a))]
            phase2_a_shrinked = [phase2_a_i * self.shrink_factor for phase2_a_i in phase2_a]
            phase2_b_shrinked = [phase2_b_i * self.shrink_factor for phase2_b_i in phase2_b]
            phase2_c_shrinked = [phase2_c_i * self.shrink_factor for phase2_c_i in phase2_c]
            phase2_dict = {'a': phase2_a_shrinked,
                           'b': phase2_b_shrinked,
                           'c': phase2_c_shrinked,
                           'slope': phase2_slope,
                           'final_volume': final_volume_phase2}

            grains_phase2_df = pd.DataFrame(phase2_dict)
            grains_phase2_df['phaseID'] = 2
            grains_df = pd.concat([grains_df, grains_phase2_df])

        grains_df.sort_values(by='final_volume', inplace=True, ascending=False)
        grains_df.reset_index(inplace=True, drop=True)
        grains_df['GrainID'] = grains_df.index
        total_volume = sum(grains_df['final_volume'].values)
        estimated_boxsize = np.cbrt(total_volume)
        self.logger.info("the total volume of your dataframe is {}. A boxsize of {} is recommended.".format(total_volume, estimated_boxsize) )

        return grains_df

    def rve_generation(self, epoch, grains_df):
        store_path = 'OutputData/' + str(datetime.datetime.now())[:10] + '_' + str(epoch)
        if not os.path.isdir(store_path):
            os.makedirs(store_path)
            os.makedirs(store_path + '/Figs')
        discrete_RSA_obj = DiscreteRsa3D(self.box_size, self.n_pts,
                                         grains_df['a'].tolist(),
                                         grains_df['b'].tolist(),
                                         grains_df['c'].tolist(),
                                         grains_df['slope'].tolist(), store_path=store_path)

        with open(store_path + '/discrete_input_vol.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(grains_df)

        if self.number_of_bands > 0:
            # initialize empty grid_array for bands called band_array
            xyz = np.linspace(-self.box_size / 2, self.box_size + self.box_size / 2, 2 * self.n_pts, endpoint=True)
            x_grid, y_grid, z_grid = np.meshgrid(xyz, xyz, xyz)
            utils_obj_band = RVEUtils(self.box_size, self.n_pts,
                                x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, bandwidth=self.bandwidth)
            band_array = np.zeros((2 * self.n_pts, 2 * self.n_pts, 2 * self.n_pts))
            band_array = utils_obj_band.gen_boundaries_3D(band_array)

            for i in range(self.number_of_bands):
                band_array = utils_obj_band.band_generator(band_array)

            rsa, x_0_list, y_0_list, z_0_list, rsa_status = discrete_RSA_obj.run_rsa(band_array, animation=self.animation)

        else:
            rsa, x_0_list, y_0_list, z_0_list, rsa_status = discrete_RSA_obj.run_rsa(animation=self.animation)

        if rsa_status:
            discrete_tesselation_obj = Tesselation3D(self.box_size, self.n_pts,
                                                     grains_df['a'].tolist(),
                                                     grains_df['b'].tolist(),
                                                     grains_df['c'].tolist(),
                                                     grains_df['slope'].tolist(),
                                                     x_0_list, y_0_list, z_0_list,
                                                     self.shrink_factor, store_path)
            rve, rve_status = discrete_tesselation_obj.run_tesselation(rsa, animation=self.animation)

        else:
            self.logger.info("The rsa did not succeed...")
            sys.exit()

        if rve_status:
            periodic_rve_df = self.utils_obj.repair_periodicity_3D(rve)
            periodic_rve_df['phaseID'] = 0

            grains_df.sort_values(by=['GrainID'])

            for i in range(len(grains_df)):
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == i + 1, 'phaseID'] = grains_df['phaseID'][i]
            if self.number_of_bands > 0:
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == -200, 'GrainID'] = i + 2
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == i + 2, 'phaseID'] = 2

            mesher_obj = Mesher(periodic_rve_df, store_path=store_path, phase_two_isotropic=True, animation=False)
            mesher_obj.mesh_and_build_abaqus_model()

        self.logger.info("RVE generation process has successfully completed...")

