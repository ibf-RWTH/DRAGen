import os
import sys
import datetime
import numpy as np
import csv
import logging
import logging.handlers
import pandas as pd

from tqdm import tqdm
from dragen.generation.DiscreteRsa2D import DiscreteRsa2D
from dragen.generation.DescreteTesselation2D import Tesselation2D
from dragen.utilities.RVE_Utils import RVEUtils
from dragen.generation import mesher


class DataTask2D:

    def __init__(self, box_size=50, n_pts=100, number_of_bands=0, bandwidth=3, shrink_factor=0.5,
                 file1=None, file2=None, gui_flag=False, anim_flag=False):
        self.logger = logging.getLogger("RVE-Gen")
        self.box_size = box_size
        self.n_pts = n_pts  # has to be even
        self.bin_size = self.box_size / self.n_pts
        self.step_half = self.bin_size / 2
        self.number_of_bands = number_of_bands
        self.bandwidth = bandwidth
        self.shrink_factor = np.sqrt(shrink_factor)
        self.gui_flag = gui_flag
        self.anim_flag = anim_flag

        if not gui_flag:
            main_dir = sys.argv[0][:-14]  # setting main_dir to root_dir by checking path of current file
        else:
            main_dir = sys.argv[0][:-31]  # setting main_dir to root_dir by checking path of current file
        os.chdir(main_dir)

        """if not anim_flag:
            self.animation = False
        else:
            self.animation = True"""
        self.animation = anim_flag

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
            phase2 = './ExampleInput/pearlite_22_grains.csv'
        else:
            phase1 = self.file1
            phase2 = self.file2

        self.logger.info("RVE generation process has started...")
        phase1_a, phase1_b, phase1_slope, phase1_phi1, phase1_PHI, phase1_phi2 = self.utils_obj.read_input(phase1, dimension)
        final_volume_phase1 = [(phase1_a[i] * phase1_b[i] * np.pi) for i in range(len(phase1_a))]
        phase1_a_shrinked = [phase1_a_i * self.shrink_factor for phase1_a_i in phase1_a]
        phase1_b_shrinked = [phase1_b_i * self.shrink_factor for phase1_b_i in phase1_b]

        phase1_dict = {'a': phase1_a_shrinked,
                       'b': phase1_b_shrinked,
                       'slope': phase1_slope,
                       'phi1': phase1_phi1,
                       'PHI': phase1_PHI,
                       'phi2': phase1_phi2,
                       'final_volume': final_volume_phase1}
        grains_df = pd.DataFrame(phase1_dict)
        grains_df['phaseID'] = 1
        print(grains_df)

        if phase2 is not None:
            phase2_a, phase2_b, phase2_slope, phase2_phi1, phase2_PHI, phase2_phi2 = self.utils_obj.read_input(phase2, dimension)
            final_volume_phase2 = [(phase2_a[i] * phase2_b[i] * np.pi) for i in range(len(phase2_a))]
            phase2_a_shrinked = [phase2_a[i] * self.shrink_factor for i in range(len(phase2_a))]
            phase2_b_shrinked = [phase2_b[i] * self.shrink_factor for i in range(len(phase2_b))]

            phase2_dict = {'a': phase2_a_shrinked,
                           'b': phase2_b_shrinked,
                           'slope': phase2_slope,
                           'phi1': phase2_phi1,
                           'PHI': phase2_PHI,
                           'phi2': phase2_phi2,
                           'final_volume': final_volume_phase2}

            grains_phase2_df = pd.DataFrame(phase2_dict)
            grains_phase2_df['phaseID'] = 2
            grains_df = pd.concat([grains_df, grains_phase2_df])

        grains_df.sort_values(by='final_volume', inplace=True, ascending=False)
        grains_df.reset_index(inplace=True, drop=True)
        grains_df['grainID'] = grains_df.index
        total_volume = sum(grains_df['final_volume'].values)
        estimated_boxsize = np.sqrt(total_volume)
        self.logger.info("the total volume of your dataframe is {}. A"
                         " boxsize of {} is recommended.".format(total_volume, estimated_boxsize) )

        return grains_df

    def rve_generation(self, epoch, grains_df):
        store_path = 'OutputData/' + str(datetime.datetime.now())[:10] + '_' + str(epoch)
        if not os.path.isdir(store_path):
            os.makedirs(store_path)
            os.makedirs(store_path + '/Figs')
        discrete_RSA_obj = DiscreteRsa2D(self.box_size, self.n_pts, grains_df['a'].tolist(),
                                         grains_df['b'].tolist(), grains_df['slope'].tolist(), store_path= store_path)

        with open(store_path + '/discrete_input_vol.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(grains_df)
        rsa, x_0_list, y_0_list, rsa_status = discrete_RSA_obj.run_rsa(animation=True)

        if rsa_status:
            discrete_tesselation_obj = Tesselation2D(self.box_size, self.n_pts, grains_df['a'].tolist(),
                                                     grains_df['b'].tolist(), grains_df['slope'].tolist(),
                                                     x_0_list, y_0_list, self.shrink_factor, store_path)
            rve, rve_status = discrete_tesselation_obj.run_tesselation(rsa)

        else:
            self.logger.info("The rsa did not succeed...")
            sys.exit()

        if rve_status:
            np.save(store_path + '/rve_array_2D', rve)

            #RVE = pd.DataFrame(pd.read_hdf(store_path + '/boxrve.h5'))
            #mesher.Mesher(rve).mesh_and_build_abaqus_model(store_path)
        self.logger.info("RVE generation process has successfully completed...")
