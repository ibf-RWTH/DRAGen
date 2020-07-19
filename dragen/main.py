import os
import sys
import datetime
import numpy as np
import csv

import tqdm.auto

from dragen.generation.discrete_RSA import DiscreteRSA
from dragen.generation.discrete_tesselation import DiscreteTesselation
from dragen.utilities.RVE_Utils import RVEUtils


class DataTask:

    def __init__(self):
        # LOGGER initialization
        self.box_size = 22.
        self.points_on_edge = 88  # has to be even
        self.step_size = self.box_size / self.points_on_edge
        self.step_half = self.step_size / 2
        self.double_box = self.box_size * 2
        self.box_half = self.box_size / 2
        self.number_of_bands = 0
        self.bandwidth = 3
        self.shrink_factor = np.cbrt(0.3)
        self.tolerance = 0.01
        self.speed = 1
        self.first_RVE = 0
        self.last_RVE = 10

    def run_main(self):
        main_dir = sys.argv[0][:-7]
        os.chdir(main_dir)

        file_utils_obj = FileUtils()
        utils_obj = RVEUtils(self.box_size, self.points_on_edge, self.bandwidth)
        discrete_RSA_obj = DiscreteRSA(self.box_size, self.points_on_edge, self.tolerance, self.number_of_bands,
                                       self.bandwidth)
        discrete_tesselation_obj = DiscreteTesselation(main_dir, self.box_size, self.points_on_edge, self.bandwidth)

        phase1csv = '../Inputdata/Bainite-1300.csv'
        phase2csv = '../Inputdata/38Mn-Pearlite.csv'
        testcase1 = '../Inputdata/Input2.csv'
        testcase2 = '../Inputdata/Input3.csv'
        testcase3 = '../Inputdata/Input4.csv'
        testcase4 = '../Inputdata/martensite.csv'

        phase1_a, phase1_b, phase1_c, volume_phase1 = file_utils_obj.read_input(phase1csv)
        phase2_a, phase2_b, phase2_c, volume_phase2 = file_utils_obj.read_input(testcase4)
        convert_list = []
        for i in tqdm(range(len(phase1_a))):
            vol = utils_obj.convert_volume(phase1_a[i], phase1_b[i], phase1_c[i])
            convert_list.append(vol)
        for i in tqdm(range(len(phase2_a))):
            vol = utils_obj.convert_volume(phase2_a[i], phase2_b[i], phase2_c[i])
            convert_list.append(vol)

        phase1_a = np.array(phase1_a)
        phase1_b = np.array(phase1_b)
        phase1_c = np.array(phase1_c)
        phase2_a = np.array(phase2_a)
        phase2_b = np.array(phase2_b)
        phase2_c = np.array(phase2_c)

        if len(phase1_b) > 0:
            phase1 = list(zip(self.shrink_factor * phase1_a, self.shrink_factor * phase1_b, self.shrink_factor * phase1_c))
        else:
            phase1 = list(zip(self.shrink_factor * phase1_a, self.shrink_factor * phase1_a, self.shrink_factor * phase1_c))
        if len(phase2_b) > 0:
            phase2 = list(zip(self.shrink_factor * phase2_a, self.shrink_factor * phase2_b, self.shrink_factor * phase2_c))
        else:
            phase2 = list(zip(self.shrink_factor * phase2_a, self.shrink_factor * phase2_a, self.shrink_factor * phase2_c))

        for i in range(self.first_RVE, self.last_RVE + 1):
            store_path = '../OutputData/' + str(datetime.datetime.now())[:10] + '_' + str(i)
            os.system('mkdir ' + store_path)
            with open(store_path + '/discrete_input_vol.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(convert_list)
            pt, rad, phase, band, status = discrete_RSA_obj.RSA3D(store_path, phase1, phase2)
            failpt = [point for point in pt if len([i for i in point if str(i)[-1] != str(5)]) > 0]

            if status:
                discrete_tesselation_obj.tesselation(store_path, pt, rad, phase, convert_list, band)
            del pt, rad, phase
            sys.exit()
