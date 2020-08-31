import os
import sys
import datetime
import numpy as np
import csv
import logging
import logging.handlers

from tqdm import tqdm

from dragen.generation.discrete_RSA import DiscreteRSA
from dragen.generation.discrete_tesselation import DiscreteTesselation
from dragen.utilities.RVE_Utils import RVEUtils


class DataTask:

    def __init__(self, box_size=22, points_on_edge=22, number_of_bands=0, bandwidth=3, speed=1, shrink_factor=0.3):
        self.logger = logging.getLogger("RVE-Gen")
        self.box_size = box_size
        self.points_on_edge = points_on_edge  # has to be even
        self.step_size = self.box_size / self.points_on_edge
        self.step_half = self.step_size / 2
        self.double_box = self.box_size * 2
        self.box_half = self.box_size / 2
        self.number_of_bands = number_of_bands
        self.bandwidth = bandwidth
        self.shrink_factor = np.cbrt(shrink_factor)
        self.tolerance = 0.01
        self.speed = speed
        main_dir = sys.argv[0][:-7]
        os.chdir(main_dir)
        self.utils_obj = RVEUtils(self.box_size, self.points_on_edge, self.bandwidth)
        self.discrete_RSA_obj = DiscreteRSA(self.box_size, self.points_on_edge, self.tolerance, self.number_of_bands,
                                            self.bandwidth)
        self.discrete_tesselation_obj = DiscreteTesselation(main_dir, self.box_size, self.points_on_edge,
                                                            self.bandwidth)

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

    def initializations(self):
        self.setup_logging()
        phase1csv = 'Inputdata/Bainite-1300.csv'
        phase2csv = '../Inputdata/38Mn-Pearlite.csv'
        testcase1 = '../Inputdata/Input2.csv'
        testcase2 = '../Inputdata/Input3.csv'
        testcase3 = '../Inputdata/Input4.csv'
        testcase4 = 'Inputdata/martensite.csv'

        self.logger.info("RVE generation process has started...")
        phase1_a, phase1_b, phase1_c, volume_phase1 = self.utils_obj.read_input(phase1csv)
        phase2_a, phase2_b, phase2_c, volume_phase2 = self.utils_obj.read_input(testcase4)
        convert_list = []
        for i in tqdm(range(len(phase1_a))):
            vol = self.utils_obj.convert_volume(phase1_a[i], phase1_b[i], phase1_c[i])
            convert_list.append(vol)
        for i in tqdm(range(len(phase2_a))):
            vol = self.utils_obj.convert_volume(phase2_a[i], phase2_b[i], phase2_c[i])
            convert_list.append(vol)

        phase1_a = np.array(phase1_a)
        phase1_b = np.array(phase1_b)
        phase1_c = np.array(phase1_c)
        phase2_a = np.array(phase2_a)
        phase2_b = np.array(phase2_b)
        phase2_c = np.array(phase2_c)

        if len(phase1_b) > 0:
            phase1 = list(
                zip(self.shrink_factor * phase1_a, self.shrink_factor * phase1_b, self.shrink_factor * phase1_c))
        else:
            phase1 = list(
                zip(self.shrink_factor * phase1_a, self.shrink_factor * phase1_a, self.shrink_factor * phase1_c))
        if len(phase2_b) > 0:
            phase2 = list(
                zip(self.shrink_factor * phase2_a, self.shrink_factor * phase2_b, self.shrink_factor * phase2_c))
        else:
            phase2 = list(
                zip(self.shrink_factor * phase2_a, self.shrink_factor * phase2_a, self.shrink_factor * phase2_c))

        return convert_list, phase1, phase2

    def rve_generation(self, epoch, convert_list, phase1, phase2):
        store_path = 'OutputData/' + str(datetime.datetime.now())[:10] + '_' + str(epoch)
        if not os.path.isdir(store_path):
            os.makedirs(store_path)
        with open(store_path + '/discrete_input_vol.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(convert_list)
        pt, rad, phase, band, status = self.discrete_RSA_obj.RSA3D(store_path, phase1, phase2)

        if status:
            self.discrete_tesselation_obj.tesselation(store_path, pt, rad, phase, convert_list, band)
        del pt, rad, phase
        self.logger.info("RVE generation process has successfully completed...")
