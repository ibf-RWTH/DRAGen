import datetime
import os
import sys
import logging
import math
import numpy as np
from typing import Dict

from dragen.main2D import DataTask2D
from dragen.main3D import DataTask3D
from dragen.main3D_GAN import DataTask3D_GAN
from dragen.substructure.run import Run as SubRun
from dragen.utilities.Helpers import HelperFunctions
from dragen.utilities.InputInfo import RveInfo


class Run(HelperFunctions):
    def __init__(
                 # mandatory arguments:
                 self, box_size: int, element_type: str, resolution: float, number_of_rves: int,
                 number_of_bands: int,  dimension: int, visualization_flag: bool,
                 store_path: str, shrink_factor: float,  phase_ratio: float, gui_flag: bool, gan_flag: bool,
                 subs_flag: bool, phases: list, abaqus_flag: bool, damask_flag: bool, moose_flag: bool,
                 anim_flag: bool, exe_flag: bool, box_size_y: int, file_dict: dict(), inclusion_flag: bool,
                 inclusion_ratio: float, band_filling: float, upper_band_bound: float, lower_band_bound: float,
                 # optional Arguments or dependent on previous flag
                 subs_file_flag=False, subs_file: str = None,
                 box_size_z: int = None, bandwidth: float = None,
                 info_box_obj=None, progress_obj=None, equiv_d: float = None, p_sigma: float = None, t_mu: float = None,
                 b_sigma: float = 0.01, decreasing_factor: float = 0.95, lower: float = None, upper: float = None,
                 circularity: float = 1, plt_name: str = None, save: bool = True, plot: bool = False,
                 filename: str = None, fig_path: str = None, orient_relationship: str = None
    ):

        super().__init__()
        # TODO: @Manuel @Max. Ich habe schon häufiger gesehen, dass in anderen Packages Konstanten immer GROß
        #  geschrieben werden. Ist das auch für uns ne Idee, dann weiß man im CODE auch immer, welche variable man
        #  auf keinen Fall einfach so ändern sollte
        RveInfo.box_size = box_size
        RveInfo.element_type = element_type
        RveInfo.box_size_y = box_size_y
        RveInfo.box_size_z = box_size_z
        RveInfo.resolution = resolution
        RveInfo.number_of_rves = number_of_rves
        RveInfo.number_of_bands = number_of_bands
        RveInfo.lower_band_bound = lower_band_bound
        RveInfo.upper_band_bound = upper_band_bound
        RveInfo.dimension = dimension
        RveInfo.visualization_flag = visualization_flag
        RveInfo.file_dict = file_dict   # TODO: Change to dict based output
        RveInfo.phase_ratio = phase_ratio
        RveInfo.store_path = store_path
        RveInfo.shrink_factor = np.cbrt(shrink_factor)
        RveInfo.gui_flag = gui_flag
        RveInfo.gan_flag = gan_flag
        RveInfo.infobox_obj = info_box_obj
        RveInfo.progress_obj = progress_obj
        RveInfo.equiv_d = equiv_d
        RveInfo.p_sigma = p_sigma
        RveInfo.t_mu = t_mu
        RveInfo.b_sigma = b_sigma
        RveInfo.decreasing_factor = decreasing_factor
        RveInfo.lower = lower
        RveInfo.upper = upper
        RveInfo.circularity = circularity
        RveInfo.plt_name = plt_name
        RveInfo.save = save
        RveInfo.plot = plot
        RveInfo.filename = filename
        RveInfo.fig_path = fig_path
        RveInfo.orientation_relationship = orient_relationship
        RveInfo.subs_flag = subs_flag
        RveInfo.subs_file_flag = subs_file_flag
        RveInfo.subs_file = subs_file
        RveInfo.phases = phases
        RveInfo.abaqus_flag = abaqus_flag
        RveInfo.damask_flag = damask_flag
        RveInfo.moose_flag = moose_flag
        RveInfo.anim_flag = anim_flag
        RveInfo.exe_flag = exe_flag
        RveInfo.RVphase2iso_flag = True
        RveInfo.element_type = 'HEX8'
        RveInfo.roughness_flag = False
        RveInfo.band_filling = band_filling
        RveInfo.inclusion_ratio = inclusion_ratio
        RveInfo.inclusion_flag = inclusion_flag
        RveInfo.root = './'
        RveInfo.input_path = './ExampleInput'

        RveInfo.n_pts = math.ceil(float(box_size) * resolution)
        if RveInfo.n_pts % 2 != 0:
            RveInfo.n_pts += 1

        if box_size_y:
            RveInfo.n_pts_y = math.ceil(float(box_size_y) * resolution)
            if RveInfo.n_pts_y % 2 != 0:
                RveInfo.n_pts_y += 1

        if box_size_z:
            RveInfo.n_pts_z = math.ceil(float(box_size_z) * resolution)
            if RveInfo.n_pts_z % 2 != 0:
                RveInfo.n_pts_z += 1
        RveInfo.bin_size = RveInfo.box_size / RveInfo.n_pts
        RveInfo.step_half = RveInfo.bin_size / 2


    @staticmethod
    def setup_logging():
        LOGS_DIR = RveInfo.store_path + '/Logs/'
        if not os.path.isdir(LOGS_DIR):
            os.makedirs(LOGS_DIR)
        f_handler = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(LOGS_DIR, 'dragen-logs'), when='midnight')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        f_handler.setFormatter(formatter)
        RveInfo.logger.addHandler(f_handler)
        RveInfo.logger.setLevel(level=logging.DEBUG)

    @staticmethod
    def initializations(epoch):

        RveInfo.store_path = RveInfo.store_path + '/OutputData/' + str(datetime.datetime.now())[:10] + '_' + str(epoch)
        RveInfo.fig_path = RveInfo.store_path + '/Figs'
        RveInfo.gen_path = RveInfo.store_path + '/Generation_Data'

        if not os.path.isdir(RveInfo.store_path):
            os.makedirs(RveInfo.store_path)

        if not os.path.isdir(RveInfo.fig_path):
            os.makedirs(RveInfo.fig_path)

        if not os.path.isdir(RveInfo.gen_path):
            os.makedirs(RveInfo.gen_path)

        f_handler = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(RveInfo.store_path, 'result-logs'), when='midnight')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        f_handler.setFormatter(formatter)
        RveInfo.result_log.addHandler(f_handler)
        RveInfo.result_log.setLevel(level=logging.DEBUG)

    def run(self):
        self.setup_logging()
        if RveInfo.infobox_obj is not None:
            RveInfo.infobox_obj.emit("the chosen resolution lead to {}^{} "
                                     "points in the grid".format(str(RveInfo.n_pts), str(RveInfo.dimension)))

        if RveInfo.subs_flag:
            RveInfo.sub_run = SubRun()

        if RveInfo.dimension == 2:

            obj2D = DataTask2D()

            for i in range(RveInfo.number_of_rves):
                self.initializations(i)
                obj2D.initializations(RveInfo.dimension, epoch=i)
                # TODO: 2D fixen
                #obj2D.rve_generation()

        elif RveInfo.dimension == 3:
            # Kann Gan und nicht GAN
            obj3D = DataTask3D()
            for i in range(RveInfo.number_of_rves):
                self.initializations(i)
                total_df = obj3D.grain_sampling()
                obj3D.rve_generation(total_df)
                if RveInfo.subs_file_flag:
                    obj3D.post_processing()


        else:
            LOGS_DIR = 'Logs/'
            logger = logging.getLogger("RVE-Gen")
            if not os.path.isdir(LOGS_DIR):
                os.makedirs(LOGS_DIR)
            f_handler = logging.handlers.TimedRotatingFileHandler(
                filename=os.path.join(LOGS_DIR, 'dragen-logs'), when='midnight')
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            f_handler.setFormatter(formatter)
            logger.addHandler(f_handler)
            logger.setLevel(level=logging.DEBUG)
            logger.info('dimension must be 2 or 3')
            sys.exit()


if __name__ == "__main__":
    box_size = 25
    box_size_y = None  # if this is None it will be set to the main box_size value
    box_size_z = None  # for sheet rve set z to None and y to different value than x the other way round is buggy
    resolution = 1.5
    number_of_rves = 1
    number_of_bands = 2
    band_filling = 1.2
    lower_band_bound = 2
    upper_band_bound = 5
    visualization_flag = True
    phase_ratio = 1
    store_path = '../'
    shrink_factor = 0.4
    dimension = 3
    # Example Files
    equiv_d = 5
    p_sigma = 0.1
    t_mu = 1.0
    b_sigma = 0.1
    inclusion_flag = False
    inclusion_ratio = 0.01
    # Example Files
    # file1 = r'C:\Venvs\dragen\ExampleInput\ferrite_54_grains_processed.csv'
    file1 = r'C:\Venvs\dragen\ExampleInput\TrainedData_2.pkl'
    file6 = r'C:\Venvs\dragen\ExampleInput\TrainedData_2.pkl'
    file2 = r'C:\Venvs\dragen\ExampleInput\pearlite_21_grains.csv'
    file3 = r'C:\Venvs\dragen\ExampleInput\38Mn-Ferrite.csv'
    # test pearlite phase
    subs_flag = False
    subs_file = '../ExampleInput/example_block_inp.csv'
    subs_file_flag = False
    gui_flag = False
    gan_flag = False
    moose_flag = True
    abaqus_flag = False
    damask_flag = True
    element_type = 'HEX8'
    anim_flag = False
    exe_flag = False
    files = {1: file1, 2: file2, 3: None, 4: None, 5: file3, 6: file6}

    '''
    specific number is fixed for each phase. 1->ferrite, 2->martensite so far. The order of input files should also have the 
    same order as phases. file1->ferrite, file2->martensite. The substructures will only be generated in martensite.
    
    Number 5 specifies the inclusions and number 6 the Band phase. Either .csv or .pckl
    '''
    phases = ['ferrite', 'Bands']
    Run(box_size, element_type=element_type, box_size_y=box_size_y, box_size_z=box_size_z, resolution=resolution,
        number_of_rves=number_of_rves,
        number_of_bands=number_of_bands, dimension=dimension,
        visualization_flag=visualization_flag, file_dict=files, equiv_d=equiv_d, p_sigma=p_sigma, t_mu=t_mu,
        b_sigma=b_sigma,
        phase_ratio=phase_ratio, store_path=store_path, shrink_factor=shrink_factor, gui_flag=gui_flag,
        gan_flag=gan_flag,
        info_box_obj=None, progress_obj=None, subs_file_flag=subs_file_flag, subs_file=subs_file, phases=phases,
        subs_flag=subs_flag, moose_flag=moose_flag, abaqus_flag=abaqus_flag, damask_flag=damask_flag,
        anim_flag=anim_flag, exe_flag=exe_flag, inclusion_flag=inclusion_flag,
        inclusion_ratio=inclusion_ratio, band_filling=band_filling, lower_band_bound=lower_band_bound,
        upper_band_bound=upper_band_bound).run()
