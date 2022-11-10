import datetime
import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
import math
import matplotlib.pyplot as plt


from dragen.main2D import DataTask2D
from dragen.main3D import DataTask3D
from dragen.substructure.run import Run as SubRun
from dragen.utilities.Helpers import HelperFunctions
from dragen.utilities.InputInfo import RveInfo

plt.ioff()  # needs to be here for background plotting with gui


class Run(HelperFunctions):
    def __init__(
            # mandatory arguments:
            # box parameters:
            self,
            dimension: int,
            box_size: int,
            box_size_y: int,
            box_size_z: int,
            resolution: float,
            number_of_rves: int,


            # specimen orientation parameters
            slope_offset: float,

            # simulation framework parameters:
            abaqus_flag: bool,
            damask_flag: bool,
            moose_flag: bool,
            element_type: str,
            pbc_flag: bool,
            submodel_flag: bool,
            phase2iso_flag: bool,
            smoothing_flag: bool,
            xfem_flag: bool,

            # generation parameters
            gui_flag: bool,

            anim_flag: bool,
            visualization_flag: bool,
            root: str,
            info_box_obj, progress_obj,

            # microstructure parameters
            phase_ratio: dict,
            file_dict: dict,
            phases: list,

            # band related  parameters:
            number_of_bands: int,
            upper_band_bound: float, lower_band_bound: float,
            band_orientation: str, band_filling: float,

            # substructure related parameters:
            subs_flag: bool,
            subs_file_flag: bool,
            subs_file: str,
            equiv_d: float,
            p_sigma: float,
            t_mu: float,
            b_sigma: float,
            decreasing_factor: float,
            lower: float,
            upper: float,
            circularity: float,
            plt_name: str,
            save: bool,
            plot: bool,
            filename: str,
            orientation_relationship: str
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
        RveInfo.band_orientation = band_orientation
        RveInfo.dimension = dimension
        RveInfo.slope_offset = slope_offset
        RveInfo.smoothing_flag = smoothing_flag
        RveInfo.visualization_flag = visualization_flag
        RveInfo.file_dict = file_dict  # TODO: Change to dict based output
        RveInfo.phase_ratio = phase_ratio
        RveInfo.gui_flag = gui_flag
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
        RveInfo.orientation_relationship = orientation_relationship
        RveInfo.subs_flag = subs_flag
        RveInfo.subs_file_flag = subs_file_flag
        RveInfo.subs_file = subs_file
        RveInfo.phases = phases
        RveInfo.abaqus_flag = abaqus_flag
        RveInfo.damask_flag = damask_flag
        RveInfo.moose_flag = moose_flag
        RveInfo.anim_flag = anim_flag

        RveInfo.phase2iso_flag = phase2iso_flag
        RveInfo.pbc_flag = pbc_flag
        RveInfo.submodel_flag = submodel_flag
        RveInfo.xfem_flag = xfem_flag

        RveInfo.roughness_flag = False
        RveInfo.band_filling = band_filling
        RveInfo.root = root

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
        LOGS_DIR = RveInfo.root + '/Logs/'
        if not os.path.isdir(LOGS_DIR):
            os.makedirs(LOGS_DIR)
        f_handler = TimedRotatingFileHandler(filename=os.path.join(LOGS_DIR, 'dragen-logs'), when='midnight')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        f_handler.setFormatter(formatter)
        RveInfo.LOGGER.addHandler(f_handler)
        RveInfo.LOGGER.setLevel(level=logging.DEBUG)

    @staticmethod
    def initializations(epoch):
        counter = "0"
        if epoch < 10:
            counter = f"00{epoch}"
        if epoch > 9 and epoch < 100:
            counter = f"0{epoch}"
        if epoch > 99:
            counter = epoch
        RveInfo.store_path = RveInfo.root + '/OutputData/' + str(datetime.datetime.now())[:10] + '_' + counter
        RveInfo.LOGGER.debug(RveInfo.store_path)
        RveInfo.fig_path = RveInfo.store_path + '/Figs'
        RveInfo.gen_path = RveInfo.store_path + '/Generation_Data'
        RveInfo.post_path = RveInfo.store_path + '/Postprocessing'

        if not os.path.isdir(RveInfo.store_path):
            os.makedirs(RveInfo.store_path)

        if not os.path.isdir(RveInfo.fig_path):
            os.makedirs(RveInfo.fig_path)

        if not os.path.isdir(RveInfo.gen_path):
            os.makedirs(RveInfo.gen_path)
        if not os.path.isdir(RveInfo.post_path):
            os.makedirs(RveInfo.post_path)

        f_handler = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(RveInfo.store_path, 'result-logs'), when='midnight')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        f_handler.setFormatter(formatter)
        RveInfo.RESULT_LOG.addHandler(f_handler)
        RveInfo.RESULT_LOG.setLevel(level=logging.DEBUG)

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
                total_df = obj2D.grain_sampling()
                rve = obj2D.rve_generation(total_df)
                obj2D.post_processing(rve)

        elif RveInfo.dimension == 3:
            # Kann Gan und nicht GAN
            obj3D = DataTask3D()
            for i in range(RveInfo.number_of_rves):
                self.initializations(i)
                total_df, ex_df = obj3D.grain_sampling()
                rve = obj3D.rve_generation(total_df)
                obj3D.post_processing(rve, total_df, ex_df)


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



