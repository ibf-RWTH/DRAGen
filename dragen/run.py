import math
import os
import sys
import logging

from PyQt5.QtWidgets import QMessageBox, QProgressBar
from dragen.main2D import DataTask2D
from dragen.main3D import DataTask3D
from dragen.postprocessing.voldistribution import PostProcVol
from dragen.pyqt_gui.ScrollLabel import ScrollLabel


class Run:
    def __init__(self, box_size: int, resolution: float, number_of_rves: int, number_of_bands: int, bandwidth: float,
                 dimension: int, visualization_flag: bool, file1: str = None, file2: str = None,
                 phase_ratio: float = None, store_path: str = None,
                 shrink_factor: float = 0.5, band_ratio_rsa: float = 0.75, band_ratio_final: float = 0.75,
                 gui_flag: bool = False, gan_flag: bool = False, info_box_obj: ScrollLabel = None,
                 progress_obj=None):

        self.box_size = box_size
        self.resolution = resolution
        self.number_of_rves = number_of_rves
        self.number_of_bands = number_of_bands
        self.band_width = bandwidth
        self.dimension = dimension
        self.visualization_flag = visualization_flag
        self.file1 = file1
        self.file2 = file2
        self.phase_ratio = phase_ratio
        self.store_path = store_path
        self.shrink_factor = shrink_factor
        self.band_ratio_rsa = band_ratio_rsa
        self.band_ratio_final = band_ratio_final
        self.gui_flag = gui_flag
        self.gan_flag = gan_flag
        self.infobox_obj = info_box_obj
        self.progress_obj = progress_obj

    def run(self):


        # calculate n_pts from box_size and resolution
        n_pts = math.ceil(float(self.box_size) * self.resolution)
        if n_pts % 2 != 0:
            n_pts += 1

        self.infobox_obj.add_text("the chosen resolution lead to {}^{} points in the grid".format(str(n_pts),
                              str(self.dimension)))

        obj3D = DataTask3D(box_size=self.box_size, n_pts=int(n_pts), number_of_bands=self.number_of_bands,
                           bandwidth=self.band_width, shrink_factor=self.shrink_factor,
                           band_ratio_rsa=self.band_ratio_rsa, band_ratio_final=self.band_ratio_final,
                           file1=self.file1, file2=self.file2, phase_ratio=self.phase_ratio, store_path=self.store_path,
                           gui_flag=True, gan_flag=self.gan_flag, anim_flag=self.visualization_flag,
                           infobox_obj=self.infobox_obj, progess_obj = self.progress_obj)

        obj2D = DataTask2D(box_size=self.box_size, n_pts=n_pts, number_of_bands=self.number_of_bands,
                           bandwidth=self.band_width, shrink_factor=self.shrink_factor,
                           band_ratio_rsa=self.band_ratio_rsa, band_ratio_final=self.band_ratio_final,
                           file1=self.file1,
                           file2=self.file2,
                           gui_flag=True, gan_flag=self.gan_flag, anim_flag=self.visualization_flag)

        if self.dimension == 2:
            grains_df = obj2D.initializations(self.dimension)
            for i in range(self.number_of_rves):
                obj2D.rve_generation(i, grains_df)

        elif self.dimension == 3:
            for i in range(self.number_of_rves):
                grains_df, store_path = obj3D.initializations(self.dimension, epoch=i)
                obj3D.rve_generation(grains_df, store_path)
                PostProcVol_obj = PostProcVol(store_path, dim_flag=3)
                PostProcVol_obj.gen_in_out_lists()
                PostProcVol_obj.gen_plots()

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
    last_RVE = 1  # Specify the number of iterations
    dimension = 3
    # Optional arguments with default values:
    # obj3D = DataTask3D()
    obj3D = DataTask3D(box_size=22, n_pts=34, number_of_bands=0, bandwidth=5,
                       shrink_factor=0.5, band_ratio_rsa=0.75, band_ratio_final=0.75,
                       file1='../ExampleInput/ferrite_54_grains.csv',
                       file2='../ExampleInput/pearlite_21_grains.csv',
                       gui_flag=False, gan_flag=True, anim_flag=False)

    obj2D = DataTask2D(box_size=20, n_pts=40, number_of_bands=1, bandwidth=5,
                       shrink_factor=0.5, band_ratio_rsa=0.75, band_ratio_final=0.75,
                       file1='../ExampleInput/ferrite_54_grains.csv',
                       file2='../ExampleInput/pearlite_21_grains.csv',
                       gui_flag=False, gan_flag=True, anim_flag=True)

    try:
        if dimension == 2:
            grains_df = obj2D.initializations(dimension)
            for i in range(last_RVE):
                obj2D.rve_generation(i, grains_df)
            sys.exit()
        elif dimension == 3:
            for i in range(last_RVE):
                grains_df, store_path = obj3D.initializations(dimension, epoch=i)
                obj3D.rve_generation(grains_df, store_path)
                PostProcVol_obj = PostProcVol(store_path, dim_flag=3)
                PostProcVol_obj.gen_in_out_lists()
                PostProcVol_obj.gen_plots()
            sys.exit()
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
    except KeyboardInterrupt:
        print('Exiting!!!')
