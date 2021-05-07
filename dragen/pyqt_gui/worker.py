import time

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QProgressBar

from dragen.run import Run
from dragen.pyqt_gui.ScrollLabel import ScrollLabel

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, box_size, resolution, number_of_rves, number_of_bands,
                 bandwidth, dimension, visualization_flag, file1, file2, phase_ratio, store_path,
                 shrink_factor: float = 0.5, band_ratio_rsa: float = 0.75, band_ratio_final: float = 0.75,
                 gui_flag: bool = None, gan_flag: bool = None,
                 info_box_obj=None, progress_obj=None):

        super().__init__()
        self.box_size = box_size
        self.resolution = resolution
        self.number_of_rves = number_of_rves
        self.number_of_bands = number_of_bands
        self.bandwidth = bandwidth
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
        self.info_box_obj = info_box_obj
        self.progress_obj = progress_obj


    def run(self):
        """Long-running task."""

        run_obj =Run(self.box_size, self.resolution, self.number_of_rves, self.number_of_bands, self.bandwidth,
                     self.dimension, self.visualization_flag, self.file1, self.file2, self.phase_ratio, self.store_path,
                     self.shrink_factor, self.band_ratio_rsa, self.band_ratio_final,
                     self.gui_flag, self.gan_flag, info_box_obj=self.info_box_obj, progress_obj=self.progress_obj)
        run_obj.run()
        self.finished.emit()
