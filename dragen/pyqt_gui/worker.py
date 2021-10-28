from PyQt5.QtCore import QObject, pyqtSignal
from dragen.run import Run


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    info_box = pyqtSignal(str)

    def __init__(self, box_size, box_size_y,box_size_z,resolution, number_of_rves, number_of_bands,
                 bandwidth, dimension, visualization_flag, file1, file2, phase_ratio, store_path,
                 shrink_factor: float = 0.5, band_ratio_rsa: float = 0.75, band_ratio_final: float = 0.75,
                 gui_flag: bool = None, gan_flag: bool = None,
                 info_box_obj=None, progress_obj=None,equiv_d: float=None, p_sigma: float=None,
                 t_mu: float=None, b_sigma: float=0.001, decreasing_factor: float=0.95,
                 lower: float=None, upper: float=None, circularity: float=1, save: bool=True,
                 filename: str=None,subs_file_flag = False,subs_file:str=None,subs_flag:bool=False,phases=['martensite']):

        super().__init__()
        self.box_size = box_size
        self.box_size_y = box_size_y
        self.box_size_z = box_size_z
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
        self.equiv_d = equiv_d
        self.p_sigma = p_sigma
        self.t_mu = t_mu
        self.b_sigma = b_sigma
        self.decreasing_facotr = decreasing_factor
        self.lower = lower
        self.upper = upper
        self.circularity = circularity
        self.save = save
        self.filename = filename
        self.subs_file_flag = subs_file_flag
        self.subs_file = subs_file
        self.subs_flag = subs_flag
        self.phases = phases

    def run(self):
        """Long-running task."""

        run_obj =Run(box_size=self.box_size,box_size_y=self.box_size_y,box_size_z=self.box_size_z,
                     resolution=self.resolution, number_of_rves=self.number_of_rves, number_of_bands=self.number_of_bands,
                     bandwidth=self.bandwidth,dimension=self.dimension, visualization_flag=self.visualization_flag,
                     file1=self.file1, file2=self.file2, phase_ratio=self.phase_ratio, store_path=self.store_path,
                     shrink_factor=self.shrink_factor, band_ratio_rsa=self.band_ratio_rsa, band_ratio_final=self.band_ratio_final,
                     gui_flag=self.gui_flag, gan_flag=self.gan_flag, info_box_obj=self.info_box, progress_obj=self.progress,
                     equiv_d=self.equiv_d, p_sigma=self.p_sigma, t_mu=self.t_mu,
                     b_sigma=self.b_sigma, decreasing_factor=self.decreasing_facotr, lower=self.lower, upper=self.upper,
                     circularity=self.circularity, save=self.save,
                     filename=self.filename,subs_file=self.subs_file,subs_flag=self.subs_flag,phases=self.phases,subs_file_flag=self.subs_file_flag)
        run_obj.run()
        self.finished.emit()
