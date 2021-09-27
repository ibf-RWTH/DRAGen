from PyQt5.QtCore import QObject, pyqtSignal
from dragen.run import Run


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    info_box = pyqtSignal(str)

    def __init__(self, box_size, resolution, number_of_rves, number_of_bands,
                 bandwidth, dimension, visualization_flag, file1, file2, phase_ratio, store_path,equiv_d: float=None, p_sigma: float=None,
                 t_mu: float=None, b_sigma: float=0.001, decreasing_factor: float=0.95,
                 lower: float=None, upper: float=None, circularity: float=1, plt_name: str=None, save: bool=True,
                 plot: bool=False,filename: str=None, fig_path: str=None,
                 shrink_factor: float = 0.4, band_ratio_rsa: float = 0.75, band_ratio_final: float = 0.75,
                 gui_flag: bool = None, gan_flag: bool = None,block_file:str = None,
                 info_box_obj=None, progress_obj=None,gen_flag='user_define'):

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
        self.equiv_d = equiv_d
        self.p_sigma = p_sigma
        self.t_mu = t_mu
        self.b_sigma = b_sigma
        self.decreasing_facotr = decreasing_factor
        self.lower = lower
        self.upper = upper
        self.circularity = circularity
        self.plt_name = plt_name
        self.save = save
        self.plot = plot
        self.filename = filename
        self.figpath = fig_path
        self.gen_flag = gen_flag
        self.blockfile = block_file

    def run(self):
        """Long-running task."""

        run_obj =Run(self.box_size, self.resolution, self.number_of_rves, self.number_of_bands, self.bandwidth,
                     self.dimension, self.visualization_flag, equiv_d=self.equiv_d,p_sigma=self.p_sigma,t_mu=self.t_mu,
                     b_sigma=self.b_sigma,decreasing_factor=self.decreasing_facotr,lower=self.lower,upper=self.upper,
                     circularity=self.circularity,plt_name=self.plt_name,save=self.save,plot=self.plot,filename=self.filename,
                     fig_path=self.figpath,file1=self.file1, file2=self.file2, phase_ratio=self.phase_ratio, store_path=self.store_path,
                     shrink_factor=self.shrink_factor, band_ratio_rsa=self.band_ratio_rsa, band_ratio_final=self.band_ratio_final,
                     gui_flag=self.gui_flag, gan_flag=self.gan_flag, info_box_obj=self.info_box, progress_obj=self.progress,block_file=self.blockfile,gen_flag=self.gen_flag)
        run_obj.run()
        self.finished.emit()
