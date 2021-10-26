import math
import os
import sys
import logging

from dragen.main2D import DataTask2D
from dragen.main3D import DataTask3D
from dragen.main3D_GAN import DataTask3D_GAN
from dragen.postprocessing.voldistribution import PostProcVol
from dragen.pyqt_gui.ScrollLabel import ScrollLabel
from dragen.substructure.run import Run as SubRun

class Run:
    def __init__(self, box_size: int, box_size_y: int = None, box_size_z: int = None, resolution: float = 1, number_of_rves: int = 1,
                 number_of_bands: int = 0, bandwidth: float = 1,
                 dimension: int = 3, visualization_flag: bool = False, file1: str = None, file2: str = None,
                 phase_ratio: float = None, store_path: str = None,
                 shrink_factor: float = 0.5, band_ratio_rsa: float = 0.75, band_ratio_final: float = 0.75,
                 gui_flag: bool = False, gan_flag: bool = False, info_box_obj=None, progress_obj=None,
                 equiv_d: float = None, p_sigma: float = None, t_mu: float = None, b_sigma: float = 0.01,
                 decreasing_factor: float = 0.95,
                 lower: float = None, upper: float = None, circularity: float = 1, plt_name: str = None,
                 save: bool = True,
                 plot: bool = False, filename: str = None, fig_path: str = None, OR: str = 'KS', gen_flag='user_define',
                 subs_file:str =None,phases:list =['ferrite','martensite']):

        self.box_size = box_size
        self.box_size_y = box_size_y
        self.box_size_z = box_size_z
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
        self.equiv_d = equiv_d
        self.p_sigma = p_sigma
        self.t_mu = t_mu
        self.b_sigma = b_sigma
        self.decreasing_factor = decreasing_factor
        self.lower = lower
        self.upper = upper
        self.circularity = circularity
        self.plt_name = plt_name
        self.save = save
        self.plot = plot
        self.filename = filename
        self.fig_path = fig_path
        self.OR = OR
        self.gen_flag = gen_flag
        self.subs_file = subs_file
        self.phases = phases

    def run(self):
        # calculate n_pts from box_size and resolution
        n_pts = math.ceil(float(self.box_size) * self.resolution)
        if n_pts % 2 != 0:
            n_pts += 1
        if self.infobox_obj is not None:
            self.infobox_obj.emit("the chosen resolution lead to {}^{} points in the grid".format(str(n_pts),
                                                                                                 str(self.dimension)))

        sub_run = SubRun(box_size=self.box_size,box_size_y=self.box_size_y,box_size_z=self.box_size_z, n_pts=n_pts,equiv_d=self.equiv_d,p_sigma=self.p_sigma,t_mu=self.t_mu,b_sigma=self.b_sigma,
                          decreasing_factor=self.decreasing_factor,lower=self.lower,upper=self.upper,circularity=self.circularity,plt_name=self.plt_name,
                          save=self.save,plot=self.plot,filename=self.filename,fig_path=self.fig_path,gen_flag=self.gen_flag,
                          subs_file=self.subs_file,OR=self.OR)

        if self.dimension == 2:
            obj2D = DataTask2D(box_size=self.box_size, n_pts=int(n_pts), number_of_bands=self.number_of_bands,
                               bandwidth=self.band_width, shrink_factor=self.shrink_factor,
                               band_ratio_rsa=self.band_ratio_rsa, band_ratio_final=self.band_ratio_final,
                               file1=self.file1, file2=self.file2, phase_ratio=self.phase_ratio,
                               store_path=self.store_path,
                               gui_flag=True, gan_flag=self.gan_flag, anim_flag=self.visualization_flag)

            for i in range(self.number_of_rves):
                grains_df, store_path = obj2D.initializations(self.dimension, epoch=i)
                obj2D.rve_generation(grains_df, store_path)

        elif self.dimension == 3:
            # Teile in zwei Stück für GAN
            if self.gan_flag:
                obj3D = DataTask3D_GAN(box_size=self.box_size,
                                       n_pts=int(n_pts), number_of_bands=self.number_of_bands,
                                       bandwidth=self.band_width, shrink_factor=self.shrink_factor,
                                       band_filling=0.99, phase_ratio=self.phase_ratio, inclusions_ratio=0.01,
                                       inclusions_flag=False,
                                       file1=None, file2=None, store_path=None, gui_flag=self.gui_flag, anim_flag=True,
                                       gan_flag=self.gan_flag, exe_flag=False,sub_run=sub_run)
            else:
                obj3D = DataTask3D(box_size=self.box_size, box_size_y=self.box_size_y, box_size_z=self.box_size_z,
                                   n_pts=int(n_pts), number_of_bands=self.number_of_bands,
                                   bandwidth=self.band_width, shrink_factor=self.shrink_factor,
                                   band_ratio_rsa=self.band_ratio_rsa, band_ratio_final=self.band_ratio_final,
                                   file1=self.file1, file2=self.file2, phase_ratio=self.phase_ratio,
                                   store_path=self.store_path,
                                   gui_flag=self.gui_flag, gan_flag=self.gan_flag, anim_flag=self.visualization_flag,
                                   infobox_obj=self.infobox_obj, progess_obj=self.progress_obj,sub_run=sub_run,phases=self.phases)
            for i in range(self.number_of_rves):
                grains_df, store_path = obj3D.initializations(self.dimension, epoch=i)
                obj3D.rve_generation(grains_df, store_path)
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
    box_size = 16
    box_size_y = None   # if this is None it will be set to the main box_size value
    box_size_z = None     # for sheet rve set z to None and y to different value than x the other way round is buggy
    resolution = 2
    number_of_rves = 1
    number_of_bands = 0
    bandwidth = 5
    visualization_flag = True
    phase_ratio = 1.0
    store_path = 'D:/test'
    shrink_factor = 0.2
    dimension = 3
    gan_flag = False
    # Example Files
    equiv_d = 5
    p_sigma = 0.1
    t_mu = 1.0
    b_sigma = 0.1
    # Example Files
    #file2 = 'F:/git/merged_substructure/ExampleInput/ferrite_54_grains.csv'
    file2 = 'F:/pycharm/2nd_mini_thesis/dragen-master/ExampleInput/example_pag_inp.csv'
    file1 = None
    # file2 = None
    # file1 = 'F:/git/merged_substructure/ExampleInput/ferrite_54_grains.csv'
    # test pearlite phase
    subs_file = 'F:/pycharm/2nd_mini_thesis/dragen-master/ExampleInput/example_block_inp.csv'
    '''
    specific number is fixed for each phase. 1->ferrite, 2->martensite so far. The order of input files should also have the 
    same order as phases. file1->ferrite, file2->martensite. The substructures will only be generated in martensite.
    '''
    phases = ['martensite']
    Run(box_size, box_size_y=box_size_y, box_size_z=box_size_z, resolution=resolution, number_of_rves=number_of_rves,
        number_of_bands=number_of_bands, bandwidth=bandwidth, dimension=dimension,
        visualization_flag=visualization_flag, file1=file1, file2=file2,equiv_d=equiv_d,p_sigma=p_sigma,t_mu=t_mu,b_sigma=b_sigma,
        phase_ratio=phase_ratio, store_path=store_path, shrink_factor=shrink_factor, gui_flag=False, gan_flag=gan_flag,
        info_box_obj=None, progress_obj=None,gen_flag='from_file',subs_file=subs_file,phases=phases).run()
