from PyQt5.QtCore import QObject, pyqtSignal
from dragen.run import Run


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    info_box = pyqtSignal(str)

    def __init__(self, ARGS):

        super().__init__()
        self.root = ARGS['root']

        self.box_size = ARGS['box_size']
        self.box_size_y = ARGS['box_size_y']
        self.box_size_z = ARGS['box_size_z']
        self.resolution = ARGS['resolution']
        self.number_of_rves = ARGS['number_of_rves']
        self.dimension = ARGS['dimension']

        self.slope_offset = ARGS['slope_offset']
        self.smoothing_flag = ARGS['smoothing']

        self.number_of_bands = ARGS['number_of_bands']
        self.lower_band_bound = ARGS['lower_band_bound']
        self.upper_band_bound = ARGS['upper_band_bound']
        self.band_orientation = ARGS['band_orientation']

        self.band_filling = ARGS['band_filling']

        self.inclusion_flag = ARGS['inclusion_flag']
        self.inclusion_ratio = ARGS['inclusion_ratio']

        self.visualization_flag = ARGS['visualization_flag']
        self.file_dict = ARGS['files']
        self.phase_ratio = ARGS['phase_ratio']
        self.store_path = ARGS['store_path']

        self.subs_flag = ARGS['subs_flag']
        self.subs_file_flag = ARGS['subs_file_flag']
        self.subs_file = ARGS['subs_file']
        self.orientation_relationship = ARGS['orientation_relationship']
        self.sub_run = ARGS['subrun']

        self.pak_file = ARGS['pak_file']
        self.equiv_d = ARGS['equiv_d']
        self.circularity = ARGS['circularity']
        self.p_sigma = ARGS['p_sigma']

        self.block_file = ARGS['block_file']  # TODO: ???
        self.t_mu = ARGS['t_mu']  # TODO: ???
        self.b_sigma = ARGS['b_sigma']  # TODO: ???
        self.decreasing_facotr = ARGS['decreasing_factor']  # TODO: ???
        self.lower = ARGS['lower']  # TODO: ???
        self.upper = ARGS['upper']  # TODO: ???
        self.plt_name = ARGS['plt_name']  # TODO: ???
        self.save = ARGS['save']  # TODO: ???
        self.plot = ARGS['plot']  # TODO: ???
        self.filename = ARGS['filename']  # TODO: ???
        self.fig_path = ARGS['fig_path']  # TODO: ???
        self.gen_path = ARGS['gen_path']  # TODO: ???
        self.post_path = ARGS['post_path']

        self.phases = ARGS['phases']
        self.abaqus_flag = ARGS['abaqus_flag']
        self.damask_flag = ARGS['damask_flag']
        self.moose_flag = ARGS['moose_flag']
        self.anim_flag = ARGS['anim_flag']
        self.phase2iso_flag = ARGS['phase2iso_flag']
        self.pbc_flag = ARGS['pbc_flag']
        self.submodel_flag = ARGS['submodel_flag']
        self.element_type = ARGS['element_type']

        self.gui_flag = ARGS['gui_flag']


    def run(self):
        """Long-running task."""
        run_obj = Run(dimension=self.dimension, box_size=self.box_size, box_size_y=self.box_size_y,
                      box_size_z=self.box_size_z, resolution=self.resolution, number_of_rves=self.number_of_rves,
                      slope_offset=self.slope_offset, abaqus_flag=self.abaqus_flag, damask_flag=self.damask_flag,
                      moose_flag=self.moose_flag, element_type=self.element_type, pbc_flag=self.pbc_flag,
                      submodel_flag=self.submodel_flag, phase2iso_flag=self.phase2iso_flag,
                      smoothing_flag=self.smoothing_flag, gui_flag=self.gui_flag, anim_flag=self.anim_flag,
                      visualization_flag=self.visualization_flag, root=self.root, info_box_obj=self.info_box,
                      progress_obj=self.progress, phase_ratio=self.phase_ratio, file_dict=self.file_dict,
                      phases=self.phases, number_of_bands=self.number_of_bands, upper_band_bound=self.upper_band_bound,
                      lower_band_bound=self.lower_band_bound, band_orientation=self.band_orientation,
                      band_filling=self.band_filling, inclusion_flag=self.inclusion_flag,
                      inclusion_ratio=self.inclusion_ratio, subs_flag=self.subs_flag,
                      subs_file_flag=self.subs_file_flag, subs_file=self.subs_file, equiv_d=self.equiv_d,
                      p_sigma=self.p_sigma, t_mu=self.t_mu, b_sigma=self.b_sigma,
                      decreasing_factor=self.decreasing_facotr, lower=self.lower, upper=self.upper,
                      circularity=self.circularity, plt_name=self.plt_name, save=self.save, plot=self.plot,
                      filename=self.filename, fig_path=self.fig_path,
                      orientation_relationship=self.orientation_relationship)
        run_obj.run()
        self.finished.emit()
