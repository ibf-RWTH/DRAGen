import os
import sys
import datetime
import numpy as np

import logging
import logging.handlers
import pandas as pd

from dragen.generation.DiscreteRsa3D import DiscreteRsa3D
from dragen.generation.DiscreteTesselation3D import Tesselation3D
from dragen.utilities.RVE_Utils import RVEUtils
from dragen.generation.mesh_subs import SubMesher
from dragen.postprocessing.voldistribution import PostProcVol

PHASENUM = {'ferrite':1,'martensite':2} #fix number for the different phases

class DataTask3D(RVEUtils):

    def __init__(self, box_size: int, box_size_y, box_size_z,
                 n_pts: int, number_of_bands: int, bandwidth: float, shrink_factor: float = 0.5,
                 band_ratio_rsa: float = 0.95, band_ratio_final: float = 0.95, phase_ratio: float = None, file1=None,
                 file2=None, store_path=None, gui_flag=False, anim_flag=False, gan_flag=False, exe_flag=False,
                 infobox_obj=None, progess_obj=None,sub_run=None,phases:list =['pearlite','ferrite']):

        self.logger = logging.getLogger("RVE-Gen")
        self.box_size = box_size
        self.box_size_y = box_size_y
        self.box_size_z = box_size_z
        self.n_pts = n_pts  # has to be even
        self.bin_size = self.box_size / self.n_pts
        self.step_half = self.bin_size / 2
        self.number_of_bands = number_of_bands
        self.bandwidth = bandwidth
        self.phase_ratio = phase_ratio
        self.shrink_factor = float(np.cbrt(shrink_factor))
        self.band_ratio_rsa = band_ratio_rsa  # Band Ratio for RSA
        self.band_ratio_final = band_ratio_final  # Band ratio for Tesselator - final is br1 * br2
        self.gui_flag = gui_flag
        self.gan_flag = gan_flag
        self.root_dir = store_path
        self.store_path = store_path
        self.fig_path = store_path+'Figs'
        self.gen_path = store_path+'Generation_Data'
        self.infobox_obj = infobox_obj
        self.progress_obj = progess_obj

        self.logger.info('the exe_flag is: ' + str(exe_flag))
        self.logger.info('root was set to: ' + self.root_dir)
        self.animation = anim_flag
        self.file1 = file1
        self.file2 = file2

        self.x_grid, self.y_grid, self.z_grid = super().gen_grid()
        self.sub_run = sub_run
        self.phases = phases
        super().__init__(box_size,  n_pts, box_size_y=box_size_y, box_size_z=box_size_z,
                         x_grid=self.x_grid, y_grid=self.y_grid, z_grid=self.z_grid, bandwidth=bandwidth, debug=False)

    def setup_logging(self):
        LOGS_DIR = self.root_dir + '/Logs/'
        if not os.path.isdir(LOGS_DIR):
            os.makedirs(LOGS_DIR)
        f_handler = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(LOGS_DIR, 'dragen-logs'), when='midnight')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        f_handler.setFormatter(formatter)
        self.logger.addHandler(f_handler)
        self.logger.setLevel(level=logging.DEBUG)

    def initializations(self, dimension, epoch):

        self.setup_logging()

        phase1_csv = self.file1
        phase2_csv = self.file2
        files = [phase1_csv,phase2_csv]
        self.logger.info("RVE generation process has started...")

        total_df = pd.DataFrame()
        for phase in self.phases:
            file_idx = PHASENUM[phase] - 1
            print('current phase is',phase,';phase input file is',files[file_idx])
            phase_input_df = super().read_input(files[file_idx], dimension)
            if self.box_size_y is None and self.box_size_z is None:
                adjusted_size = np.cbrt((self.box_size ** 3 -
                                        (self.box_size ** 2 * self.number_of_bands * self.bandwidth))
                                        * self.phase_ratio)
                grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)

            elif self.box_size_y is not None and self.box_size_z is None:
                adjusted_size = np.cbrt((self.box_size ** 2 * self.box_size_y -
                                         (self.box_size ** 2 * self.number_of_bands * self.bandwidth))
                                        * self.phase_ratio)
                grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)
            elif self.box_size_y is None and self.box_size_z is not None:
                adjusted_size = np.cbrt((self.box_size ** 2 * self.box_size_z -
                                         (self.box_size ** 2 * self.number_of_bands * self.bandwidth))
                                        * self.phase_ratio)
                grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)

            else:
                adjusted_size = np.cbrt((self.box_size * self.box_size_y * self.box_size_z -
                                         (self.box_size ** 2 * self.number_of_bands * self.bandwidth))
                                        * self.phase_ratio)
                grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)

            grains_df['phaseID'] = PHASENUM[phase]
            total_df = pd.concat([total_df,grains_df])

        grains_df = super().process_df(total_df, self.shrink_factor)

        total_volume = sum(grains_df['final_conti_volume'].values)
        estimated_boxsize = np.cbrt(total_volume)
        self.logger.info("the total volume of your dataframe is {}. A boxsize of {} is recommended.".
                         format(total_volume, estimated_boxsize))

        self.store_path = self.root_dir + '/OutputData/' + str(datetime.datetime.now())[:10] + '_' + str(epoch)
        self.fig_path = self.store_path + '/Figs'
        self.gen_path = self.store_path + '/Generation_Data'

        if not os.path.isdir(self.store_path):
            os.makedirs(self.store_path)
        if not os.path.isdir(self.fig_path):
            os.makedirs(self.fig_path)  # Second if needed
        if not os.path.isdir(self.gen_path):
            os.makedirs(self.gen_path)  # Second if needed

        grains_df.to_csv(self.gen_path + '/grain_data_input.csv', index=False)
        return grains_df, self.store_path

    def rve_generation(self, grains_df, store_path) -> str:

        discrete_RSA_obj = DiscreteRsa3D(self.box_size, self.box_size_y, self.box_size_z, self.n_pts,
                                         grains_df['a'].tolist(),
                                         grains_df['b'].tolist(),
                                         grains_df['c'].tolist(),
                                         grains_df['alpha'].tolist(),
                                         store_path=store_path,
                                         infobox_obj=self.infobox_obj,
                                         progress_obj=self.progress_obj)

        if self.number_of_bands > 0:
            # initialize empty grid_array for bands called band_array

            band_array = super().gen_array()
            band_array = super().gen_boundaries_3D(band_array)

            for i in range(self.number_of_bands):
                band_array = super().band_generator(band_array)

            rsa, x_0_list, y_0_list, z_0_list, rsa_status = discrete_RSA_obj.run_rsa(self.band_ratio_rsa, band_array,
                                                                                     animation=self.animation,
                                                                                     gui=self.gui_flag)
            grains_df['x_0'] = x_0_list
            grains_df['y_0'] = y_0_list
            grains_df['z_0'] = z_0_list

        else:
            print(grains_df.head(1))
            rsa, x_0_list, y_0_list, z_0_list, rsa_status = discrete_RSA_obj.run_rsa(animation=self.animation,
                                                                                     gui=self.gui_flag)
            grains_df['x_0'] = x_0_list
            grains_df['y_0'] = y_0_list
            grains_df['z_0'] = z_0_list

        if rsa_status:
            discrete_tesselation_obj = Tesselation3D(self.box_size, self.box_size_y, self.box_size_z, self.n_pts,
                                                     grains_df, self.shrink_factor, self.band_ratio_final, store_path,
                                                     infobox_obj=self.infobox_obj, progress_obj=self.progress_obj)
            rve, rve_status = discrete_tesselation_obj.run_tesselation(rsa, animation=self.animation, gui=self.gui_flag)

        else:
            self.logger.info("The rsa did not succeed...")
            sys.exit()

        if rve_status:
            periodic_rve_df = super().repair_periodicity_3D(rve)
            periodic_rve_df['phaseID'] = 0
            print('len rve edge:', np.cbrt(len(periodic_rve_df)))
            # An den NaN-Werten in dem DF liegt es nicht!

            grains_df.sort_values(by=['GrainID'])
            #debug_df = grains_df.copy()
            for i in range(len(grains_df)):
                # Set grain-ID to number of the grain
                # Denn Grain-ID ist entweder >0 oder -200 oder >-200
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == i+1, 'phaseID'] = grains_df['phaseID'][i]
                #debug_df.loc[debug_df.index == i, 'vol_rve_df'] = \
                #    len(periodic_rve_df.loc[periodic_rve_df['GrainID'] == i + 1])*self.bin_size**3



            if self.number_of_bands > 0:
                # Set the points where == -200 to phase 2 and to grain ID i + 2
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == -200, 'GrainID'] = (i + 2)
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == (i + 2), 'phaseID'] = 2

            # Start the Mesher
            #grains_df.to_csv('grains_df.csv', index=False)
            #periodic_rve_df.to_csv('periodic_rve_df.csv', index=False)
            subs_rve = self.sub_run.run(rve_df=periodic_rve_df, grains_df=grains_df, store_path=self.store_path,
                                        logger=self.logger) # returns rve df containing substructures
            mesher_obj = SubMesher(box_size_x=self.box_size, box_size_y=self.box_size_y, box_size_z=self.box_size_z,
                                rve=subs_rve,subs_df=grains_df, store_path=store_path,
                                phase_two_isotropic=False, animation=self.animation,
                                infobox_obj=self.infobox_obj, progress_obj=self.progress_obj, gui=self.gui_flag,
                                element_type='C3D8')
            mesher_obj.mesh_and_build_abaqus_model()
        return store_path

    def post_processing(self):
        obj = PostProcVol(self.store_path, dim_flag=3)
        phase1_ratio_conti_in, phase1_ref_r_conti_in, phase1_ratio_discrete_in, phase1_ref_r_discrete_in, \
        phase2_ratio_conti_in, phase2_ref_r_conti_in, phase2_ratio_discrete_in, phase2_ref_r_discrete_in, \
        phase1_ratio_conti_out, phase1_ref_r_conti_out, phase1_ratio_discrete_out, phase1_ref_r_discrete_out, \
        phase2_ratio_conti_out, phase2_ref_r_conti_out, phase2_ratio_discrete_out, phase2_ref_r_discrete_out = \
            obj.gen_in_out_lists()


        print(phase2_ratio_conti_in)
        if len(self.phases) > 1:

            obj.gen_pie_chart_phases(phase1_ratio_conti_in, phase2_ratio_conti_in, 'input_conti')
            obj.gen_pie_chart_phases(phase1_ratio_conti_out, phase2_ratio_conti_out, 'output_conti')
            obj.gen_pie_chart_phases(phase1_ratio_discrete_in, phase2_ratio_discrete_in, 'input_discrete')
            obj.gen_pie_chart_phases(phase1_ratio_discrete_out, phase2_ratio_discrete_out, 'output_discrete')

            obj.gen_plots(phase1_ref_r_discrete_in, phase1_ref_r_discrete_out, 'phase 1 discrete', 'phase1vs2_discrete',
                          phase2_ref_r_discrete_in, phase2_ref_r_discrete_out, 'phase 2 discrete')
            obj.gen_plots(phase1_ref_r_conti_in, phase1_ref_r_conti_out, 'phase 1 conti', 'phase1vs2_conti',
                          phase2_ref_r_conti_in, phase2_ref_r_conti_out, 'phase 2 conti')
            if self.gui_flag:
                self.infobox_obj.emit('checkout the evaluation report of the rve stored at:\n'
                                  '{}/Postprocessing'.format(self.store_path))
        else:
            print('the only phase is {}'.format(self.phases[0]))
            if self.phases[0] == 'ferrite':
                obj.gen_plots(phase1_ref_r_conti_in, phase1_ref_r_conti_out, 'conti', 'in_vs_out_conti')
                obj.gen_plots(phase1_ref_r_discrete_in, phase1_ref_r_discrete_out, 'discrete', 'in_vs_out_discrete')

            elif self.phases[0] == 'martensite':
                obj.gen_plots(phase2_ref_r_conti_in, phase2_ref_r_conti_out, 'conti', 'in_vs_out_conti')
                obj.gen_plots(phase2_ref_r_discrete_in, phase2_ref_r_discrete_out, 'discrete', 'in_vs_out_discrete')


        self.logger.info("RVE generation process has successfully completed...")
