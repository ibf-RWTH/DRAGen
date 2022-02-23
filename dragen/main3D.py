import sys
import numpy as np

import pandas as pd

from dragen.generation.DiscreteRsa3D import DiscreteRsa3D
from dragen.generation.DiscreteTesselation3D import Tesselation3D
from dragen.utilities.Helpers import HelperFunctions
from dragen.generation.mesh_subs import SubMesher
from dragen.generation.Mesher3D import AbaqusMesher
from dragen.generation.mooseMesher import MooseMesher
from dragen.postprocessing.voldistribution import PostProcVol
from dragen.utilities.InputInfo import RveInfo
from dragen.InputGenerator.C_WGAN_GP import WGANCGP

import dragen.generation.spectral as spectral


class DataTask3D(HelperFunctions):

    def __init__(self):
        super().__init__()

    def grain_sampling(self):
        """
        In this function the correct number of grains for the chosen Volume is sampled from given csv files
        """
        files = RveInfo.file_dict
        RveInfo.logger.info("RVE generation process has started...")
        total_df = pd.DataFrame()

        for phase in RveInfo.phases:
            file_idx = RveInfo.PHASENUM[phase]
            print('current phase is', phase, ';phase input file is', files[file_idx])
            phase_input_df = super().read_input(files[file_idx], RveInfo.dimension)
            if phase == 'Inclusions':  # Das fehlte vorher
                phase_ratio = RveInfo.inclusion_ratio
            elif phase == 'ferrite':
                phase_ratio = RveInfo.phase_ratio
            elif phase == 'martensite':
                phase_ratio = 1 - RveInfo.phase_ratio

            if RveInfo.box_size_y is None and RveInfo.box_size_z is None:
                adjusted_size = np.cbrt((RveInfo.box_size ** 3 -
                                         (RveInfo.box_size ** 2 * RveInfo.number_of_bands * RveInfo.band_width))
                                        * phase_ratio)
                grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)
            elif RveInfo.box_size_y is not None and RveInfo.box_size_z is None:
                adjusted_size = np.cbrt((RveInfo.box_size ** 2 * RveInfo.box_size_y -
                                         (RveInfo.box_size ** 2 * RveInfo.number_of_bands * RveInfo.band_width))
                                        * phase_ratio)
                grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)
            elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
                adjusted_size = np.cbrt((RveInfo.box_size ** 2 * RveInfo.box_size_z -
                                         (RveInfo.box_size ** 2 * RveInfo.number_of_bands * RveInfo.band_width))
                                        * phase_ratio)
                grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)
            else:
                adjusted_size = np.cbrt((RveInfo.box_size * RveInfo.box_size_y * RveInfo.box_size_z -
                                         (RveInfo.box_size ** 2 * RveInfo.number_of_bands * RveInfo.band_width))
                                        * phase_ratio)
                grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)

            grains_df['phaseID'] = RveInfo.PHASENUM[phase]
            total_df = pd.concat([total_df, grains_df])

        grains_df = super().process_df(total_df, RveInfo.shrink_factor)

        total_volume = sum(
            grains_df[grains_df['phaseID'] != 5]['final_conti_volume'].values)  # Inclusions dont influence filling
        estimated_boxsize = np.cbrt(total_volume)
        RveInfo.logger.info("the total volume of your dataframe is {}. A boxsize of {} is recommended.".
                            format(total_volume, estimated_boxsize))

        grains_df.to_csv(RveInfo.gen_path + '/grain_data_input.csv', index=False)
        print(grains_df)
        return grains_df

    def rve_generation(self, grains_df):

        print(grains_df.__len__())
        print(grains_df[grains_df['phaseID'] == 5].__len__())
        normal_grains_df = grains_df[grains_df['phaseID'] <= 4]
        print(normal_grains_df.__len__())
        discrete_RSA_obj = DiscreteRsa3D(normal_grains_df['a'].tolist(),  # TODO: Das ist noch mit Inclusions hier
                                         normal_grains_df['b'].tolist(),
                                         normal_grains_df['c'].tolist(),
                                         normal_grains_df['alpha'].tolist())

        if RveInfo.number_of_bands > 0:
            # TODO: @Niklas add the new band process here
            print('Not implemented')
            breakpoint()

        else:
            rsa, x_0_list, y_0_list, z_0_list, rsa_status = discrete_RSA_obj.run_rsa()
            normal_grains_df['x_0'] = x_0_list
            normal_grains_df['y_0'] = y_0_list
            normal_grains_df['z_0'] = z_0_list

        if rsa_status:
            discrete_tesselation_obj = Tesselation3D(normal_grains_df)
            rve, rve_status = discrete_tesselation_obj.run_tesselation(rsa)

        else:
            RveInfo.logger.info("The rsa did not succeed...")
            sys.exit()

        """
        PLACE THE INCLUSIONS!
        """
        if rve_status and RveInfo.inclusion_flag and RveInfo.inclusion_ratio != 0:
            inclusions_df = grains_df[grains_df['phaseID'] == 5]
            discrete_RSA_inc_obj = DiscreteRsa3D(inclusions_df['a'].tolist(),
                                                 inclusions_df['b'].tolist(),
                                                 inclusions_df['c'].tolist(),
                                                 inclusions_df['alpha'].tolist())

            rve, rve_status = discrete_RSA_inc_obj.run_rsa_inclusions(rve)

        """
        GENERATE INOUT DATA FOR SIMULATIONS HERE
        """
        if rsa_status:
            periodic_rve_df = super().repair_periodicity_3D(rve)
            periodic_rve_df['phaseID'] = 0
            print('len rve edge:', np.cbrt(len(periodic_rve_df)))
            # An den NaN-Werten in dem DF liegt es nicht!

            grains_df.sort_values(by=['GrainID'])
            # debug_df = grains_df.copy()
            for i in range(len(grains_df)):
                # Set grain-ID to number of the grain
                # Denn Grain-ID ist entweder >0 oder -200 oder >-200
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == i + 1, 'phaseID'] = grains_df['phaseID'][i]
                # debug_df.loc[debug_df.index == i, 'vol_rve_df'] = \
                #    len(periodic_rve_df.loc[periodic_rve_df['GrainID'] == i + 1])*RveInfo.bin_size**3

            if RveInfo.number_of_bands > 0:
                # Set the points where == -200 to phase 2 and to grain ID i + 2
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == -200, 'GrainID'] = (i + 2)
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == (i + 2), 'phaseID'] = 2

            # Start the Mesher
            # grains_df.to_csv('grains_df.csv', index=False)
            # periodic_rve_df.to_csv('periodic_rve_df.csv', index=False)
            rve_shape = (max(np.where(rve > 0)[0]) - min(np.where(rve > 0)[0]) + 1,
                         max(np.where(rve > 0)[1]) - min(np.where(rve > 0)[1]) + 1,
                         max(np.where(rve > 0)[2]) - min(np.where(rve > 0)[2]) + 1)

            if RveInfo.damask_flag:
                # 1.) Write out Volume
                grains_df.sort_values(by=['GrainID'], inplace=True)
                disc_vols = np.zeros((1, grains_df.shape[0])).flatten().tolist()
                for i in range(len(grains_df)):
                    disc_vols[i] = np.count_nonzero(rve == i + 1) * RveInfo.bin_size ** 3

                grains_df['meshed_conti_volume'] = disc_vols
                grains_df.to_csv(RveInfo.store_path + '/Generation_Data/grain_data_output_conti.csv', index=False)

                # Startpoint: Rearrange the negative ID's
                last_grain_id = rve.max()  # BEWARE: For the .vti file, the grid must start at ZERO
                print('The last grain ID is:', last_grain_id)
                print('The number of bands is:', RveInfo.number_of_bands)

                if RveInfo.number_of_bands >= 1:
                    print('Nur Bänder')
                    phase_list = grains_df['phaseID'].tolist()
                    rve[np.where(rve == -200)] = last_grain_id + 1
                    phase_list.append(2)

                if RveInfo.inclusion_ratio > 0 and (RveInfo.inclusion_flag is True):
                    print('Nur Inclusions')
                    phase_list = grains_df['phaseID'].tolist()
                    for i in range(len(inclusions_df)):
                        rve[np.where(rve == -(200 + i + 1))] = last_grain_id + i + 1
                    print(phase_list.__len__())

                else:
                    print('Keine Bänder, nur grains')
                    phase_list = grains_df['phaseID'].tolist()

                spectral.write_material(store_path=RveInfo.store_path, grains=phase_list)
                spectral.write_load(RveInfo.store_path)
                spectral.write_grid(store_path=RveInfo.store_path,
                                    rve=rve,
                                    spacing=RveInfo.box_size / 1000)

            if RveInfo.moose_flag:
                MooseMesher(rve_shape=rve_shape, rve=periodic_rve_df, grains_df=grains_df).run()

            if RveInfo.abaqus_flag:
                mesher_obj = None
                if RveInfo.subs_flag == True:
                    print("substructure generation is turned on...")
                    # returns rve df containing substructures
                    subs_rve = RveInfo.sub_run.run(rve_df=periodic_rve_df, grains_df=grains_df)
                    mesher_obj = SubMesher(rve_shape=rve_shape, rve=subs_rve, subs_df=grains_df)

                elif RveInfo.subs_flag == False:
                    print("substructure generation is turned off...")
                    mesher_obj = AbaqusMesher(rve_shape=rve_shape, rve=periodic_rve_df, grains_df=grains_df)
                if mesher_obj:
                    mesher_obj.run()

    def post_processing(self):

        phase1_ratio_conti_in, phase1_ref_r_conti_in, phase1_ratio_discrete_in, phase1_ref_r_discrete_in, \
        phase2_ratio_conti_in, phase2_ref_r_conti_in, phase2_ratio_discrete_in, phase2_ref_r_discrete_in, \
        phase1_ratio_conti_out, phase1_ref_r_conti_out, phase1_ratio_discrete_out, phase1_ref_r_discrete_out, \
        phase2_ratio_conti_out, phase2_ref_r_conti_out, phase2_ratio_discrete_out, phase2_ref_r_discrete_out = \
            PostProcVol().gen_in_out_lists()

        print(phase2_ratio_conti_in)
        if len(RveInfo.phases) > 1:

            PostProcVol().gen_pie_chart_phases(phase1_ratio_conti_in, phase2_ratio_conti_in, 'input_conti')
            PostProcVol().gen_pie_chart_phases(phase1_ratio_conti_out, phase2_ratio_conti_out, 'output_conti')
            PostProcVol().gen_pie_chart_phases(phase1_ratio_discrete_in, phase2_ratio_discrete_in, 'input_discrete')
            PostProcVol().gen_pie_chart_phases(phase1_ratio_discrete_out, phase2_ratio_discrete_out, 'output_discrete')

            PostProcVol().gen_plots(phase1_ref_r_discrete_in, phase1_ref_r_discrete_out, 'phase 1 discrete',
                                    'phase1vs2_discrete',
                                    phase2_ref_r_discrete_in, phase2_ref_r_discrete_out, 'phase 2 discrete')
            PostProcVol().gen_plots(phase1_ref_r_conti_in, phase1_ref_r_conti_out, 'phase 1 conti', 'phase1vs2_conti',
                                    phase2_ref_r_conti_in, phase2_ref_r_conti_out, 'phase 2 conti')
            if RveInfo.gui_flag:
                RveInfo.infobox_obj.emit('checkout the evaluation report of the rve stored at:\n'
                                         '{}/Postprocessing'.format(RveInfo.store_path))
        else:
            print('the only phase is {}'.format(RveInfo.phases[0]))
            if RveInfo.phases[0] == 'ferrite':
                PostProcVol().gen_plots(phase1_ref_r_conti_in, phase1_ref_r_conti_out, 'conti', 'in_vs_out_conti')
                PostProcVol().gen_plots(phase1_ref_r_discrete_in, phase1_ref_r_discrete_out, 'discrete',
                                        'in_vs_out_discrete')

            elif RveInfo.phases[0] == 'martensite':
                PostProcVol().gen_plots(phase2_ref_r_conti_in, phase2_ref_r_conti_out, 'conti', 'in_vs_out_conti')
                PostProcVol().gen_plots(phase2_ref_r_discrete_in, phase2_ref_r_discrete_out, 'discrete',
                                        'in_vs_out_discrete')

        if RveInfo.subs_flag:
            RveInfo.sub_run.post_processing(k=3)
        RveInfo.logger.info("RVE generation process has successfully completed...")
