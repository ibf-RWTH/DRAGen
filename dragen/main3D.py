import sys
import math
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dragen.generation.DiscreteRsa3D import DiscreteRsa3D
from dragen.generation.DiscreteTesselation3D import Tesselation3D
from dragen.utilities.Helpers import HelperFunctions
from dragen.generation.mesh_subs import SubMesher
from dragen.generation.Mesher3D import AbaqusMesher
from dragen.generation.mooseMesher import MooseMesher
from dragen.postprocessing.voldistribution import PostProcVol
from dragen.postprocessing.Shape_analysis import shape
from dragen.utilities.InputInfo import RveInfo
from dragen.substructure.run import Run as substrucRun
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
        RveInfo.LOGGER.info("RVE generation process has started...")
        total_df = pd.DataFrame()

        # TODO: Generiere Bandwidths hier!
        if RveInfo.number_of_bands > 0:
            low = RveInfo.lower_band_bound
            high = RveInfo.upper_band_bound
            RveInfo.bandwidths = np.random.uniform(low=low, high=high, size=RveInfo.number_of_bands)
            print(RveInfo.bandwidths)
            sum_bw = RveInfo.bandwidths.sum()
        else:
            sum_bw = 0
        input_data = pd.DataFrame()
        for phase in RveInfo.phases:
            file_idx = RveInfo.PHASENUM[phase]

            print('current phase is', phase, ';phase input file is', files[file_idx])
            print('current phase is', phase, ';phase ratio file is', RveInfo.phase_ratio[file_idx])

            # Check file ending:
            if files[file_idx].endswith('.csv'):
                phase_input_df = super().read_input(files[file_idx], RveInfo.dimension)
            elif files[file_idx].endswith('.pkl'):
                phase_input_df = super().read_input_gan(files[file_idx], RveInfo.dimension, size=1000)

            if phase != 'Bands':
                if RveInfo.box_size_y is None and RveInfo.box_size_z is None:

                    adjusted_size = np.cbrt((RveInfo.box_size ** 3 -
                                             (RveInfo.box_size ** 2 * sum_bw))
                                            * RveInfo.phase_ratio[file_idx])
                    grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)
                elif RveInfo.box_size_y is not None and RveInfo.box_size_z is None:
                    adjusted_size = np.cbrt((RveInfo.box_size ** 2 * RveInfo.box_size_y -
                                             (RveInfo.box_size ** 2 * sum_bw))
                                            * RveInfo.phase_ratio[file_idx])
                    grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)
                elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
                    adjusted_size = np.cbrt((RveInfo.box_size ** 2 * RveInfo.box_size_z -
                                             (RveInfo.box_size ** 2 * sum_bw))
                                            * RveInfo.phase_ratio[file_idx])
                    grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)
                else:
                    adjusted_size = np.cbrt((RveInfo.box_size * RveInfo.box_size_y * RveInfo.box_size_z -
                                             (RveInfo.box_size ** 2 * sum_bw))
                                            * RveInfo.phase_ratio[file_idx])
                    grains_df = super().sample_input_3D(phase_input_df, bs=adjusted_size)

                grains_df['phaseID'] = RveInfo.PHASENUM[phase]
                total_df = pd.concat([total_df, grains_df])
                phase_input_df['phaseID'] = RveInfo.PHASENUM[phase]
                input_data = pd.concat([input_data, phase_input_df])

            else:
                grains_df = phase_input_df.copy()
                grains_df['phaseID'] = RveInfo.PHASENUM[phase]
                total_df = pd.concat([total_df, grains_df])

        print('Processing now')

        grains_df = super().process_df(total_df, RveInfo.SHRINK_FACTOR)
        total_volume = sum(
            grains_df[grains_df['phaseID'] <= 6]['final_conti_volume'].values)  # Inclusions and bands dont influence filling
        estimated_boxsize = np.cbrt(total_volume)
        RveInfo.LOGGER.info(f"The total number of grains is {grains_df.__len__()}")
        RveInfo.LOGGER.info("the total volume of your dataframe is {}. A boxsize of {} is recommended.".
                            format(total_volume, estimated_boxsize))

        input_data.to_csv(RveInfo.gen_path + '/input_data.csv', index=False)

        return grains_df

    def rve_generation(self, total_df):

        """
        Separate the different datas
            grains_df = Grains which are used in RSA and Tesselation directly
            inclusions_df = data which is placed directly after the tesselation (not growing)
            bands_df = data used for the formation of bands
        """
        grains_df = total_df.loc[total_df['phaseID'] <= 4, :]
        grains_df = grains_df.sort_values(by='final_conti_volume', ascending=False)
        grains_df.reset_index(inplace=True, drop=True)
        grains_df.loc[:, 'GrainID'] = grains_df.index + 1

        if RveInfo.phase_ratio[RveInfo.PHASENUM['Inclusions']] > 0:
            inclusions_df = total_df[total_df['phaseID'] == 5]
            inclusions_df.sort_values(by='final_conti_volume', inplace=True, ascending=False)
            inclusions_df.reset_index(inplace=True, drop=True)
            inclusions_df['GrainID'] = inclusions_df.index

        if RveInfo.number_of_bands > 0:
            bands_df = total_df[total_df['phaseID'] == 6]
            bands_df = bands_df.sort_values(by='final_conti_volume', ascending=False)
            bands_df.reset_index(inplace=True, drop=True)
            bands_df.loc[:, 'GrainID'] = bands_df.index

        """
        BAND GENERATION HERE!
        """
        if RveInfo.number_of_bands > 0:
            box_size_y = RveInfo.box_size if RveInfo.box_size_y is None else RveInfo.box_size_y
            print(box_size_y)
            band_data = bands_df.copy()
            adjusted_size = np.cbrt((RveInfo.bandwidths[0] * RveInfo.box_size ** 2) * RveInfo.band_filling)
            bands_df = super().sample_input_3D(band_data, adjusted_size, constraint=RveInfo.bandwidths[0])
            bands_df.sort_values(by='final_conti_volume', inplace=True, ascending=False)
            bands_df.reset_index(inplace=True, drop=True)
            bands_df['GrainID'] = bands_df.index
            print(bands_df)
            discrete_RSA_obj = DiscreteRsa3D(bands_df['a'].tolist(),
                                             bands_df['b'].tolist(),
                                             bands_df['c'].tolist(),
                                             bands_df['alpha'].tolist())

            # Zum abspeichern der Werte
            band_list = list()

            # Berechne center and store the values:
            band_center_0 = int(RveInfo.bin_size + np.random.rand() * (box_size_y - RveInfo.bin_size))
            band_half_0 = float(RveInfo.bandwidths[0] / 2)
            band_list.append([band_half_0, band_center_0])

            # initialize empty grid_array for bands called band_array
            rsa = super().gen_array()
            band_rsa = super().gen_boundaries_3D(rsa)
            rsa_start = super().band_generator(band_array=band_rsa, bandwidth=RveInfo.bandwidths[0], center=band_center_0)

            # Place first band
            x_0_list = list()
            y_0_list = list()
            z_0_list = list()

            # band_list.append([band_half_0, band_center_0])
            rsa, x_0, y_0, z_0, rsa_status = discrete_RSA_obj.run_rsa_clustered(previous_rsa=rsa_start,
                                                                                band_array=rsa_start,
                                                                                animation=True)

            x_0_list.extend(x_0)
            y_0_list.extend(y_0)
            z_0_list.extend(z_0)

            # Place the Rest of the Bands
            for i in range(1, RveInfo.number_of_bands):
                print(i)
                print('RSA zu Beginn Band 2.')

                # ---------------------------------------------------------------------------------------------------
                # Sample grains for the second band
                adjusted_size = np.cbrt((RveInfo.bandwidths[i] * RveInfo.box_size ** 2) * RveInfo.band_filling)
                new_df = super().sample_input_3D(band_data, adjusted_size, constraint=RveInfo.bandwidths[0])
                new_df.sort_values(by='final_conti_volume', inplace=True, ascending=False)
                new_df.reset_index(inplace=True, drop=True)
                new_df['GrainID'] = new_df.index
                bands_df = pd.concat([bands_df, new_df])
                bands_df.reset_index(inplace=True, drop=True)
                bands_df['GrainID'] = bands_df.index
                # ---------------------------------------------------------------------------------------------------

                # Berechne neuen Center und prüfe überschneidung
                intersect = True
                while intersect == True:
                    band_center = int(RveInfo.bin_size + np.random.rand() * (box_size_y - RveInfo.bin_size))
                    band_half = float(RveInfo.bandwidths[0] / 2)
                    # Intersection when c_old - c_new < b_old + b_new (for each band)
                    for [bw_old, bc_old] in band_list:
                        bw_dist = bw_old + band_half
                        bc_dist = abs(bc_old - band_center)
                        if bc_dist <= bw_dist:
                            # one single intercept is enough for breaking the loop
                            intersect = True
                            break
                        else:
                            intersect = False

                # Get maximum value from previous RSA as starting pint
                startindex = int(np.amin(rsa) + 1000) * -1
                print(startindex)

                rsa = super().gen_array()
                band_rsa = super().gen_boundaries_3D(rsa)
                band_array_new = super().band_generator(band_array=band_rsa, bandwidth=RveInfo.bandwidths[0], center=band_center)

                discrete_RSA_obj = DiscreteRsa3D(new_df['a'].tolist(),
                                                 new_df['b'].tolist(),
                                                 new_df['c'].tolist(),
                                                 new_df['alpha'].tolist())

                rsa, x_0, y_0, z_0, rsa_status = discrete_RSA_obj.run_rsa_clustered(previous_rsa=rsa,
                                                                                    band_array=band_array_new,
                                                                                    animation=True,
                                                                                    startindex=startindex)

                x_0_list.extend(x_0)
                y_0_list.extend(y_0)
                z_0_list.extend(z_0)
        else:
            rsa_status = True

        """
        NORMAL RSA HERE
        """
        if rsa_status:
            discrete_RSA_obj = DiscreteRsa3D(grains_df['a'].tolist(),
                                             grains_df['b'].tolist(),
                                             grains_df['c'].tolist(),
                                             grains_df['alpha'].tolist())

            if RveInfo.number_of_bands > 0:
                rsa, x_0_list, y_0_list, z_0_list, rsa_status = discrete_RSA_obj.run_rsa(band_ratio_rsa=1,
                                                                                         banded_rsa_array=rsa,
                                                                                         x0_alt=x_0_list,
                                                                                         y0_alt=y_0_list,
                                                                                         z0_alt=z_0_list)

            else:
                rsa, x_0_list, y_0_list, z_0_list, rsa_status = discrete_RSA_obj.run_rsa()

        """
        TESSELATOR HERE
        """
        if rsa_status:
            if RveInfo.number_of_bands > 0:
                rsa = super().rearange_grain_ids_bands(bands_df=bands_df,
                                                       grains_df=grains_df,
                                                       rsa=rsa)

                # Concat all the data
                whole_df = pd.concat([grains_df, bands_df])
                whole_df.reset_index(inplace=True, drop=True)
                whole_df['GrainID'] = whole_df.index
                print(whole_df['GrainID'])

                whole_df['x_0'] = x_0_list
                whole_df['y_0'] = y_0_list
                whole_df['z_0'] = z_0_list
                discrete_tesselation_obj = Tesselation3D(whole_df)
                rve, rve_status = discrete_tesselation_obj.run_tesselation(rsa, band_idx_start=grains_df.__len__())
            else:
                whole_df = grains_df.copy()
                whole_df.reset_index(inplace=True, drop=True)
                whole_df['GrainID'] = whole_df.index
                whole_df['x_0'] = x_0_list
                whole_df['y_0'] = y_0_list
                whole_df['z_0'] = z_0_list
                discrete_tesselation_obj = Tesselation3D(whole_df)
                rve, rve_status = discrete_tesselation_obj.run_tesselation(rsa)

            # Change the band_ids to -200
            for i in range(len(grains_df), len(whole_df)+1):
                rve[np.where(rve == i + 1)] = -200

        else:
            RveInfo.LOGGER.info("The RSA did not succeed...")
            sys.exit()

        """
        PLACE THE INCLUSIONS!
        """
        if rve_status and RveInfo.phase_ratio[RveInfo.PHASENUM['Inclusions']] != 0:
            discrete_RSA_inc_obj = DiscreteRsa3D(inclusions_df['a'].tolist(),
                                                 inclusions_df['b'].tolist(),
                                                 inclusions_df['c'].tolist(),
                                                 inclusions_df['alpha'].tolist())

            rve, rve_status = discrete_RSA_inc_obj.run_rsa_inclusions(rve)
        elif not rve_status:
            print('Tesselator Failed!')

        """
        GENERATE INPUT DATA FOR SIMULATIONS HERE
        """
        if rve_status:
            # TODO: Hier gibt es einen relativ großen Mesh/Grid-Preprocessing Block --> Auslagern
            periodic_rve_df, periodic_rve = super().repair_periodicity_3D(rve)
            periodic_rve_df['phaseID'] = 0
            print('len rve edge:', np.cbrt(len(periodic_rve_df)))
            # An den NaN-Werten in dem DF liegt es nicht!

            grains_df.sort_values(by=['GrainID'])
            # debug_df = grains_df.copy()
            max_grain_id = int(periodic_rve.max())
            for i in range(max_grain_id):
                # Set grain-ID to number of the grain
                # Denn Grain-ID ist entweder >0 oder -200 oder >-200
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == i+1, 'phaseID'] = grains_df.loc[i, 'phaseID']

            if RveInfo.phase_ratio[RveInfo.PHASENUM['Inclusions']] > 0:
                # Set the points where < -200 to phase 5 and to grain ID i + j + 3
                for j in range(inclusions_df.__len__()):
                    periodic_rve_df.loc[periodic_rve_df['GrainID'] == -(200 + j + 1), 'GrainID'] = max_grain_id + j + 1
                    periodic_rve_df.loc[periodic_rve_df['GrainID'] == (max_grain_id + j + 1), 'phaseID'] = 5
                    periodic_rve[np.where(periodic_rve == -(200 + j + 1))] = max_grain_id + j + 1
                max_grain_id = periodic_rve.max()
                grains_df = pd.concat([grains_df, inclusions_df])
                grains_df.reset_index(inplace=True, drop=True)
                grains_df.loc[grains_df['phaseID'] == 5, 'GrainID'] = grains_df.loc[grains_df['phaseID'] == 5].index + 1

            if RveInfo.number_of_bands > 0 and RveInfo.phase_ratio[RveInfo.PHASENUM['Inclusions']] > 0:
                # Set the points where == -200 to phase 2 and to grain ID i + j + 3
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == -200, 'GrainID'] = max_grain_id + 1
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == (max_grain_id + 3), 'phaseID'] = 2
                periodic_rve[np.where(periodic_rve == -200)] = max_grain_id + 1
            else:
                # Set the points where == -200 to phase 2 and to grain ID i + 2
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == -200, 'GrainID'] = max_grain_id + 1
                periodic_rve_df.loc[periodic_rve_df['GrainID'] == (i + 2), 'phaseID'] = 2
                periodic_rve[np.where(periodic_rve == -200)] = max_grain_id + 1

            # Start the Mesher
            # grains_df.to_csv('grains_df.csv', index=False)
            # periodic_rve_df.to_csv('periodic_rve_df.csv', index=False)
            rve_shape = periodic_rve.shape
            print("rve shape is", rve_shape)
            # Write out Volumes
            grains_df = super().get_final_disc_vol_3D(grains_df, periodic_rve)
            print('grain_df keys:')
            grains_df.to_csv(RveInfo.store_path + '/Generation_Data/grain_data_output.csv', index=False)
            print('########################')
            print('Meshing starts')
            if RveInfo.damask_flag:
                # Startpoint: Rearrange the negative ID's
                last_grain_id = periodic_rve.max()  # BEWARE: For the .vti file, the grid must start at ZERO
                print('The last grain ID is:', last_grain_id)
                print('The number of bands is:', RveInfo.number_of_bands)

                if RveInfo.number_of_bands >= 1:
                    print('Nur Bänder')
                    phase_list = grains_df['phaseID'].tolist()
                    #periodic_rve[np.where(periodic_rve == -200)] = last_grain_id + 1
                    phase_list.append(2)

                elif RveInfo.phase_ratio[RveInfo.PHASENUM['Inclusions']] > 0:
                    print('Nur Inclusions')
                    phase_list = grains_df['phaseID'].tolist()
                    """for i in range(len(inclusions_df)):
                        #periodic_rve[np.where(periodic_rve == -(200 + i + 1))] = last_grain_id + i + 1
                        phase_list.append(5)"""
                    print(phase_list.__len__())

                else:
                    print('Keine Bänder, nur grains')
                    phase_list = grains_df['phaseID'].tolist()
                print(grains_df['phi1'])
                spectral.write_material(store_path=RveInfo.store_path, grains=phase_list, angles=grains_df[['phi1', 'PHI', 'phi2']])
                spectral.write_load(RveInfo.store_path)
                spectral.write_grid(store_path=RveInfo.store_path,
                                    rve=rve,
                                    spacing=RveInfo.box_size / 1000)

            if RveInfo.moose_flag:
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print(np.unique(grains_df['phaseID'].values))
                print(np.unique(periodic_rve_df['phaseID'].values))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                MooseMesher(rve_shape=rve_shape, rve=periodic_rve_df, grains_df=grains_df).run()
                # store phases and texture in seperate txt files to make it work within moose
                grains_df[['phi1', 'PHI', 'phi2']].to_csv(path_or_buf=RveInfo.store_path+'/EulerAngles.txt',
                                                          header=False, index=False)
                phases = periodic_rve_df.groupby(['GrainID']).mean()['phaseID']
                phases.to_csv(path_or_buf=RveInfo.store_path+'/phases.txt',  header=False, index=False)

            if RveInfo.abaqus_flag:
                mesher_obj = None
                if RveInfo.subs_flag:
                    print("substructure generation is turned on...")
                    # returns rve df containing substructures
                    # print("phase id is ,", grains_df.iloc[0]["phaseID"])
                    subs_rve = substrucRun().run(rve_df=periodic_rve_df, grains_df=grains_df)
                    # try:
                    #     subs_rve = substrucRun().run(rve_df=periodic_rve_df, grains_df=grains_df)
                    # except Exception as e:
                    #     print(e)
                    mesher_obj = SubMesher(rve_shape=rve_shape, rve=subs_rve, subs_df=grains_df)

                elif RveInfo.subs_flag == False:
                    print('######subsflag false######')
                    print("substructure generation is turned off...")
                    print(max(periodic_rve_df.GrainID))
                    print(np.unique(periodic_rve_df['GrainID'].values))
                    print(grains_df['GrainID'].values)
                    print(len(grains_df))

                    mesher_obj = AbaqusMesher(rve_shape=rve_shape, rve=periodic_rve_df, grains_df=grains_df)
                if mesher_obj:
                    mesher_obj.run()
        else:
            print('Tessellation did not succeed')
        return periodic_rve

    def post_processing(self, rve):


        start1 = int(rve.shape[0] / 4)
        stop1 = int(rve.shape[0] / 4 + rve.shape[0] / 4 * 2)
        start2 = int(rve.shape[1] / 4)
        stop2 = int(rve.shape[1] / 4 + rve.shape[1] / 4 * 2)

        if RveInfo.dimension == 3:
            start3 = int(rve.shape[2] / 4)
            stop3 = int(rve.shape[2] / 4 + rve.shape[2] / 4 * 2)
            rve = rve[start1:stop1, start2:stop2, start3:stop3]
        else:
            rve = rve[start1:stop1, start2:stop2]

        phase_ratios = list()
        ref_r_in = dict()
        ref_r_out = dict()
        grain_shapes_in = shape().get_input_ellipses()
        for phase in RveInfo.phases:
            phase_id = RveInfo.PHASENUM[phase]
            slice_ID = 0
            if phase_id < 5:
                # generate pair plots for shape comparison for each phase
                grain_shapes = pd.DataFrame()
                for i in range(math.floor(rve.shape[2] / 4)):
                    print('phase_id, i, slice_ID', phase_id, i, slice_ID)
                    grain_shapes_slice = shape().get_ellipses(rve, slice_ID, phase_id)
                    slice_ID += 4
                    if len(grain_shapes_slice) == 0:
                        RveInfo.RESULT_LOG.info(f'No {phase} found in slice {slice_ID}. Slice was neglected for {phase}')
                        continue
                    grain_shapes = pd.concat([grain_shapes, grain_shapes_slice])

                grain_shapes['inout'] = 'out'
                grain_shapes = grain_shapes.rename(columns={"AR": "AR (-)", "slope": "slope (°)"})
                grain_shapes_in_thisPhase = grain_shapes_in.loc[grain_shapes_in['phaseID'] == phase_id, ['AR', 'slope', 'inout']]

                grain_shapes_in_thisPhase = grain_shapes_in_thisPhase.sample(n=grain_shapes.__len__())
                grain_shapes_in_thisPhase = grain_shapes_in_thisPhase.rename(columns={"AR": "AR (-)", "slope": "slope (°)"})

                grain_shapes = pd.concat([grain_shapes, grain_shapes_in_thisPhase])
                grain_shapes = grain_shapes.sort_values(by=['inout'])
                grain_shapes.reset_index(inplace=True, drop=True)

                plot_kws = {"s": 2}

                sns.set_palette(sns.color_palette(RveInfo.rwth_colors))
                sns.set_context(rc={"font.size": 16, "axes.labelsize":20})
                g = sns.pairplot(data=grain_shapes, hue='inout', plot_kws=plot_kws)
                grain_shapes.to_csv('{}/Postprocessing/shape_control_{}.csv'.format(RveInfo.store_path, phase))
                g.fig.subplots_adjust(top=.9)
                g.fig.suptitle(phase)

                plt.savefig('{}/Postprocessing/shape_control_{}.png'.format(RveInfo.store_path, phase))
                plt.close()

                current_phase_ref_r_in, current_phase_ratio_out, current_phase_ref_r_out = \
                    PostProcVol().gen_in_out_lists(phaseID=phase_id)
                phase_ratios.append(current_phase_ratio_out)
                ref_r_in[phase] = current_phase_ref_r_in
                ref_r_out[phase] = current_phase_ref_r_out

            if phase_id == 5:
                current_phase_ref_r_in, current_phase_ratio_out, current_phase_ref_r_out = \
                    PostProcVol().gen_in_out_lists(phaseID=phase_id)
                phase_ratios.append(current_phase_ratio_out)

        if len(RveInfo.phases) > 1:
            input_ratio = list()
            labels = list()
            for i, phase in enumerate(RveInfo.phases):
                if RveInfo.PHASENUM[phase] > 5:  # phase ratio postprocessing for bands not relevant
                    continue
                ratio = RveInfo.phase_ratio[RveInfo.PHASENUM[phase]]
                input_ratio.append(ratio)
                label = RveInfo.phases[i]
                labels.append(label)
            PostProcVol().gen_pie_chart_phases(input_ratio, labels, 'input')
            PostProcVol().gen_pie_chart_phases(phase_ratios, labels, 'output')

        for phase in RveInfo.phases:
            if RveInfo.PHASENUM[phase] > 4: # postprocessing for inclusions and bands not yet supported
                continue
            PostProcVol().gen_plots(ref_r_in[phase], ref_r_out[phase], phase)
            if RveInfo.gui_flag:
                RveInfo.infobox_obj.emit('checkout the evaluation report of the rve stored at:\n'
                                         '{}/Postprocessing'.format(RveInfo.store_path))

        if RveInfo.subs_flag:
            substrucRun().post_processing(k=3)
        RveInfo.LOGGER.info("RVE generation process has successfully completed...")
