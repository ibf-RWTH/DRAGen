# TODO: Cleanup and correct banding errors

import os
import sys
import datetime
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import logging.handlers
import pandas as pd

from dragen.generation.DiscreteRsa3D import DiscreteRsa3D
from dragen.generation.DiscreteTesselation3D import Tesselation3D
from dragen.utilities.RVE_Utils import RVEUtils
from dragen.generation.mesher import Mesher
from dragen.postprocessing.voldistribution import PostProcVol
from dragen.generation import spectral

from InputGenerator.C_WGAN_GP import WGANCGP
from InputGenerator.linking import Reconstructor


class DataTask3D_GAN(RVEUtils):

    def __init__(self, box_size=30, n_pts=50, number_of_bands=0, bandwidth=3, shrink_factor=0.5, band_filling=0.99,
                 phase_ratio=0.95, inclusions_ratio=0.01, solver='Spectral',
                 file1=None, file2=None, store_path=None, gui_flag=False, anim_flag=False, gan_flag=False,
                 exe_flag=False, inclusions_flag=True):
        self.logger = logging.getLogger("RVE-Gen")
        self.box_size = box_size
        self.n_pts = n_pts  # has to be even
        self.bin_size = self.box_size / self.n_pts
        self.step_half = self.bin_size / 2
        self.shrink_factor = np.cbrt(shrink_factor)
        self.gui_flag = gui_flag
        self.gan_flag = gan_flag
        self.solver = solver
        self.inclusions_flag = inclusions_flag if self.gan_flag else False  # geht nicht ohne GAN
        self.root_dir = './'

        """
        Aktueller Parametersatz für GAN:
        Parameters:  
        """
        self.number_of_bands = number_of_bands
        if self.number_of_bands == 0:
            self.bandwidth_sum = 0
        else:
            self.bandwidth_sum = np.sum(np.asarray(bandwidth))  # To call sum() on list/tuple whatever
            print('Sum BW: ', self.bandwidth_sum)
            self.bandwidth = np.asarray(bandwidth)
            print('Bandbreiten-Array: ', self.bandwidth)
        self.band_filling = band_filling  # Percentage of Filling for the banded structure
        self.phase_ratio = phase_ratio  # 1 means all ferrite, 0 means all Martensite
        self.inclusion_ratio = inclusions_ratio  # Space occupied by inclusions
        if exe_flag:
            self.root_dir = store_path
        if not gui_flag and not self.gan_flag:
            self.root_dir = sys.argv[0][:-14]  # setting root_dir to root_dir by checking path of current file
            print(self.root_dir)
        elif gui_flag and not exe_flag:
            self.root_dir = store_path
        elif self.gan_flag:
            self.root_dir = store_path

        self.logger.info('the exe_flag is: ' + str(exe_flag))
        self.logger.info('root was set to: ' + self.root_dir)
        self.animation = anim_flag
        self.file1 = file1
        self.file2 = file2
        self.utils_obj = RVEUtils(self.box_size, self.n_pts, self.bandwidth)
        print(self.bandwidth)
        if self.gan_flag:
            print('Erzeuge GAN')
            self.GAN = self.run_cwgangp()

        self.x_grid, self.y_grid, self.z_grid = super().gen_grid()
        super().__init__(box_size, n_pts, self.x_grid, self.y_grid, self.z_grid, bandwidth, debug=False)

    def write_specs(self, storepath):
        """
        Write a data-File containing all the informations necessary to create the rve to store_path
        """
        with open(storepath + '/' + 'Specs.txt', 'w') as specs:
            # Network specs
            specs.writelines('-----------------------------------------------------------\n')
            specs.writelines('---------------------RVE-Specifications--------------------\n')
            specs.writelines('-----------------------------------------------------------\n\n\n')
            today = datetime.date.today()
            d1 = today.strftime("%d/%m/%Y")
            specs.writelines('Created at: ' + d1)
            specs.writelines('\n\n')
            specs.writelines('Size of the RVE {}µm - {}\n'.format(self.box_size, self.n_pts))
            specs.writelines('Solver Typ: {}\n\n'.format(self.solver))
            specs.writelines('Number of Bands: {}\n'.format(float(self.number_of_bands)))
            for i in range(int(self.number_of_bands)):
                specs.writelines('bandwidth: {}µm\n'.format(float(self.bandwidth[i])))
            specs.writelines('Phase ratios:\n \tOverall-percentage: {}%\n \tBand-Percentage: {}%\n '
                             '\tIsland-Percentage: {}%\n'.
                             format(100 * self.phase_ratio, 100 * float(self.percentage_in_bands),
                                    100 * float(self.new_phase_ratio)))
            if self.inclusions_flag:
                specs.writelines('Inclusion-percentage: {}\n'.format(self.inclusion_ratio))
            else:
                specs.writelines('No inclusions in the RVE! \n\n')

            specs.writelines('For informations, contributing etc. please contact the DRAGEN-Team:\n')
            specs.writelines('\tDRAGen@iehk.rwth-aachen.de\n\n')

    def run_cwgangp(self):
        SOURCE = self.root_dir + '/ExampleInput'

        # Data:
        df1 = pd.read_csv(SOURCE + '/Input_TDxBN_AR.csv')
        df2 = pd.read_csv(SOURCE + '/Input_RDxBN_AR.csv')
        df3 = pd.read_csv(SOURCE + '/Input_RDxTD_AR.csv')

        # Inclusions
        df4 = pd.read_csv(SOURCE + '/Einschlüsse_TDxBN_AR.csv')
        df5 = pd.read_csv(SOURCE + '/Einschlüsse_RDxBN_AR.csv')
        df6 = pd.read_csv(SOURCE + '/Einschlüsse_RDxTD_AR.csv')

        df7 = pd.read_csv(SOURCE + '/Input_Martensit_BNxRD.csv')
        # Set up CWGAN-GP with all data
        store_path = self.root_dir + '/OutputData/' + str(datetime.datetime.now())[:10] + '_' + str(0)
        if not os.path.isdir(store_path):
            os.makedirs(store_path)
        GAN = WGANCGP(df_list=[df1, df2, df3, df4, df5, df6, df7], storepath=store_path, num_features=3, gen_iters=100)

        # Run training for 5000 Its - 150.000 is default
        # GAN.train()

        # Evaluate Afterwards
        # GAN.evaluate()
        return GAN

    def sample_gan_input(self, adjusted_size, labels=(), size=1000, maximum=None):
        TDxBN = self.GAN.sample_batch(label=labels[0], size=size * 2)
        RDxBN = self.GAN.sample_batch(label=labels[1], size=size * 2)
        RDxTD = self.GAN.sample_batch(label=labels[2], size=size * 2)

        # Run the Reconstruction
        Bot = Reconstructor(TDxBN, RDxBN, RDxTD, drop=True)
        Bot.run(n_points=size)  # Could take a while with more than 500 points...

        # Calculate the Boxsize based on Bands and original size
        adjusted_size = adjusted_size
        print('Adjusted Size:', adjusted_size)
        Bot.get_rve_input(bs=adjusted_size, maximum=maximum)
        return Bot.rve_inp  # This is the RVE-Input data

    def sample_gan_input_2d(self, label, boxsize, size=1000, maximum=None):
        df = self.GAN.sample_batch(label=label, size=size * 2)

        # 1.) Switch the axis
        # locations: [Area, Aspect Ratio, Slope] - Sind so fixed
        df2 = df.copy().dropna(axis=0)
        columns = df2.columns
        df2['Axes1'] = np.nan
        df2['Axes2'] = np.nan
        df2['Axes1'] = (df2[columns[0]] * df2[columns[1]] / np.pi) ** 0.5  # Major
        df2['Axes2'] = df2[columns[0]] / (np.pi * df2['Axes1'])
        # Switch axis in 45 - 135
        for j, row in df2.iterrows():
            if 45 < row[2] <= 135:
                temp1 = df2['Axes2'].iloc[j]
                temp2 = df2['Axes1'].iloc[j]
                df2['Axes1'].iloc[j] = temp1
                df2['Axes2'].iloc[j] = temp2
            else:
                pass
        # Set c = b due to coming rotation
        df2['Axes3'] = df2['Axes2']
        sns.kdeplot(data=df2, x='Axes1')
        sns.kdeplot(data=df2, x='Axes2')
        sns.kdeplot(data=df2, x='Axes3')
        plt.legend(['Ax1', 'Ax2', 'Ax3'])
        plt.savefig(self.store_path + '/MartensiteSize.png')
        plt.close()

        sns.pairplot(df2)
        plt.savefig(self.store_path + '/Pairplot.png')

        # 2.) Sample the right amount of Input
        bs = boxsize
        max_volume = bs * bs * bs
        grain_vol = 0
        data = df2.copy()
        inp_list = list()
        counter = 0
        loop_counter = 0
        while grain_vol < max_volume:
            idx = np.random.randint(0, data.__len__())
            grain = data[['Axes1', 'Axes2', 'Axes3', 'Slope']].iloc[idx].tolist()
            data = data.drop(labels=data.index[idx], axis=0)
            if maximum is None:
                vol = 4 / 3 * np.pi * grain[0] * grain[1] * grain[2]
                grain_vol += vol
                inp_list.append([grain[0], grain[1], grain[2], grain[3], vol])
            else:
                if (grain[0] > maximum) or (grain[0] > maximum*3) or (grain[2] > maximum*3):
                    pass
                else:
                    # Only append if smaller
                    vol = 4 / 3 * np.pi * grain[0] * grain[1] * grain[2]
                    grain_vol += vol
                    inp_list.append([grain[0], grain[1], grain[2], grain[3], vol])

            if data.__len__() <= 10:
                # Reset the data
                data = df2.copy()
                counter += 1
                with open(self.store_path + '/rve.sta', 'a') as sta:
                    sta.writelines('\nReseting the database for the {}th time\n'.format(counter))
                    if counter >= 3:
                        sta.writelines('Considering increasing the bandwidth\n')
            loop_counter += 1

        with open(self.store_path + '/rve.sta', 'a') as sta:
            sta.writelines('\nNeeded {}-Loops to sample points for the bands!\n'.format(loop_counter))

        # Del last if to big and more than one value:
        if grain_vol > 1.5 * max_volume and inp_list.__len__() > 0:
            # Pop
            idx = np.random.randint(0, inp_list.__len__())
            inp_list.pop(idx)

        header = ['a', 'b', 'c', 'alpha', 'volume']
        df3 = pd.DataFrame(inp_list, columns=header)

        # Add temporary euler angles
        df3['phi1'] = (np.random.rand(df3.__len__()) * 360)
        df3['PHI'] = (np.random.rand(df3.__len__()) * 180)
        df3['phi2'] = (np.random.rand(df3.__len__()) * 360)

        return df3

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
        grains_df = None
        self.setup_logging()
        self.store_path = self.root_dir + '/OutputData/' + str(datetime.datetime.now())[:10] + '_' + str(epoch)
        self.fig_path = self.store_path + '/Figs'
        self.gen_path = self.store_path + '/Generation_Data'

        if not os.path.isdir(self.store_path):
            os.makedirs(self.store_path)
        if not os.path.isdir(self.fig_path):
            os.makedirs(self.fig_path)  # Second if needed
        if not os.path.isdir(self.gen_path):
            os.makedirs(self.gen_path)  # Second if needed

        return grains_df, self.store_path

    def rve_generation(self, grains_df, store_path):

        with open(store_path + '/rve.sta', 'a') as sta:
            sta.writelines('------------------------------------------------------------------------------\n')
            sta.writelines('----------------------------RVE Generation started----------------------------\n')
            sta.writelines('------------------------------------------------------------------------------\n')
            sta.writelines('\n\n')
            now = datetime.datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            sta.writelines('Starttime: ' + dt_string)
            sta.writelines('\n\n\n')
            sta.writelines('Calculating the phase ratio...\n')

        print('Bandberechnung')
        percentage_in_bands = (self.bandwidth_sum * self.box_size ** 2 * self.band_filling) / \
                              (self.box_size ** 3)
        print(percentage_in_bands)

        self.percentage_in_bands = percentage_in_bands
        new_phase_ratio = self.phase_ratio - percentage_in_bands
        self.new_phase_ratio = new_phase_ratio
        self.write_specs(self.store_path)
        if new_phase_ratio < 0:
            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines('The bands are containing more martensite then specfified for the overall volume! \n')
                sta.writelines('Setting the phase ratio for the rest of the volume to 1 \n')
            new_phase_ratio = 0
        else:
            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines('The phase ratio for the rest of the volume is: {} \n'.format(new_phase_ratio))
        """
        -------------------------------------------------------------------------------
        BAND-PHASE GENERATION HERE
        phaseID == 2 (Martensite)
        adjusted_size is: n_bands * bw * bs**2
        flag before the field
        -------------------------------------------------------------------------------
        """
        # FIXME: Sobald es miehr als ein Band gibt, gibt es ziemliche Probleme, irgendwas stimmt da nicht
        # FIXME: Wenn die unterschiedlich Dick sind, wird das auch nicht wirklich funktionieren
        """
        
        """
        if self.number_of_bands > 0:
            starttime = time.time()
            # FIXME: Das muss noch runter, samplen PRO band
            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines("\n\nRVE generation process has started with Band-creation \n")
                sta.writelines(
                    "The total volume of the RVE is {0}*{0}*{0} = {1} \n".format(self.box_size, self.box_size ** 3))
                adjusted_size = np.cbrt((self.bandwidth[0] * self.box_size ** 2) * self.band_filling)
                phase1 = self.sample_gan_input_2d(size=10000, label=6, boxsize=adjusted_size,
                                                  maximum=self.bandwidth[0] * 1.5)
                bands_df = super().process_df(phase1, float(self.shrink_factor))
                bands_df['phaseID'] = 2
                sta.writelines('Sampled {} Martensite-Points for band-Creation! \n'.format(bands_df.__len__()))
                sta.writelines('The total conti volume is: {} \n'.format(np.sum(bands_df['final_conti_volume'].values)))
                sta.writelines(
                    'The total discrete volume is: {} \n'.format(np.sum(bands_df['final_discrete_volume'].values)))

            discrete_RSA_obj = DiscreteRsa3D(self.box_size, self.n_pts,
                                             bands_df['a'].tolist(),
                                             bands_df['b'].tolist(),
                                             bands_df['c'].tolist(),
                                             bands_df['alpha'].tolist(), store_path=store_path)

            # initialize empty grid_array for bands called band_array
            xyz = np.linspace(-self.box_size / 2, self.box_size + self.box_size / 2, 2 * self.n_pts, endpoint=True)
            x_grid, y_grid, z_grid = np.meshgrid(xyz, xyz, xyz)

            utils_obj_band = RVEUtils(self.box_size, self.n_pts,
                                      x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, bandwidth=self.bandwidth)
            band_array = np.zeros((2 * self.n_pts, 2 * self.n_pts, 2 * self.n_pts))
            band_array = utils_obj_band.gen_boundaries_3D(band_array)

            # Place first band
            x_0_list = list()
            y_0_list = list()
            z_0_list = list()

            # Zum abspeichern der Werte
            band_list = list()

            # Berechne center and store the values:
            band_center_0 = int(self.bin_size + np.random.rand() * (self.box_size - self.bin_size))
            band_half_0 = float(self.bandwidth[0] / 2)
            band_list.append([band_half_0, band_center_0])

            rsa_start = utils_obj_band.band_generator(store_path=self.store_path, band_array=band_array,
                                                      bandwidth=self.bandwidth[0], center=band_center_0)
            rsa, x_0, y_0, z_0, rsa_status = discrete_RSA_obj.run_rsa_clustered(
                previous_rsa=rsa_start,
                band_array=rsa_start,
                animation=True)

            x_0_list.extend(x_0)
            y_0_list.extend(y_0)
            z_0_list.extend(z_0)

            # Place the Rest of the Bands
            for i in range(1, self.number_of_bands):
                print(i)
                print('RSA zu Beginn Band 2.')

                # ---------------------------------------------------------------------------------------------------
                # Sample grains for the second band
                with open(store_path + '/rve.sta', 'a') as sta:
                    sta.writelines('\n\n Band number {}\n'.format(i+1))
                    adjusted_size = np.cbrt((self.bandwidth[i] * self.box_size ** 2) * self.band_filling)
                    phase1 = self.sample_gan_input_2d(size=10000, label=6, boxsize=adjusted_size,
                                                      maximum=self.bandwidth[i] * 1.5)
                    new_df = super().process_df(phase1, float(self.shrink_factor))
                    new_df['phaseID'] = 2
                    sta.writelines('Sampled {} Martensite-Points for band-Creation! \n'.format(new_df.__len__()))
                    sta.writelines('The total conti volume is: {} \n'.format(np.sum(new_df['final_conti_volume'].values)))
                    sta.writelines(
                        'The total discrete volume is: {} \n'.format(np.sum(new_df['final_discrete_volume'].values)))
                    bands_df = pd.concat([bands_df, new_df])
                    bands_df.reset_index(inplace=True, drop=True)
                    bands_df['GrainID'] = bands_df.index
                # ---------------------------------------------------------------------------------------------------

                # Fixme: Band-Array_new hat immer noch beide Bänder - CenterBerechnung rausziehen
                band_array = np.zeros((2 * self.n_pts, 2 * self.n_pts, 2 * self.n_pts))
                band_array = utils_obj_band.gen_boundaries_3D(band_array)

                # Berechne neuen Center und prüfe überschneidung
                intersect = True
                while intersect == True:
                    band_center = int(self.bin_size + np.random.rand() * (self.box_size - self.bin_size))
                    band_half = self.bandwidth[i] / 2
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


                band_array_new = utils_obj_band.band_generator(store_path=self.store_path, band_array=band_array,
                                                               bandwidth=self.bandwidth[i], center=band_center)
                discrete_RSA_obj = DiscreteRsa3D(self.box_size, self.n_pts,
                                                 new_df['a'].tolist(),
                                                 new_df['b'].tolist(),
                                                 new_df['c'].tolist(),
                                                 new_df['alpha'].tolist(), store_path=store_path)

                # Get maximum value from previous RSA as starting pint
                startindex = int(np.amin(rsa)+1000)*-1
                print(startindex)
                rsa, x_0, y_0, z_0, rsa_status = discrete_RSA_obj.run_rsa_clustered(
                    previous_rsa=rsa,
                    band_array=band_array_new,
                    animation=True, startindex=startindex)  # Maximum value of previous is startpoint of new

                x_0_list.extend(x_0)
                y_0_list.extend(y_0)
                z_0_list.extend(z_0)

            endtime = datetime.timedelta(seconds=(time.time() - starttime))

            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines('Elapsed time for band creation: {}\n\n'.format(endtime))
        else:
            rsa_status = True

        """
        ----------------------------------------------------------------------------------
        Placing the rest of the grains!
        Ferrite and Martensit Island
        adjusted_size is: bs**3 - n_bands * bw * bs**2 (Remaining Space)
        phases are 1 and 2, therefore sample two dfs
        Ratio is based on Martensite ratio in the steel grade
        ----------------------------------------------------------------------------------
        """
        if rsa_status:
            starttime = time.time()
            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines("RVE generation process continues with placing of the ferrite grains and martensite "
                               "islands \n")
                # Ferrite:
                adjusted_size_ferrite = np.cbrt(
                    (self.box_size ** 3 - self.band_filling * self.bandwidth_sum * self.box_size ** 2) * (
                                1 - new_phase_ratio))
                phase1 = self.sample_gan_input(size=800, labels=[0, 1, 2], adjusted_size=adjusted_size_ferrite)
                grains_df = super().process_df(phase1, float(self.shrink_factor))
                grains_df['phaseID'] = 1  # Ferrite_Grains
                sta.writelines('Sampled {} Ferrite-Points for the matrix! \n'.format(grains_df.__len__()))
                sta.writelines(
                    'The total conti volume is: {} \n'.format(np.sum(grains_df['final_conti_volume'].values)))
                sta.writelines(
                    'The total discrete volume is: {} \n'.format(np.sum(grains_df['final_discrete_volume'].values)))
                # Martensite
                adjusted_size_martensite = np.cbrt(
                    (
                                self.box_size ** 3 - self.band_filling * self.bandwidth_sum * self.box_size ** 2) * new_phase_ratio)
                phase2 = self.sample_gan_input_2d(size=800, label=6, boxsize=adjusted_size_martensite,
                                                  maximum=self.box_size / 5)
                grains_df_2 = super().process_df(phase2, float(self.shrink_factor))
                grains_df_2['phaseID'] = 2  # Martensite_Islands
                sta.writelines('Sampled {} Martensite-Islands for the matrix! \n'.format(grains_df_2.__len__()))
                sta.writelines(
                    'The total conti volume is: {} \n'.format(np.sum(grains_df_2['final_conti_volume'].values)))
                sta.writelines(
                    'The total discrete volume is: {} \n'.format(np.sum(grains_df_2['final_discrete_volume'].values)))

                # Sort again because of Concat
                grains_df = pd.concat([grains_df, grains_df_2])
                grains_df.sort_values(by='final_conti_volume', inplace=True, ascending=False)
                grains_df.reset_index(inplace=True, drop=True)
                grains_df['GrainID'] = grains_df.index

            discrete_RSA_obj = DiscreteRsa3D(self.box_size, self.n_pts,
                                             grains_df['a'].tolist(),
                                             grains_df['b'].tolist(),
                                             grains_df['c'].tolist(),
                                             grains_df['alpha'].tolist(), store_path=store_path)

            # Run the 'rest' of the rsa:
            if self.number_of_bands > 0:
                rsa, x_0_list, y_0_list, z_0_list, rsa_status = discrete_RSA_obj.run_rsa(band_ratio_rsa=1,
                                                                                         banded_rsa_array=rsa,
                                                                                         animation=self.animation,
                                                                                         x0_alt=x_0_list,
                                                                                         y0_alt=y_0_list,
                                                                                         z0_alt=z_0_list,
                                                                                         gui=False)
            else:
                rsa, x_0_list, y_0_list, z_0_list, rsa_status = discrete_RSA_obj.run_rsa(band_ratio_rsa=1,
                                                                                         banded_rsa_array=None,
                                                                                         animation=self.animation,
                                                                                         gui=False)
            endtime = datetime.timedelta(seconds=(time.time() - starttime))
            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines('\nElapsed time for normal RSA: {}\n\n'.format(endtime))
        else:
            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines("The normal rsa did not succeed...Please check input and logging for Errors \n")
            sys.exit()

        """
        ---------------------------------------------------------------------
        RUN THE TESSELATION
        Tesselation is based on both of the RSA's, the normal and the clustered one. To run the tesselation, the band 
        and the "normal"-df's have to be concated to run a sufficient generation
        ---------------------------------------------------------------------
        """
        if rsa_status:
            starttime = time.time()
            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines("\n\nRVE generation process continues with the tesselation of grains!\n")
            # Passe Grain-IDs an
            if self.number_of_bands > 0:
                rsa = self.utils_obj.rearange_grain_ids_bands(bands_df=bands_df,
                                                              grains_df=grains_df,
                                                              rsa=rsa)

                # Concat all the data
                whole_df = pd.concat([grains_df, bands_df])
                whole_df.reset_index(inplace=True, drop=True)
                whole_df['GrainID'] = whole_df.index
            else:
                whole_df = grains_df.copy()
                whole_df.reset_index(inplace=True, drop=True)
                whole_df['GrainID'] = whole_df.index

            whole_df['x_0'] = x_0_list
            whole_df['y_0'] = y_0_list
            whole_df['z_0'] = z_0_list
            grains_df.to_csv(self.gen_path + '/grain_data_input.csv', index=False)
            discrete_tesselation_obj = Tesselation3D(box_size=self.box_size, n_pts=self.n_pts,
                                                     grains_df=whole_df, shrinkfactor=self.shrink_factor, band_ratio=1,
                                                     store_path=store_path)
            if self.number_of_bands > 0:
                rve, rve_status = discrete_tesselation_obj.run_tesselation(rsa, animation=self.animation, gui=False,
                                                                           band_idx_start=grains_df.__len__(),
                                                                           grain_df=grains_df)
            else:
                rve, rve_status = discrete_tesselation_obj.run_tesselation(rsa, animation=self.animation, gui=False,
                                                                           grain_df=grains_df)
            # Change the band_ids to -200
            for i in range(len(grains_df), len(whole_df)):
                rve[np.where(rve == i + 1)] = -200

            endtime = datetime.timedelta(seconds=(time.time() - starttime))
            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines('Elapsed time for Tesselation: {}\n\n'.format(endtime))
        else:
            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines("The tesselation did not succeed...Check input and logging for Errors \n")
            sys.exit()

        """
        -------------------------------------------------------------------------
        PLACE THE INCLUSIONS!
        The inclusions will not grow and are therefore placed in the rve AFTER the
        Tesselation! The inclusions will placed inside another grain and cannot cut a 
        grain boundary. Labels are lower -200
        -------------------------------------------------------------------------
        """
        if rve_status and self.inclusions_flag:
            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines("\n\nRVE generation process reaches final steps: Placing inclusions in the matrix\n")
                adjusted_size = self.box_size * np.cbrt(self.inclusion_ratio)  # 1% as an example
                inclusions = self.sample_gan_input(size=200, labels=(3, 4, 5), adjusted_size=adjusted_size)
                inclusions_df = super().process_df(inclusions, float(self.shrink_factor))
                inclusions_df['phaseID'] = 2  # Inclusions also elastic
                sta.writelines('Sampled {} inclusions for the RVE\n'.format(inclusions_df.__len__()))
                sta.writelines(
                    'The total conti volume is: {}\n'.format(np.sum(inclusions_df['final_conti_volume'].values)))
                sta.writelines(
                    'The total discrete volume is: {}\n'.format(np.sum(inclusions_df['final_discrete_volume'].values)))

            discrete_RSA_inc_obj = DiscreteRsa3D(self.box_size, self.n_pts,
                                                 inclusions_df['a'].tolist(),
                                                 inclusions_df['b'].tolist(),
                                                 inclusions_df['c'].tolist(),
                                                 inclusions_df['alpha'].tolist(), store_path=store_path)

            rve, rve_status = discrete_RSA_inc_obj.run_rsa_inclusions(rve, animation=True)
            # print(np.asarray(np.unique(rve, return_counts=True)).T)
            # print(grains_df)

        """
        -------------------------------------------------------------------------
        CREATE INPUT FOR NUMERICAL SOLVER!
        There are two options available at the moment, Abaqus/implicit and 
        DAMASK_Spectral.
        Control via solver == 'FEM' or 'Spectral'
        -------------------------------------------------------------------------
        """
        if self.solver == 'Spectral':
            with open(store_path + '/rve.sta', 'a') as sta:
                sta.writelines("\n\nRVE generation process nearly complete: Creating input for DAMASK Spectral now: \n")

            # 1.) Write out Volume
            grains_df.sort_values(by=['GrainID'], inplace=True)
            disc_vols = np.zeros((1, grains_df.shape[0])).flatten().tolist()
            for i in range(len(grains_df)):
                # grainID = grains_df.GrainID[i]
                disc_vols[i] = np.count_nonzero(rve == i + 1) * self.bin_size ** 3

            grains_df['meshed_conti_volume'] = disc_vols
            grains_df.to_csv(self.store_path + '/Generation_Data/grain_data_output_conti.csv', index=False)

            # Startpoint: Rearange the negative ID's
            last_grain_id = rve.max()
            if self.inclusions_flag:
                for i in range(len(inclusions_df)):
                    rve[np.where(rve == -(200 + i + 1))] = last_grain_id + i + 1
                rve[np.where(rve == -200)] = last_grain_id + i + 2
            else:
                rve[np.where(rve == -200)] = last_grain_id + 1
            # print(np.asarray(np.unique(rve, return_counts=True)).T)

            # 2.) Make Geometry
            band = True if self.number_of_bands > 0 else False
            if band:
                spectral.make_geom(rve=rve, grid_size=self.n_pts,
                                   spacing=self.box_size,
                                   store_path=store_path)
            else:
                spectral.make_geom(rve=rve, grid_size=self.n_pts, spacing=self.box_size,
                                   store_path=store_path)

            # 3.) Make config / material inpu
            if self.inclusions_flag:
                full_grains = pd.concat([grains_df, inclusions_df])
                full_grains.reset_index(inplace=True)
                full_grains['GrainID'] = full_grains.index
                # print(full_grains)
                spectral.make_config(store_path=store_path, n_grains=full_grains.__len__(), band=band,
                                     grains_df=full_grains)
            else:
                spectral.make_config(store_path=store_path, n_grains=grains_df.__len__(), band=band,
                                     grains_df=grains_df)

            # 4.) Last - Make load
            spectral.make_load_from_defgrad(file_path='../ExampleInput/Defgrad.txt', store_path=store_path)

        elif self.solver == 'FEM':
            if rve_status:
                with open(store_path + '/rve.sta', 'a') as sta:
                    sta.writelines("\n\nRVE generation process nearly complete: Creating Abaqus input now: \n")
                periodic_rve_df = super().repair_periodicity_3D(rve)
                periodic_rve_df['phaseID'] = 0
                grains_df.sort_values(by=['GrainID'])
                # print(grains_df)

                for i in range(len(grains_df)):
                    # Set grain-ID to number of the grain
                    # Denn Grain-ID ist entweder >0 oder -200 oder >-200
                    periodic_rve_df.loc[periodic_rve_df['GrainID'] == i + 1, 'phaseID'] = grains_df['phaseID'][i]

                # For the inclusions:
                if self.inclusions_flag:
                    # Zuweisung der negativen grain-ID's
                    for j in range(len(inclusions_df)):
                        periodic_rve_df.loc[periodic_rve_df['GrainID'] == -(200 + j + 1), 'GrainID'] = (i + j + 2)
                        periodic_rve_df.loc[periodic_rve_df['GrainID'] == (i + j + 2), 'phaseID'] = 2

                # print(periodic_rve_df)
                # print(periodic_rve_df['GrainID'].value_counts())

                if self.number_of_bands > 0 and self.inclusions_flag:
                    # Set the points where == -200 to phase 2 and to grain ID i + j + 3
                    periodic_rve_df.loc[periodic_rve_df['GrainID'] == -200, 'GrainID'] = (i + j + 3)
                    periodic_rve_df.loc[periodic_rve_df['GrainID'] == (i + j + 3), 'phaseID'] = 2
                else:
                    # Set the points where == -200 to phase 2 and to grain ID i + 2
                    periodic_rve_df.loc[periodic_rve_df['GrainID'] == -200, 'GrainID'] = (i + 2)
                    periodic_rve_df.loc[periodic_rve_df['GrainID'] == (i + 2), 'phaseID'] = 2

                # Start the Mesher
                mesher_obj = Mesher(periodic_rve_df, store_path=store_path, phase_two_isotropic=True, animation=False,
                                    grains_df=grains_df, gui=False, elem='C3D4')
                mesher_obj.mesh_and_build_abaqus_model()

    def post_processing(self):
        if self.solver == 'Spectral':
            with open(self.store_path + '/rve.sta', 'a') as sta:
                sta.writelines('Attention: Discrete and continuous Output are equal for the spectral grid! \n')
        obj = PostProcVol(self.store_path, dim_flag=3)
        phase1_ratio_conti_in, phase1_ref_r_conti_in, phase1_ratio_discrete_in, phase1_ref_r_discrete_in, \
        phase2_ratio_conti_in, phase2_ref_r_conti_in, phase2_ratio_discrete_in, phase2_ref_r_discrete_in, \
        phase1_ratio_conti_out, phase1_ref_r_conti_out, phase1_ratio_discrete_out, phase1_ref_r_discrete_out, \
        phase2_ratio_conti_out, phase2_ref_r_conti_out, phase2_ratio_discrete_out, phase2_ref_r_discrete_out = \
            obj.gen_in_out_lists()

        if phase2_ratio_conti_in != 0:
            obj.gen_pie_chart_phases(phase1_ratio_conti_in, phase2_ratio_conti_in, 'input_conti')
            obj.gen_pie_chart_phases(phase1_ratio_conti_out, phase2_ratio_conti_out, 'output_conti')
            obj.gen_pie_chart_phases(phase1_ratio_discrete_in, phase2_ratio_discrete_in, 'input_discrete')
            obj.gen_pie_chart_phases(phase1_ratio_discrete_out, phase2_ratio_discrete_out, 'output_discrete')

            obj.gen_plots(phase1_ref_r_discrete_in, phase1_ref_r_discrete_out, 'phase 1 discrete',
                          'phase1vs2_discrete',
                          phase2_ref_r_discrete_in, phase2_ref_r_discrete_out, 'phase 2 discrete')
            obj.gen_plots(phase1_ref_r_conti_in, phase1_ref_r_conti_out, 'phase 1 conti', 'phase1vs2_conti',
                          phase2_ref_r_conti_in, phase2_ref_r_conti_out, 'phase 2 conti')
            if self.gui_flag:
                self.infobox_obj.emit('checkout the evaluation report of the rve stored at:\n'
                                      '{}/Postprocessing'.format(self.store_path))
        else:
            obj.gen_plots(phase1_ref_r_conti_in, phase1_ref_r_conti_out, 'conti', 'in_vs_out_conti')
            obj.gen_plots(phase1_ref_r_discrete_in, phase1_ref_r_discrete_out, 'discrete', 'in_vs_out_discrete')

        with open(self.store_path + '/rve.sta', 'a') as sta:
            sta.writelines("\n\nRVE generation process has successfully completed... \n")
            sta.writelines('------------------------------------------------------------------------------\n')
            sta.writelines('-----------------------------RVE Generation ended-----------------------------\n')
            sta.writelines('------------------------------------------------------------------------------\n')

        # Important if you want to instantiate several DataTaskObjs
        self.logger.handlers.clear()
