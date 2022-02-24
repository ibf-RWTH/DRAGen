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
from sympy import Symbol
from sympy.solvers import solve

from dragen.generation.DiscreteRsa3D import DiscreteRsa3D
from dragen.generation.DiscreteTesselation3D import Tesselation3D
from dragen.utilities.RVE_Utils import RVEUtils
from dragen.generation.mesher import Mesher
from dragen.postprocessing.voldistribution import PostProcVol
from dragen.generation import spectral

from dragen.InputGenerator.C_WGAN_GP import WGANCGP
from dragen.InputGenerator.linking import Reconstructor


# What is the smallest possible axis size? - Maybe 0.5, implemented as lower boundary
class DataTask3D_GAN(RVEUtils):

    def __init__(self, ganfile, trial, epoch, batch,
                 ferrite_grain_resizing=1,
                 ferrite_ar_resizing=1,
                 martensite_grain_resizing=1,
                 martensite_ar_resizing=1,
                 box_size=20, n_pts=32, number_of_bands=0, bandwidth=0, shrink_factor=0.4, band_filling=1.3,
                 phase_ratio=0.95, inclusions_ratio=0.01,
                 solver='Spectral', file1=None, file2=None, store_path=None, gui_flag=False, anim_flag=False,
                 gan_flag=False, exe_flag=False, inclusions_flag=False):
        """identifier"""
        self.epoch = epoch
        self.trial = trial
        self.batch = batch

        """Microstructure variiing parameters"""
        self.phase_ratio = phase_ratio  # 1 means all ferrite, 0 means all Martensite
        self.inclusion_ratio = inclusions_ratio  # Space occupied by inclusions
        self.ferrite_grain_resizing = np.cbrt(
            ferrite_grain_resizing)  # Ferrite grain resizing (1,2 means every grain will increase 20%)
        self.martensite_grain_resizing = np.cbrt(
            martensite_grain_resizing)  # martensite island resizing (1,2 means every grain will increase 20%)
        self.ferrite_ar_resizing = ferrite_ar_resizing  # Ferrite ar changing
        self.martensite_ar_resizing = martensite_ar_resizing  # Martensite ar changing

        """Classical Dragen Parameters"""
        self.logger = logging.getLogger("RVE-Gen")
        self.box_size = box_size
        self.n_pts = n_pts  # has to be even
        self.bin_size = self.box_size / self.n_pts
        self.step_half = self.bin_size / 2
        self.shrink_factor = np.cbrt(shrink_factor)
        self.gui_flag = gui_flag
        self.gan_flag = gan_flag
        self.solver = solver
        self.inclusions_flag = inclusions_flag
        self.root_dir = os.getcwd()
        """
        Aktueller Parametersatz für GAN:
        Parameters:  
        """
        self.number_of_bands = number_of_bands
        if self.number_of_bands == 0:
            self.bandwidth_sum = 0
            self.bandwidth = 0
        else:
            self.bandwidth_sum = np.sum(np.asarray(bandwidth))  # To call sum() on list/tuple whatever
            print('Sum BW: ', self.bandwidth_sum)
            self.bandwidth = np.asarray(bandwidth)
            print('Bandbreiten-Array: ', self.bandwidth)
        self.band_filling = band_filling  # Percentage of Filling for the banded structure
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

        self.ganfile = ganfile
        if self.gan_flag:
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

            specs.writelines('\nModifications of microstructural parameters: \n')
            specs.writelines('Adjustment of martensite size: x {}\n'.format(self.martensite_grain_resizing ** 3))
            specs.writelines('Adjustment of martensite aspect ratios: x {}\n'.format(self.martensite_ar_resizing))

            specs.writelines('\nAdjustment of ferrite size: x {}\n'.format(self.ferrite_grain_resizing ** 3))
            specs.writelines('Adjustment of ferrite aspect ratios: x {}\n'.format(self.ferrite_ar_resizing))

            specs.writelines('\nFor informations, contributing etc. please contact the DRAGEN-Team:\n')
            specs.writelines('\tDRAGen@iehk.rwth-aachen.de\n\n')

    def run_cwgangp(self):
        SOURCE = self.root_dir + '/GANInput'

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
        store_path = self.root_dir + '/Simulations/' + str(datetime.datetime.now())[:10] + \
                     '_Epoch_{}_Trial_{}_batch_{}'.format(self.epoch,
                                                          self.trial,
                                                          self.batch)
        if not os.path.isdir(store_path):
            os.makedirs(store_path)
        GAN = WGANCGP(df_list=[df1, df2, df3, df4, df5, df6, df7], storepath=store_path, num_features=3,
                      gen_iters=500000)
        # Load Data here
        GAN.load_trained_states(single=False, file_list=self.ganfile)

        return GAN

    def plot_comparison(self, inp_list, inp_list_raw, phase):
        header = ['a', 'b', 'c', 'alpha', 'volume']
        df3 = pd.DataFrame(inp_list, columns=header)
        df4 = pd.DataFrame(inp_list_raw, columns=header)

        df3['Type'] = 'Modified'
        df4['Type'] = 'Original'
        full = pd.concat([df3, df4]).reset_index(drop=True)
        full['AR'] = full['a'] / full['b']


        sns.pairplot(full, hue='Type')
        plt.savefig(self.store_path + f'/{phase}Comparison.png')
        plt.close()

    def sample_gan_input(self, adjusted_size, labels=(), size=1000, maximum=None):
        """
        a=x, b=y, c=z
        x and z are the "long" axis, y is the "short" axis going in sheet normal-direction
        So the aspect-ratio to adjust is the aspect ratio between c / b and a / b
        BEWARE: This is different for martensite, where the ratios are b / a and c / a
        (which are equal)
        """
        def reconstruct(size):
            TDxBN = self.GAN.sample_batch(label=labels[0], size=size * 2)
            RDxBN = self.GAN.sample_batch(label=labels[1], size=size * 2)
            RDxTD = self.GAN.sample_batch(label=labels[2], size=size * 2)

            # Run the Reconstruction
            Bot = Reconstructor(TDxBN, RDxBN, RDxTD, drop=True)
            Bot.run(n_points=size)  # Could take a while with more than 500 points...

            return Bot.result_df.copy()

        # Calculate the Boxsize based on Bands and original size
        bs = adjusted_size
        max_volume = bs * bs * bs
        grain_vol = 0
        data = reconstruct(size)
        inp_list = list()  # List for modified points
        inp_list_raw = list()  # List for unmodified points
        print('Maximum Volume is: ', max_volume)
        while (grain_vol > 1.05 * max_volume) or (grain_vol < 1.0 * max_volume):
            # Check available data first
            if data.__len__() <= 10:
                self.logger.info(f'Size is {data.__len__()}')
                data = reconstruct(size)
                self.logger.info(f'Size is {data.__len__()}')
                self.logger.info(f'Filled Volume percentage is: {grain_vol / max_volume}')

            if grain_vol > 1.05 * max_volume:
                idx = np.random.randint(0, inp_list.__len__())
                grain_vol -= inp_list[idx][-1]
                inp_list.pop(idx)
            elif grain_vol < 1.0 * max_volume:
                idx = np.random.randint(0, data.__len__())
                grain = data[['a_final', 'b_final', 'c_final', 'SlopeAB']].iloc[idx].tolist()
                data = data.drop(labels=data.index[idx], axis=0)

                a = grain[0] * self.ferrite_grain_resizing
                b = grain[1] * self.ferrite_grain_resizing
                c = grain[2] * self.ferrite_grain_resizing

                # Add the adjusting of aspect ratio here
                if self.ferrite_ar_resizing == 1:
                    pass
                else:
                    vol = 4 / 3 * np.pi * a * b * c
                    ar1 = a / b * self.ferrite_ar_resizing
                    ar2 = c / b * self.ferrite_ar_resizing
                    a = Symbol('a', real=True)
                    b = Symbol('b', real=True)
                    c = Symbol('c', real=True)
                    f1 = 4 / 3 * np.pi * a * b * c - vol
                    f2 = ar1 * b - a
                    f3 = ar2 * b - c
                    axis = solve([f1, f2, f3], [a, b, c], dict=True)
                    a = float(axis[0][a])
                    b = float(axis[0][b])
                    c = float(axis[0][c])

                # check if size exceeds boxsize
                if a * 2 >= self.box_size or a <= 0.5 or b <= 0.5 or c <= 0.5:
                    continue

                if maximum is None:
                    vol = 4 / 3 * np.pi * a * b * c
                    grain_vol += vol
                    inp_list.append([a, b, c, grain[3], vol])
                    inp_list_raw.append([grain[0], grain[1], grain[2], grain[3],
                                         4 / 3 * np.pi * grain[0] * grain[1] * grain[2]])
                else:
                    if (grain[0] > maximum) or (grain[1] > maximum) or (grain[2] > maximum):
                        pass
                    else:
                        # Only append if smaller
                        vol = 4 / 3 * np.pi * grain[0] * grain[1] * grain[2]
                        grain_vol += vol
                        inp_list.append([a, b, c, grain[3], vol])
                        inp_list_raw.append([grain[0], grain[1], grain[2], grain[3],
                                             4 / 3 * np.pi * grain[0] * grain[1] * grain[2]])
                print(grain_vol)

        # Plot the comparisons
        self.plot_comparison(inp_list, inp_list_raw, phase='Ferrite')

        # Add temporary euler angles
        header = ['a', 'b', 'c', 'alpha', 'volume']
        df3 = pd.DataFrame(inp_list, columns=header)
        df3['phi1'] = (np.random.rand(df3.__len__()) * 360)
        df3['PHI'] = (np.random.rand(df3.__len__()) * 180)
        df3['phi2'] = (np.random.rand(df3.__len__()) * 360)

        with open(self.store_path + '/rve.log', 'a') as log:
            log.writelines('The Ideal volume is: {:.4f}\n'.format(adjusted_size * adjusted_size * adjusted_size))

        return df3  # This is the RVE-Input data

    def sample_gan_input_2d(self, label, boxsize, size=1000, maximum=None, band=False):

        # Sample new data
        def get_raw_data(size):
            df = self.GAN.sample_batch(label=label, size=size)
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

            return df2

        # If bands are there, no modification needed
        if not band:
            martensite_ar_resizing = self.martensite_ar_resizing
            martensite_grain_resizing = self.martensite_grain_resizing
        else:
            martensite_ar_resizing = 1
            martensite_grain_resizing = 1

        # Sample the right amount of Input - MODIFY AS SPECIFIED
        bs = boxsize
        max_volume = bs * bs * bs
        with open(self.store_path + '/rve.log', 'a') as log:
            log.writelines(
                'The Ideal volume is: {:.4f}\n'.format(float(max_volume)))
        grain_vol = 0
        data = get_raw_data(size)
        inp_list = list()
        inp_list_raw = list()
        if self.number_of_bands >= 1:  # Higher bounds w bands to hit martensite percentage
            lower_bound = 1.05
            upper_bound = 1.15
        else:  # Adjust Bounds if no Bands are present
            lower_bound = 1.0
            upper_bound = 1.05

        # Adjust the grain volume to exact value
        while (grain_vol > upper_bound * max_volume) or (grain_vol < lower_bound * max_volume):
            # Check available data first
            if data.__len__() <= 10:
                self.logger.info(f'Size is {data.__len__()}')
                data = get_raw_data(size)
                self.logger.info(f'Size is {data.__len__()}')
                self.logger.info(f'Filled Volume percentage is: {grain_vol/max_volume}')

            if grain_vol > upper_bound * max_volume:
                idx = np.random.randint(0, inp_list.__len__())
                grain_vol -= inp_list[idx][-1]
                inp_list.pop(idx)
            elif grain_vol < lower_bound * max_volume:
                idx = np.random.randint(0, data.__len__())
                grain = data[['Axes1', 'Axes2', 'Axes3', 'Slope']].iloc[idx].tolist()
                data = data.drop(labels=data.index[idx], axis=0)
                a = grain[0] * martensite_grain_resizing
                b = grain[1] * martensite_grain_resizing
                c = grain[2] * martensite_grain_resizing

                # Add the adjusting of aspect ratio here
                if martensite_ar_resizing == 1:
                    pass
                else:
                    vol = 4 / 3 * np.pi * a * b * c
                    ar = b / a * martensite_ar_resizing
                    a = Symbol('a', real=True)
                    b = Symbol('b', real=True)
                    c = Symbol('c', real=True)
                    f1 = 4 / 3 * np.pi * a * b * c - vol
                    f2 = ar * a - b
                    f3 = ar * a - c
                    axis = solve([f1, f2, f3], [a, b, c], dict=True)
                    a = float(axis[0][a])
                    b = float(axis[0][b])
                    c = float(axis[0][c])

                # check if size exceeds boxsize
                if b * 2 >= self.box_size or a <= 0.5 or b <= 0.5 or c <= 0.5:
                    continue

                if maximum is None:
                    vol = 4 / 3 * np.pi * a * b * c
                    grain_vol += vol
                    inp_list.append([a, b, c, grain[3], vol])
                    inp_list_raw.append(
                        [grain[0], grain[1], grain[2], grain[3], 4 / 3 * np.pi * grain[0] * grain[1] * grain[2]])
                else:
                    if (grain[0] > maximum) or (grain[1] > maximum * 2) or (grain[2] > maximum * 2):
                        pass
                    else:
                        # Only append if smaller
                        vol = 4 / 3 * np.pi * a * b * c
                        grain_vol += vol
                        inp_list.append([a, b, c, grain[3], vol])
                        inp_list_raw.append(
                            [grain[0], grain[1], grain[2], grain[3], 4 / 3 * np.pi * grain[0] * grain[1] * grain[2]])

                print(grain_vol)
            print(inp_list.__len__())

        if not band:
            phase = 'Martensite'
        else:
            phase = 'BandedMartensite'
        self.plot_comparison(inp_list, inp_list_raw, phase=phase)

        # Add temporary euler angles
        header = ['a', 'b', 'c', 'alpha', 'volume']
        df3 = pd.DataFrame(inp_list, columns=header)
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

    def initializations(self):
        grains_df = None
        self.setup_logging()
        self.store_path = self.root_dir + '/Simulations/' + str(datetime.datetime.now())[:10] + \
                          '_Epoch_{}_Trial_{}_batch_{}'.format(self.epoch,
                                                               self.trial,
                                                               self.batch)
        self.fig_path = self.store_path + '/Figs'
        self.gen_path = self.store_path + '/Generation_Data'
        self.logger.info(f'Shrinkfactor is {self.shrink_factor}')
        if not os.path.isdir(self.store_path):
            os.makedirs(self.store_path)
        if not os.path.isdir(self.fig_path):
            os.makedirs(self.fig_path)  # Second if needed
        if not os.path.isdir(self.gen_path):
            os.makedirs(self.gen_path)  # Second if needed

        return grains_df, self.store_path

    def rve_generation(self, grains_df, store_path):

        with open(store_path + '/rve.log', 'a') as log:
            log.writelines('------------------------------------------------------------------------------\n')
            log.writelines('----------------------------RVE Generation started----------------------------\n')
            log.writelines('------------------------------------------------------------------------------\n')
            log.writelines('\n\n')
            now = datetime.datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            log.writelines('Starttime: ' + dt_string)
            log.writelines('\n\n\n')
            log.writelines('Calculating the phase ratio...\n')

        print('Bandberechnung')
        percentage_in_bands = (self.bandwidth_sum * self.box_size ** 2) / \
                              (self.box_size ** 3)
        print(percentage_in_bands)

        self.percentage_in_bands = percentage_in_bands
        new_phase_ratio = self.phase_ratio - percentage_in_bands
        self.new_phase_ratio = new_phase_ratio
        self.write_specs(self.store_path)
        if new_phase_ratio < 0:
            with open(store_path + '/rve.log', 'a') as log:
                log.writelines('The bands are containing more martensite then specfified for the overall volume! \n')
                log.writelines('Setting the phase ratio for the rest of the volume to 1 \n')
            new_phase_ratio = 0
        else:
            with open(store_path + '/rve.log', 'a') as log:
                log.writelines('The phase ratio for the rest of the volume is: {} \n'.format(new_phase_ratio))
        """
        -------------------------------------------------------------------------------
        BAND-PHASE GENERATION HERE
        phaseID == 2 (Martensite)
        adjusted_size is: n_bands * bw * bs**2
        flag before the field
        -------------------------------------------------------------------------------
        """

        if self.number_of_bands > 0:
            starttime = time.time()
            with open(store_path + '/rve.log', 'a') as log:
                log.writelines("\nRVE generation process has started with Band-creation \n")
                log.writelines(
                    "The total volume of the RVE is {0}*{0}*{0} = {1} \n".format(self.box_size, self.box_size ** 3))
                adjusted_size = np.cbrt((self.bandwidth[0] * self.box_size ** 2) * self.band_filling)
                phase1 = self.sample_gan_input_2d(size=10000, label=6, boxsize=adjusted_size,
                                                  maximum=self.bandwidth[0], band=True)
                bands_df = super().process_df(phase1, float(self.shrink_factor))
                bands_df['phaseID'] = 2
                log.writelines('Sampled {} Martensite-Points for band-Creation! \n'.format(bands_df.__len__()))
                log.writelines('The total conti volume is: {} \n'.format(np.sum(bands_df['final_conti_volume'].values)))
                log.writelines(
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
                with open(store_path + '/rve.log', 'a') as log:
                    log.writelines('\n\n Band number {}\n'.format(i + 1))
                    adjusted_size = np.cbrt((self.bandwidth[i] * self.box_size ** 2) * self.band_filling)
                    phase1 = self.sample_gan_input_2d(size=10000, label=6, boxsize=adjusted_size,
                                                      maximum=self.bandwidth[i], band=True)
                    new_df = super().process_df(phase1, float(self.shrink_factor))
                    new_df['phaseID'] = 2
                    log.writelines('Sampled {} Martensite-Points for band-Creation! \n'.format(new_df.__len__()))
                    log.writelines \
                        ('The total conti volume is: {} \n'.format(np.sum(new_df['final_conti_volume'].values)))
                    log.writelines(
                        'The total discrete volume is: {} \n'.format(np.sum(new_df['final_discrete_volume'].values)))
                    bands_df = pd.concat([bands_df, new_df])
                    bands_df.reset_index(inplace=True, drop=True)
                    bands_df['GrainID'] = bands_df.index
                # ---------------------------------------------------------------------------------------------------

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
                startindex = int(np.amin(rsa) + 1000) * -1
                print(startindex)
                rsa, x_0, y_0, z_0, rsa_status = discrete_RSA_obj.run_rsa_clustered(
                    previous_rsa=rsa,
                    band_array=band_array_new,
                    animation=True, startindex=startindex)  # Maximum value of previous is startpoint of new

                x_0_list.extend(x_0)
                y_0_list.extend(y_0)
                z_0_list.extend(z_0)

            endtime = datetime.timedelta(seconds=(time.time() - starttime))

            with open(store_path + '/rve.log', 'a') as log:
                log.writelines('Elapsed time for band creation: {}\n\n'.format(endtime))
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
            with open(store_path + '/rve.log', 'a') as log:
                log.writelines("RVE generation process continues with placing of the ferrite grains and martensite "
                               "islands \n")
                # Ferrite:
                adjusted_size_ferrite = np.cbrt(
                    (self.box_size ** 3 - self.band_filling * self.bandwidth_sum * self.box_size ** 2) * (
                            1 - new_phase_ratio))
                phase1 = self.sample_gan_input(size=800, labels=[0, 1, 2], adjusted_size=adjusted_size_ferrite)
                grains_df = super().process_df(phase1, float(self.shrink_factor))
                grains_df['phaseID'] = 1  # Ferrite_Grains
                log.writelines('Sampled {} Ferrite-Points for the matrix! \n'.format(grains_df.__len__()))
                log.writelines(
                    'The total conti volume is: {} \n'.format(np.sum(grains_df['final_conti_volume'].values)))
                log.writelines(
                    'The total discrete volume is: {} \n'.format(np.sum(grains_df['final_discrete_volume'].values)))
                # Martensite
                adjusted_size_martensite = np.cbrt(
                    (
                            self.box_size ** 3 - self.bandwidth_sum * self.box_size ** 2)
                    * new_phase_ratio)
                phase2 = self.sample_gan_input_2d(size=1000, label=6, boxsize=adjusted_size_martensite,
                                                  maximum=None)
                grains_df_2 = super().process_df(phase2, float(self.shrink_factor))
                grains_df_2['phaseID'] = 2  # Martensite_Islands
                log.writelines('Sampled {} Martensite-Islands for the matrix! \n'.format(grains_df_2.__len__()))
                log.writelines(
                    'The total conti volume is: {} \n'.format(np.sum(grains_df_2['final_conti_volume'].values)))
                log.writelines(
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
            with open(store_path + '/rve.log', 'a') as log:
                log.writelines('\nElapsed time for normal RSA: {}\n\n'.format(endtime))
        else:
            with open(store_path + '/rve.log', 'a') as log:
                log.writelines("The normal rsa did not succeed...Please check input and logging for Errors \n")
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
            with open(store_path + '/rve.log', 'a') as log:
                log.writelines("\n\nRVE generation process continues with the tesselation of grains!\n")
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
            with open(store_path + '/rve.log', 'a') as log:
                log.writelines('Elapsed time for Tesselation: {}\n\n'.format(endtime))
        else:
            with open(store_path + '/rve.log', 'a') as log:
                log.writelines("The tesselation did not succeed...Check input and logging for Errors \n")
            sys.exit()

        """
        -------------------------------------------------------------------------
        PLACE THE INCLUSIONS!
        The inclusions will not grow and are therefore placed in the rve AFTER the
        Tesselation! The inclusions will placed inside another grain and cannot cut a 
        grain boundary. Labels are lower -200
        -------------------------------------------------------------------------
        """
        if rve_status and self.inclusions_flag and self.inclusion_ratio != 0:
            with open(store_path + '/rve.log', 'a') as log:
                log.writelines("\n\nRVE generation process reaches final steps: Placing inclusions in the matrix\n")
                adjusted_size = self.box_size * np.cbrt(self.inclusion_ratio)  # 1% as an example
                inclusions = self.sample_gan_input(size=200, labels=(3, 4, 5), adjusted_size=adjusted_size)
                inclusions_df = super().process_df(inclusions, float(self.shrink_factor))
                inclusions_df['phaseID'] = 2  # Inclusions also elastic
                log.writelines('Sampled {} inclusions for the RVE\n'.format(inclusions_df.__len__()))
                log.writelines(
                    'The total conti volume is: {}\n'.format(np.sum(inclusions_df['final_conti_volume'].values)))
                log.writelines(
                    'The total discrete volume is: {}\n'.format(np.sum(inclusions_df['final_discrete_volume'].values)))

            discrete_RSA_inc_obj = DiscreteRsa3D(self.box_size, self.n_pts,
                                                 inclusions_df['a'].tolist(),
                                                 inclusions_df['b'].tolist(),
                                                 inclusions_df['c'].tolist(),
                                                 inclusions_df['alpha'].tolist(), store_path=store_path)

            rve, rve_status = discrete_RSA_inc_obj.run_rsa_inclusions(rve, animation=True)

        """
        -------------------------------------------------------------------------
        CREATE INPUT FOR NUMERICAL SOLVER!
        There are two options available at the moment, Abaqus/implicit and 
        DAMASK_Spectral.
        Control via solver == 'FEM' or 'Spectral'
        -------------------------------------------------------------------------
        """
        if self.solver == 'Spectral':
            with open(store_path + '/rve.log', 'a') as log:
                log.writelines("\n\nRVE generation process nearly complete: Creating input for DAMASK Spectral now: \n")

            # 1.) Write out Volume
            grains_df.sort_values(by=['GrainID'], inplace=True)
            disc_vols = np.zeros((1, grains_df.shape[0])).flatten().tolist()
            for i in range(len(grains_df)):
                # grainID = grains_df.GrainID[i]
                disc_vols[i] = np.count_nonzero(rve == i + 1) * self.bin_size ** 3

            grains_df['meshed_conti_volume'] = disc_vols
            grains_df.to_csv(self.store_path + '/Generation_Data/grain_data_output_conti.csv', index=False)

            # Startpoint: Rearrange the negative ID's
            last_grain_id = rve.max()  # BEWARE: For the .vti file, the grid must start at ZERO
            print('The last grain ID is:', last_grain_id)
            print('The number of bands is:', self.number_of_bands)
            if self.inclusions_flag and (self.number_of_bands >= 1) and self.inclusion_ratio != 0:
                with open(store_path + '/rve.log', 'a') as log:
                    log.writelines("Band and Inclusions in RVE: \n")
                phase_list = grains_df['phaseID'].tolist()
                for i in range(len(inclusions_df)):
                    rve[np.where(rve == -(200 + i + 1))] = last_grain_id + i + 1
                    phase_list.append(3)  # Inclusion is phase THREE

                # Add one Martensite for the band
                rve[np.where(rve == -200)] = last_grain_id + i + 2  # Band is last ID
                phase_list.append(2)

                # Calc the volume and write it to specs.txt
                martensite_vol = 0
                ferrite_vol = 0
                for j in range(phase_list.__len__()):
                    if phase_list[j] == 1:
                        vol = np.count_nonzero(rve == j + 1)
                        ferrite_vol += vol
                    else:
                        vol = np.count_nonzero(rve == j + 1)
                        martensite_vol += vol

                full_points = martensite_vol + ferrite_vol
                print('Summe der Punkte: ', full_points)
                print('Größe des RVEs: ', self.n_pts ** 3)

                ferrite_per = ferrite_vol / full_points
                martensite_per = martensite_vol / full_points

                with open(store_path + '/Specs.txt', 'a') as specs:
                    specs.writelines('Percentage of Ferrite: {:.6%}\n'.format(ferrite_per))
                    specs.writelines('Percentage of Martensite: {:.6%}\n'.format(martensite_per))

            elif self.number_of_bands >= 1:
                with open(store_path + '/rve.log', 'a') as log:
                    log.writelines("Only Band in RVE: \n")
                # Case with bands here add one martensite for the band
                phase_list = grains_df['phaseID'].tolist()
                rve[np.where(rve == -200)] = last_grain_id + 1
                phase_list.append(2)

                # Calc the volume and write it to specs.txt
                martensite_vol = 0
                ferrite_vol = 0
                for j in range(phase_list.__len__()):
                    if phase_list[j] == 1:
                        vol = np.count_nonzero(rve == j + 1)
                        ferrite_vol += vol
                    else:
                        vol = np.count_nonzero(rve == j + 1)
                        martensite_vol += vol

                full_points = martensite_vol + ferrite_vol
                print('Summe der Punkte: ', full_points)
                print('Größe des RVEs: ', self.n_pts ** 3)

                ferrite_per = ferrite_vol / full_points
                martensite_per = martensite_vol / full_points

                with open(store_path + '/Specs.txt', 'a') as specs:
                    specs.writelines('Percentage of Ferrite: {:.6%}\n'.format(ferrite_per))
                    specs.writelines('Percentage of Martensite: {:.6%}\n'.format(martensite_per))

            elif self.inclusions_flag and self.inclusion_ratio != 0:
                with open(store_path + '/rve.log', 'a') as log:
                    log.writelines("Only Inclusions in RVE: \n")
                phase_list = grains_df['phaseID'].tolist()
                for i in range(len(inclusions_df)):
                    rve[np.where(rve == -(200 + i + 1))] = last_grain_id + i + 1
                    phase_list.append(3)

                # Calc the volume and write it to specs.txt
                martensite_vol = 0
                ferrite_vol = 0
                for j in range(phase_list.__len__()):
                    if phase_list[j] == 1:
                        vol = np.count_nonzero(rve == j + 1)
                        ferrite_vol += vol
                    else:
                        vol = np.count_nonzero(rve == j + 1)
                        martensite_vol += vol

                full_points = martensite_vol + ferrite_vol
                print('Summe der Punkte: ', full_points)
                print('Größe des RVEs: ', self.n_pts ** 3)

                ferrite_per = ferrite_vol / full_points
                martensite_per = martensite_vol / full_points

                with open(store_path + '/Specs.txt', 'a') as specs:
                    specs.writelines('Percentage of Ferrite: {:.6%}\n'.format(ferrite_per))
                    specs.writelines('Percentage of Martensite: {:.6%}\n'.format(martensite_per))
            else:
                with open(store_path + '/rve.log', 'a') as log:
                    log.writelines("Nothing despite grains in RVE: \n")
                # No bands, no inclusions, just turn grains to phase_list
                phase_list = grains_df['phaseID'].tolist()

                # Calc the volume and write it to specs.txt
                martensite_vol = 0
                ferrite_vol = 0
                for j in range(phase_list.__len__()):
                    if phase_list[j] == 1:
                        vol = np.count_nonzero(rve == j + 1)
                        ferrite_vol += vol
                    else:
                        vol = np.count_nonzero(rve == j + 1)
                        martensite_vol += vol

                full_points = martensite_vol + ferrite_vol
                print('Summe der Punkte: ', full_points)
                print('Größe des RVEs: ', self.n_pts ** 3)

                ferrite_per = ferrite_vol / full_points
                martensite_per = martensite_vol / full_points

                with open(store_path + '/Specs.txt', 'a') as specs:
                    specs.writelines('Percentage of Ferrite: {:.6%}\n'.format(ferrite_per))
                    specs.writelines('Percentage of Martensite: {:.6%}\n'.format(martensite_per))

            spectral.write_material(store_path=store_path, grains=phase_list)
            # spectral.write_load(store_path)
            spectral.write_grid(store_path=store_path, rve=rve, spacing=self.box_size / 1000, grains=phase_list)

        elif self.solver == 'FEM':
            if rve_status:
                with open(store_path + '/rve.log', 'a') as log:
                    log.writelines("\n\nRVE generation process nearly complete: Creating Abaqus input now: \n")
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
            with open(self.store_path + '/rve.log', 'a') as log:
                log.writelines('Attention: Discrete and continuous Output are equal for the spectral grid! \n')
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

        # Write a detection file
        with open(self.store_path + '/gen.finished', 'w') as fin:
            fin.writelines('Generation Finished')

        with open(self.store_path + '/rve.log', 'a') as log:
            log.writelines("\n\nRVE generation process has successfully completed... \n")
            log.writelines('------------------------------------------------------------------------------\n')
            log.writelines('-----------------------------RVE Generation ended-----------------------------\n')
            log.writelines('------------------------------------------------------------------------------\n')

        # Important if you want to instantiate several DataTaskObjs
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)


# Function to call
def call_dragen(trial, epoch, batch, mgr, mar, fgr, far, phase_ratio, n_bands, size=16, res=32, inclusion_flag=True,
                inclusion_rate=.0):

    path = r'GANInput'
    ganfile = [path + '/TrainedData_{}.pkl'.format(i) for i in range(7)]
    bandwidths = pd.read_csv(path + '/Bandwidths.csv')

    if n_bands > 0:
        min_bw = 0.0
        while min_bw <= 1:
            indices = np.random.randint(0, bandwidths.__len__(), size=n_bands)
            bandwidth = bandwidths.iloc[indices].to_numpy()
            min_bw = np.amin(bandwidth)
    else:
        bandwidth = 0

    obj3D = DataTask3D_GAN(trial=trial, epoch=epoch, batch=batch,
                           martensite_grain_resizing=mgr,
                           martensite_ar_resizing=mar,
                           ferrite_grain_resizing=fgr,
                           ferrite_ar_resizing=far,
                           phase_ratio=float(phase_ratio),
                           ganfile=ganfile,
                           box_size=size, n_pts=res,
                           number_of_bands=n_bands,
                           bandwidth=bandwidth, shrink_factor=0.4, band_filling=1.3, inclusions_ratio=inclusion_rate,
                           inclusions_flag=inclusion_flag, solver='Spectral', file1=None, file2=None, store_path=os.getcwd(),
                           gui_flag=False, anim_flag=False, gan_flag=True, exe_flag=False)
    grains_df, store_path = obj3D.initializations()
    obj3D.rve_generation(grains_df, store_path)
    obj3D.post_processing()
