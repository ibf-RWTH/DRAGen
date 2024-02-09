import sys

import damask
import pandas as pd
import numpy as np
import datetime
from tkinter import messagebox
from dragen.utilities.InputInfo import RveInfo
from dragen.InputGenerator.C_WGAN_GP import WGANCGP
from dragen.InputGenerator.linking import Reconstructor


class HelperFunctions:
    """Common Representative Volume Element (RVE) operations."""

    def __init__(self, x_grid=None, y_grid=None, z_grid=None) -> None:

        # The following variables are not available in InputInfo due to possible changes
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_grid = z_grid

    @staticmethod
    # TODO: implement array with real shape and change code to roll method rather than having 27 arrays sourrounding
    def gen_array_new() -> np.zeros:
        npts_x = RveInfo.n_pts
        if RveInfo.box_size_y is None and RveInfo.box_size_z is None:
            array = np.zeros((npts_x, npts_x, npts_x), order='C', dtype='int16')
        elif RveInfo.box_size_y is not None and RveInfo.box_size_z is None:
            array = np.zeros((npts_x, RveInfo.n_pts_y, npts_x), order='C', dtype='int16')
        elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
            array = np.zeros((npts_x, npts_x, RveInfo.n_pts_z), order='C', dtype='int16')
        else:
            array = np.zeros((2 * npts_x, 2 * RveInfo.n_pts_y, 2 * RveInfo.n_pts_z), order='C', dtype='int16')
        return array

    @staticmethod
    def gen_array() -> np.zeros:
        npts_x = RveInfo.n_pts
        if RveInfo.box_size_y is None and RveInfo.box_size_z is None:
            array = np.zeros((2 * npts_x, 2 * npts_x, 2 * npts_x), order='C')
        elif RveInfo.box_size_y is not None and RveInfo.box_size_z is None:
            array = np.zeros((2 * npts_x, 2 * RveInfo.n_pts_y, 2 * npts_x), order='C')

        elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
            array = np.zeros((2 * npts_x, 2 * npts_x, 2 * RveInfo.n_pts_z), order='C')

        else:
            array = np.zeros((2 * npts_x, 2 * RveInfo.n_pts_y, 2 * RveInfo.n_pts_z), order='C')
        return array

    @staticmethod
    def gen_array_2d() -> np.zeros:
        array = np.zeros((2 * RveInfo.n_pts, 2 * RveInfo.n_pts))
        return array

    @staticmethod
    def gen_grid_new(shape):
        npts_x = RveInfo.n_pts
        if RveInfo.box_size_y is None and RveInfo.box_size_z is None:
            xyz = np.linspace(0, RveInfo.box_size, shape[0], endpoint=True, dtype=np.float32)
            x_grid, y_grid, z_grid = np.meshgrid(xyz, xyz, xyz, indexing='ij')
        elif RveInfo.box_size_y is not None and RveInfo.box_size_z is None:
            xz = np.linspace(0, RveInfo.box_size, shape[0], endpoint=True, dtype=np.float32)
            y = np.linspace(0, RveInfo.box_size_y, shape[1], endpoint=True, dtype=np.float32)
            x_grid, y_grid, z_grid = np.meshgrid(xz, y, xz, indexing='ij')

        elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
            xy = np.linspace(0, RveInfo.box_size, shape[0], endpoint=True, dtype=np.float32)
            z = np.linspace(0, RveInfo.box_size_z, shape[2], endpoint=True, dtype=np.float32)
            x_grid, y_grid, z_grid = np.meshgrid(xy, xy, z, indexing='ij')
        else:
            x = np.linspace(0, RveInfo.box_size, shape[0], endpoint=True, dtype=np.float32)
            y = np.linspace(0, RveInfo.box_size_y, shape[1], endpoint=True, dtype=np.float32)
            z = np.linspace(0, RveInfo.box_size_z, shape[2], endpoint=True, dtype=np.float32)
            x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        return x_grid, y_grid, z_grid

    @staticmethod
    def gen_grid():
        npts_x = RveInfo.n_pts
        if RveInfo.box_size_y is None and RveInfo.box_size_z is None:
            xyz = np.linspace(-RveInfo.box_size / 2, RveInfo.box_size + RveInfo.box_size / 2, 2 * npts_x, endpoint=True)
            x_grid, y_grid, z_grid = np.meshgrid(xyz, xyz, xyz, indexing='ij')
        elif RveInfo.box_size_y is not None and RveInfo.box_size_z is None:
            xz = np.linspace(-RveInfo.box_size / 2, RveInfo.box_size + RveInfo.box_size / 2, 2 * npts_x, endpoint=True)
            y = np.linspace(-RveInfo.box_size_y / 2, RveInfo.box_size_y + RveInfo.box_size_y / 2, 2 * RveInfo.n_pts_y, endpoint=True)
            x_grid, y_grid, z_grid = np.meshgrid(xz, y, xz, indexing='ij')

        elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
            xy = np.linspace(-RveInfo.box_size / 2, RveInfo.box_size + RveInfo.box_size / 2, 2 * npts_x, endpoint=True)
            z = np.linspace(-RveInfo.box_size_z / 2, RveInfo.box_size_z + RveInfo.box_size_z / 2, 2 * RveInfo.n_pts_z, endpoint=True)
            x_grid, y_grid, z_grid = np.meshgrid(xy, xy, z, indexing='ij')
        else:
            x = np.linspace(-RveInfo.box_size / 2, RveInfo.box_size + RveInfo.box_size / 2, 2 * npts_x, endpoint=True)
            y = np.linspace(-RveInfo.box_size_y / 2, RveInfo.box_size_y + RveInfo.box_size_y / 2, 2 * RveInfo.n_pts_y, endpoint=True)
            z = np.linspace(-RveInfo.box_size_z / 2, RveInfo.box_size_z + RveInfo.box_size_z / 2, 2 * RveInfo.n_pts_z, endpoint=True)
            x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
        return x_grid, y_grid, z_grid

    @staticmethod
    def gen_grid2d():
        xy = np.linspace(-RveInfo.box_size / 2, RveInfo.box_size + RveInfo.box_size / 2, 2 * RveInfo.n_pts, endpoint=True)
        x_grid, y_grid = np.meshgrid(xy, xy, indexing='ij')
        return x_grid, y_grid

    def read_input(self, file_name, dimension) -> pd.DataFrame: #TODO @Max: Ich würde hier noch mal drübergucken, glaub die funktion ist ziemlich langsam bei große .csvs (Niklas)
        """Reads the given input file and returns the volume along with radii, rotation angles and texture parameters.
        Parameter :
        file_name : String, name of the input file
        """
        data = pd.read_csv(file_name)
        radius_a, radius_b, radius_c, alpha, tex_phi1, tex_PHI, tex_phi2 = ([] for i in range(7))

        if 'a' in data.head(0) and data['a'].count() != 0:
            for rad in data['a']:
                radius_a.append(rad)
        else:
            if not RveInfo.gui_flag:
                print('No "a" in given .csv-Inputfile! RVE-Generation was canceled!')
            else:
                messagebox.showinfo(message='No "a" in given .csv-Inputfile! RVE-Generation was canceled!',
                                    title='ERROR')
            RveInfo.LOGGER.info('ERROR: No "a" in given .csv-Inputfile! RVE-Generation was canceled!')
            sys.exit()

        if 'b' in data.head(0) and data['b'].count() != 0:
            for rad in data['b']:
                radius_b.append(rad)
        else:
            radius_b = radius_a
            RveInfo.LOGGER.info('No "b" in given .csv-Inputfile! Assumption: b = a')

        if 'c' in data.head(0) and data['c'].count() != 0:
            for rad in data['c']:
                radius_c.append(rad)
        else:
            radius_c = radius_a
            RveInfo.LOGGER.info('No "c" in given .csv-Inputfile! Assumption: c = a')

        if 'alpha' in data.head(0) and data['alpha'].count() != 0:
            for ang in data['alpha']:
                alpha.append(ang)
        else:
            alpha = [0] * len(radius_a)
            RveInfo.LOGGER.info('No "alpha" in given .csv-Inputfile! Assumption: alpha = 0, no rotation')

        if 'phi1' in data.head(0) and data['phi1'].count() != 0 and 'PHI' in data.head(0) and data['PHI'].count() != 0 \
                and 'phi2' in data.head(0) and data['phi2'].count() != 0:
            for tex in data['phi1']:
                tex_phi1.append(tex)
            for tex in data['PHI']:
                tex_PHI.append(tex)
            for tex in data['phi2']:
                tex_phi2.append(tex)
        else:
            RveInfo.LOGGER.info(
                'No texture parameters (phi1, PHI, phi2) in given .csv-Inputfile! Assumption: random texture')
            i = 0
            while i < len(radius_a):
                o = damask.Rotation.from_random(1).as_Euler_angles(degrees=True)  # Rotation based on Damask
                tex_phi1.append(float(o[:, 0]))
                tex_PHI.append(float(o[:, 1]))
                tex_phi2.append(float(o[:, 2]))
                i = i + 1

        if dimension == 3:

            grain_dict = {"a": radius_a, "b": radius_b, "c": radius_c, "alpha": alpha,
                                             "phi1": tex_phi1, "PHI": tex_PHI, "phi2": tex_phi2}
            grain_df = pd.DataFrame(data=grain_dict,
                                    columns=["a", "b", "c", "alpha", "phi1", "PHI", "phi2"])
            return grain_df
        elif dimension == 2:
            grain_dict = {"a": radius_a, "b": radius_b, "alpha": alpha,
                          "phi1": tex_phi1, "PHI": tex_PHI, "phi2": tex_phi2}
            grain_df = pd.DataFrame(data=grain_dict,
                                    columns=["a", "b", "alpha", "phi1", "PHI", "phi2"])

            return grain_df

    def read_input_gan(self, file_name, dimension, size) -> pd.DataFrame:
        """
        Reads a .pckl file and transferes it to a df with grains
        ATTENTION: Assumes data like Area, Aspect Ratio, Slope (Angles)
            --> Not suitable if Gan is directly trained on axis sizes
        """
        GAN = WGANCGP(df_list=[], storepath=RveInfo.store_path, num_features=3,
                      gen_iters=500000)
        # Load Data here
        GAN.load_trained_states(single=False, file_list=[file_name])

        df = GAN.sample_batch(label=0, size=size)  # Man braucht hier auf jeden Fall Pandas > 1.15
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
        # Set c = a due to coming rotation - TODO: Setzt das voraus, muss bei den Trainingsdaten passen
        df2['Axes3'] = df2['Axes1']
        data = df2.copy()

        if df.columns.__len__() <= 3:
            o = damask.Rotation.from_random(data.__len__()).as_Euler_angles(degrees=True)
            data['phi1'] = o[:, 0]
            data['PHI'] = o[:, 1]
            data['phi2'] = o[:, 2]
            data = data[[data.columns[2], data.columns[3], data.columns[4], data.columns[5],
                         data.columns[6], data.columns[7], data.columns[8]]]
        else:
            data = data[[data.columns[2], data.columns[6], data.columns[7], data.columns[8],
                        data.columns[3], data.columns[4], data.columns[5]]]
        data.columns = ['alpha', 'a', 'b', 'c', 'phi1', 'PHI', 'phi2']

        if dimension == 3:
            return data

        elif dimension == 2:
            data = data.drop(labels=['c'], axis=1).reset_index(drop=True)
            return data

    def sample_input_3D(self, data, bs, constraint=None) -> pd.DataFrame:
        if constraint is None:
            constraint = 10000
        else:
            constraint = constraint
        max_volume = bs**3
        min_rad = RveInfo.bin_size

        data["volume"] = 4/3*np.pi*data["a"]*data["b"]*data["c"]
        data = data.loc[data["a"] < RveInfo.box_size/2]

        grain_vol = 0
        old_idx = list()
        input_df = pd.DataFrame()
        print('len:', data.__len__())
        while (grain_vol > 1.05 * max_volume) or (grain_vol < 1.0 * max_volume):

            if grain_vol > 1.05*max_volume:

                grain_vol -= input_df["volume"].iloc[-1]

                input_df = input_df[:-1]  # delete last row if volume was exceeded
                old_idx.pop(-1)

            elif grain_vol < max_volume:
                idx = np.random.randint(0, data.__len__(), dtype='int64')

                grain = data.loc[data.index[idx]]
                grain_df = pd.DataFrame(grain).transpose()
                data = data.drop(labels=data.index[idx], axis=0)
                if (grain['a'] > constraint*2) or (grain['b'] > constraint) or (grain['c'] > constraint*2):  # Dickenunterschied für die Bänbder
                    continue
                elif np.cbrt(grain["a"]*grain["b"]*grain["c"]) < min_rad:
                    continue

                old_idx.append(idx)
                vol = grain["volume"]
                grain_vol += vol
                input_df = pd.concat([input_df, grain_df])
                if len(data) == 0:
                    RveInfo.LOGGER.info('Input data was exceeded not enough data!!')
                    break

        print('Volume of df', input_df['volume'].sum())
        input_df['old_gid'] = old_idx # get old idx so that the substructure generator knows which grains are chosen in the input data
        return input_df

    def sample_input_2D(self, data, bs, constraint=None) -> pd.DataFrame:

        if constraint is None:
            constraint = 10000
        else:
            constraint = constraint
        max_volume = bs**2
        data["volume"] = np.pi*data["a"]*data["b"]
        data = data.loc[data["a"] < RveInfo.box_size/2]

        grain_vol = 0
        old_idx = list()
        input_df = pd.DataFrame()
        while (grain_vol > 1.05 * max_volume) or (grain_vol < 1.0 * max_volume):

            if grain_vol > 1.05*max_volume:

                grain_vol -= input_df["volume"].iloc[-1]

                input_df = input_df[:-1]  # delete last row if volume was exceeded
                old_idx.pop(-1)

            elif grain_vol < max_volume:
                idx = np.random.randint(0, data.__len__())

                grain = data.loc[data.index[idx]]
                grain_df = pd.DataFrame(grain).transpose()
                data = data.drop(labels=data.index[idx], axis=0)
                if (grain['a'] > constraint*2) or (grain['b'] > constraint):  # Dickenunterschied für die Bänbder
                    continue

                old_idx.append(idx)
                vol = grain["volume"]
                grain_vol += vol
                input_df = pd.concat([input_df,grain_df])
                if len(data) == 0:
                    RveInfo.LOGGER.info('Input data was exceeded not enough data!!')
                    break

        print('Volume of df', input_df['volume'].sum())
        input_df['old_gid'] = old_idx # get old idx so that the substructure generator knows which grains are chosen in the input data
        return input_df

    def convert_volume_3D(self, radius_a, radius_b, radius_c):
        """Compute the volume for the given radii.
        Parameters :
        radius_a : Integer, radius along x-axis
        radius_b : Integer, radius along y-axis
        radius_b : Integer, radius along z-axis
        """
        #vol = 4/3*np.pi*radius_b*radius_b*radius_c
        array = self.gen_array_new()
        ellipsoid = self.ellipsoid(array.shape, radius_a, radius_b, radius_c)
        inside = ellipsoid <= 1
        array[inside] = 1
        if RveInfo.low_rsa_resolution:
            d_vol = np.count_nonzero(array)*(2*RveInfo.bin_size)**3
        else:
            d_vol = np.count_nonzero(array) * (RveInfo.bin_size) ** 3
        return d_vol

    def convert_volume_2D(self, radius_a, radius_b):
        """Compute the volume for the given radii.
        Parameters :
        radius_a : Integer, radius along x-axis
        radius_b : Integer, radius along y-axis
        """
        array = self.gen_array_2d()
        ellipse = self.ellipse(radius_a, radius_b,  0, 0)
        inside = ellipse <= 1
        array[inside] = 1
        d_vol = np.count_nonzero(array)*RveInfo.bin_size**2
        RveInfo.LOGGER.info("Volume for the given radii: {}".format(d_vol))
        return d_vol

    def band_generator(self, band_array: np.array, center, bandwidth):
        """Creates a band of given bandwidth for given points in interval [step_half, box_size)
        with bin_size spacing along the axis.
        Parameters :
        xyz_grid : Array, list of points in interval [step_half, box_size) with bin_size spacing
        plane : String, default is 'xy'
        Bandidentifier will be -200 in rve_array
        """
        band_is_placed = False
        band_half = bandwidth / 2

        empty_array = band_array.copy()
        empty_array[empty_array == -200] = 0

        if RveInfo.band_orientation == 'xy':
            r = self.gen_grid()[2]
        elif RveInfo.band_orientation == 'yz':
            r = self.gen_grid()[0]
        elif RveInfo.band_orientation == 'xz':
            r = self.gen_grid()[1]
        else:
            RveInfo.LOGGER.error("Error: plane must be defined as xy, yz or xz! Default: xy")
            sys.exit(1)

        while not band_is_placed:

            # band_ center doesnt necessarily need to be an integer
            band_center = center
            print('center: ', band_center)
            left_bound = band_center - band_half
            right_bound = band_center + band_half
            empty_array[(r > left_bound) & (r < right_bound) & (band_array == 0)] = 1

            # get theoretical volume of current band and volume of bands that have previously been placed
            band_vol_0_theo = np.count_nonzero(empty_array == 1)
            rve_band_vol_old = np.count_nonzero(band_array == -200)

            # place current band
            band_array[(r >= left_bound) & (r <= right_bound) & (band_array == 0)] = -200

            # get total volume of all bands
            rve_band_vol_new = np.count_nonzero(band_array == -200)

            # compare real volume and theoretical volume of current band if bands are exactly on top of
            # each other band_vol_0_theo = 0 which must be avoided
            if ((rve_band_vol_old + band_vol_0_theo) == rve_band_vol_new) and not band_vol_0_theo == 0:
                band_is_placed = True
                RveInfo.LOGGER.info("Band generator - Bandwidth: {}, Left bound: {} and Right bound: {}"
                                    .format(bandwidth, left_bound, right_bound))

        return band_array

    def make_periodic_2D(self, points_array, ellipse_points, iterator) -> np.ndarray:
        points_array_mod = np.zeros(points_array.shape)
        points_array_mod[points_array == iterator] = iterator

        for i in range(1, 9):
            points_array_copy = np.zeros(points_array.shape)
            points_array_copy[(ellipse_points <= 1) & (points_array == -1 * i)] = -100 - i
            if i % 2 != 0:
                points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
            elif (i == 2) | (i == 6):
                points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
            else:
                points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
        return points_array_mod

    def make_periodic_3D_new(self, ellipse_array, nbins_x, nbins_y, n_bins_z) -> np.ndarray:
        if RveInfo.box_size_y is None and RveInfo.box_size_z is None:
            ellipse_array_periodic = np.roll(ellipse_array, (nbins_x, nbins_y, n_bins_z), axis=(0, 1, 2))
        elif RveInfo.box_size_y is not None and RveInfo.box_size_z is None:
            ellipse_array_periodic = np.roll(ellipse_array, (nbins_x, n_bins_z), axis=(0, 2))
        elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
            ellipse_array_periodic = np.roll(ellipse_array, (nbins_x, nbins_y), axis=(0, 1))
        elif RveInfo.box_size_y is not None and RveInfo.box_size_z is not None:
            ellipse_array_periodic = ellipse_array
        return ellipse_array_periodic


    # TODO: hier muss ne neue Funktion her mit dem rollen
    def make_periodic_3D(self, points_array, ellipse_points, iterator) -> np.ndarray:
        points_array_mod = np.zeros(points_array.shape)
        points_array_mod[points_array == iterator] = iterator
        t_0 = datetime.datetime.now()
        if RveInfo.box_size_y is None and RveInfo.box_size_z is None:
            #print("are we here")
            for i in range(1, 27):  # move points in x,y and z dir
                points_array_copy = np.zeros(points_array.shape)
                points_array_copy[(ellipse_points <= 1) & (points_array == -1 * i)] = -100 - i
                if (i == 1) | (i == 3) | (i == 7) | (i == 9) | \
                        (i == 18) | (i == 20) | (i == 24) | (i == 26):  # move points in x,y and z direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=2)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 10) | (i == 12) | (i == 15) | (i == 17):  # move points in x and y direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 4) | (i == 6) | (i == 21) | (i == 23):  # move points in x and z direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=2)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 2) | (i == 8) | (i == 19) | (i == 25):  # move points in y and z direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=2)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 13) | (i == 14):  # move points in x direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 11) | (i == 16):  # move points in y direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 5) | (i == 22):  # move points in z direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=2)
                    points_array_mod[points_array_copy == -100 - i] = iterator
        elif RveInfo.box_size_y is not None and RveInfo.box_size_z is None:
            for i in range(1, 27):  # move points in x,y and z dir
                points_array_copy = np.zeros(points_array.shape)
                points_array_copy[(ellipse_points <= 1) & (points_array == -1 * i)] = -100 - i
                if (i == 1) | (i == 3) | (i == 7) | (i == 9) | \
                        (i == 18) | (i == 20) | (i == 24) | (i == 26):  # move points in x,y and z direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=2)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 10) | (i == 12) | (i == 15) | (i == 17):  # move points in x and y direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 4) | (i == 6) | (i == 21) | (i == 23):  # move points in x and z direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=2)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 2) | (i == 8) | (i == 19) | (i == 25):  # move points in y and z direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=2)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 13) | (i == 14):  # move points in x direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 5) | (i == 22):  # move points in z direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=2)
                    points_array_mod[points_array_copy == -100 - i] = iterator

        elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
            for i in range(1, 27):  # move points in x,y and z dir
                points_array_copy = np.zeros(points_array.shape)
                points_array_copy[(ellipse_points <= 1) & (points_array == -1 * i)] = -100 - i
                if (i == 1) | (i == 3) | (i == 7) | (i == 9) | \
                        (i == 18) | (i == 20) | (i == 24) | (i == 26):  # move points in x,y and z direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=2)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 10) | (i == 12) | (i == 15) | (i == 17):  # move points in x and y direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 4) | (i == 6) | (i == 21) | (i == 23):  # move points in x and z direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=2)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 2) | (i == 8) | (i == 19) | (i == 25):  # move points in y and z direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=2)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 13) | (i == 14):  # move points in x direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=0)
                    points_array_mod[points_array_copy == -100 - i] = iterator
                elif (i == 11) | (i == 16):  # move points in y direction
                    points_array_copy = np.roll(points_array_copy, RveInfo.n_pts, axis=1)
                    points_array_mod[points_array_copy == -100 - i] = iterator

        elif RveInfo.box_size_y is not None and RveInfo.box_size_z is not None:
            for i in range(1, 27):  # move points in x,y and z dir
                points_array_copy = np.zeros(points_array.shape)
                points_array_copy[(ellipse_points <= 1) & (points_array == -1 * i)] = -100 - i
                points_array_mod[points_array_copy == -100 - i] = iterator
        time_elapse = datetime.datetime.now() - t_0
        if RveInfo.debug:
            RveInfo.LOGGER.info('time spent on periodicity for grain {}: {}'.format(iterator, time_elapse.total_seconds()))
        return points_array_mod

    def gen_boundaries_2D(self, points_array) -> np.ndarray:
        box_size = RveInfo.box_size
        x_grid, y_grid = self.gen_grid2d()
        points_array[np.where((x_grid > box_size) & (y_grid > box_size))] = -1
        points_array[(x_grid < box_size) & (y_grid > box_size)] = -2
        points_array[(x_grid < 0) & (y_grid > box_size)] = -3
        points_array[(x_grid < 0) & (y_grid < box_size)] = -4
        points_array[(x_grid > box_size) & (y_grid < box_size)] = -8
        points_array[(x_grid > box_size) & (y_grid < 0)] = -7
        points_array[(x_grid < box_size) & (y_grid < 0)] = -6
        points_array[(x_grid < 0) & (y_grid < 0)] = -5
        return points_array
    # TODO: wird in neuer Version glaub ich nicht notwendig sein
    def gen_boundaries_3D(self, points_array) -> np.ndarray:
        t_0 = datetime.datetime.now()
        box_size = RveInfo.box_size
        x_grid, y_grid, z_grid = self.gen_grid()

        """
        Each region around the RVE needs to be labled on order to move grainparts
        outside the rve_box to the correct position and make everything periodic
        the lables are shown below.
        It is higly recommended to not change anything here it will only destroy
        the periodicity

        z < 0
                ###########################
                #       #       #       #
                #   -7  #   -8  #   -9  # y > bs
                #       #       #       #
                ###########################
                #       #       #       #
                #   -4  #   -5  #   -6  # y > 0
                #       #       #       #
                ###########################
                #       #       #       #
          y     #   -1  #   -2  #   -3  # y < 0
          ^     #       #       #       #
          |__>x ###########################
                #  x<0  #  x>0  #  x>bs #

        z > 0
                ###########################
                #       #       #       #
                #  -15  #  -16  #  -17  # y > bs
                #       #       #       #
                ###########################
                #       #       #       #
                #  -13  #  RVE  #  -14  # y > 0
                #       #       #       #
                ###########################
                #       #       #       #
          y     #  -10  #  -11  #  -12  # y < 0
          ^     #       #       #       #
          |__>x ###########################
                #  x<0  #  x>0  #  x>bs #

        Z > box_size
                ###########################
                #       #       #       #
                #  -24  #  -25  #  -26  # y > bs
                #       #       #       #
                ###########################
                #       #       #       #
                #  -21  #  -22  #  -23  # y > 0
                #       #       #       #
                ###########################
                #       #       #       #
          y     #  -18  #  -19  #  -20  # y < 0
          ^     #       #       #       #
          |__>x ###########################    
                #  x<0  #  x>0  #  x>bs #
        
        Boxsizes have to be adjusted in case we are not modelling a cube
        """

        if RveInfo.box_size_y is None and RveInfo.box_size_z is None:
            # z < 0
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid < 0)] = -1
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid < 0)] = -2
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid < 0)] = -3

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid < 0)] = -4
            points_array[(x_grid > 0) & (y_grid > 0) & (z_grid < 0)] = -5
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid < 0)] = -6

            points_array[(x_grid < 0) & (y_grid > box_size) & (z_grid < 0)] = -7
            points_array[(x_grid > 0) & (y_grid > box_size) & (z_grid < 0)] = -8
            points_array[(x_grid > box_size) & (y_grid > box_size) & (z_grid < 0)] = -9

            # z > 0
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid > 0)] = -10
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid > 0)] = -11
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid > 0)] = -12

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid > 0)] = -13
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid > 0)] = -14

            points_array[(x_grid < 0) & (y_grid > box_size) & (z_grid > 0)] = -15
            points_array[(x_grid > 0) & (y_grid > box_size) & (z_grid > 0)] = -16
            points_array[(x_grid > box_size) & (y_grid > box_size) & (z_grid > 0)] = -17


            #Z > box_size
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid > box_size)] = -18
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid > box_size)] = -19
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid > box_size)] = -20

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid > box_size)] = -21
            points_array[(x_grid > 0) & (y_grid > 0) & (z_grid > box_size)] = -22
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid > box_size)] = -23

            points_array[(x_grid < 0) & (y_grid > box_size) & (z_grid > box_size)] = -24
            points_array[(x_grid > 0) & (y_grid > box_size) & (z_grid > box_size)] = -25
            points_array[(x_grid > box_size) & (y_grid > box_size) & (z_grid > box_size)] = -26


        elif RveInfo.box_size_y is not None and RveInfo.box_size_z is None:
            box_size_y = RveInfo.box_size_y
            # z < 0
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid < 0)] = -1
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid < 0)] = -2
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid < 0)] = -3

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid < 0)] = -4
            points_array[(x_grid > 0) & (y_grid > 0) & (z_grid < 0)] = -5
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid < 0)] = -6

            points_array[(x_grid < 0) & (y_grid > box_size_y) & (z_grid < 0)] = -7
            points_array[(x_grid > 0) & (y_grid > box_size_y) & (z_grid < 0)] = -8
            points_array[(x_grid > box_size) & (y_grid > box_size_y) & (z_grid < 0)] = -9

            # z > 0
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid > 0)] = -10
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid > 0)] = -11
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid > 0)] = -12

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid > 0)] = -13
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid > 0)] = -14

            points_array[(x_grid < 0) & (y_grid > box_size_y) & (z_grid > 0)] = -15
            points_array[(x_grid > 0) & (y_grid > box_size_y) & (z_grid > 0)] = -16
            points_array[(x_grid > box_size) & (y_grid > box_size_y) & (z_grid > 0)] = -17

            # Z > box_size
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid > box_size)] = -18
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid > box_size)] = -19
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid > box_size)] = -20

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid > box_size)] = -21
            points_array[(x_grid > 0) & (y_grid > 0) & (z_grid > box_size)] = -22
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid > box_size)] = -23

            points_array[(x_grid < 0) & (y_grid > box_size_y) & (z_grid > box_size)] = -24
            points_array[(x_grid > 0) & (y_grid > box_size_y) & (z_grid > box_size)] = -25
            points_array[(x_grid > box_size) & (y_grid > box_size_y) & (z_grid > box_size)] = -26

        elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
            box_size_z = RveInfo.box_size_z
            # z < 0
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid < 0)] = -1
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid < 0)] = -2
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid < 0)] = -3

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid < 0)] = -4
            points_array[(x_grid > 0) & (y_grid > 0) & (z_grid < 0)] = -5
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid < 0)] = -6

            points_array[(x_grid < 0) & (y_grid > box_size) & (z_grid < 0)] = -7
            points_array[(x_grid > 0) & (y_grid > box_size) & (z_grid < 0)] = -8
            points_array[(x_grid > box_size) & (y_grid > box_size) & (z_grid < 0)] = -9

            # z > 0
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid > 0)] = -10
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid > 0)] = -11
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid > 0)] = -12

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid > 0)] = -13
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid > 0)] = -14

            points_array[(x_grid < 0) & (y_grid > box_size) & (z_grid > 0)] = -15
            points_array[(x_grid > 0) & (y_grid > box_size) & (z_grid > 0)] = -16
            points_array[(x_grid > box_size) & (y_grid > box_size) & (z_grid > 0)] = -17

            # Z > box_size
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid > box_size_z)] = -18
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid > box_size_z)] = -19
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid > box_size_z)] = -20

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid > box_size_z)] = -21
            points_array[(x_grid > 0) & (y_grid > 0) & (z_grid > box_size_z)] = -22
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid > box_size_z)] = -23

            points_array[(x_grid < 0) & (y_grid > box_size) & (z_grid > box_size_z)] = -24
            points_array[(x_grid > 0) & (y_grid > box_size) & (z_grid > box_size_z)] = -25
            points_array[(x_grid > box_size) & (y_grid > box_size) & (z_grid > box_size_z)] = -26

        else:
            box_size_y = RveInfo.box_size_y
            box_size_z = RveInfo.box_size_z
            # z < 0
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid < 0)] = -1
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid < 0)] = -2
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid < 0)] = -3

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid < 0)] = -4
            points_array[(x_grid > 0) & (y_grid > 0) & (z_grid < 0)] = -5
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid < 0)] = -6

            points_array[(x_grid < 0) & (y_grid > box_size_y) & (z_grid < 0)] = -7
            points_array[(x_grid > 0) & (y_grid > box_size_y) & (z_grid < 0)] = -8
            points_array[(x_grid > box_size) & (y_grid > box_size_y) & (z_grid < 0)] = -9

            # z > 0
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid > 0)] = -10
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid > 0)] = -11
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid > 0)] = -12

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid > 0)] = -13
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid > 0)] = -14

            points_array[(x_grid < 0) & (y_grid > box_size_y) & (z_grid > 0)] = -15
            points_array[(x_grid > 0) & (y_grid > box_size_y) & (z_grid > 0)] = -16
            points_array[(x_grid > box_size) & (y_grid > box_size_y) & (z_grid > 0)] = -17

            # Z > box_size
            points_array[(x_grid < 0) & (y_grid < 0) & (z_grid > box_size_z)] = -18
            points_array[(x_grid > 0) & (y_grid < 0) & (z_grid > box_size_z)] = -19
            points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid > box_size_z)] = -20

            points_array[(x_grid < 0) & (y_grid > 0) & (z_grid > box_size_z)] = -21
            points_array[(x_grid > 0) & (y_grid > 0) & (z_grid > box_size_z)] = -22
            points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid > box_size_z)] = -23

            points_array[(x_grid < 0) & (y_grid > box_size_y) & (z_grid > box_size_z)] = -24
            points_array[(x_grid > 0) & (y_grid > box_size_y) & (z_grid > box_size_z)] = -25
            points_array[(x_grid > box_size) & (y_grid > box_size_y) & (z_grid > box_size_z)] = -26

        time_elapse = datetime.datetime.now() - t_0
        if RveInfo.debug:
            RveInfo.LOGGER.info('time spent on gen_boundaries: {}'.format(time_elapse.total_seconds()))
        return points_array

    def repair_periodicity_2D(self, rve_array: np.ndarray) -> pd.DataFrame:

        start1 = int(rve_array.shape[0] / 4)
        stop1 = int(rve_array.shape[0] / 4 + rve_array.shape[0] / 4 * 2)+1
        start2 = int(rve_array.shape[1] / 4)
        stop2 = int(rve_array.shape[1] / 4 + rve_array.shape[1] / 4 * 2)+1

        rve = rve_array[start1:stop1, start2:stop2]

        # define first boundary row/column with grainID Values of row and column #0
        rve[-1, :] = rve[0, :]
        rve[:, -1] = rve[:, 0]

        # load some variables
        box_size = RveInfo.box_size
        n_pts = RveInfo.n_pts

        # Transform np.array to coordinates
        xy = np.linspace(-box_size / 2, box_size + box_size / 2, 2 * RveInfo.n_pts, endpoint=True)
        x_grid, y_grid = np.meshgrid(xy, xy, indexing='ij')

        rve_x_idx, rve_y_idx = np.where(rve_array >= 1)
        boundary_x_idx, boundary_y_idx = np.where(rve_array < 0)

        rve_tuples = [*zip(rve_x_idx, rve_y_idx)]
        boundary_tuples = [*zip(boundary_x_idx, boundary_y_idx)]

        rve_x = [x_grid[rve_tuples_i[0]][rve_tuples_i[1]] for rve_tuples_i in rve_tuples]
        rve_y = [y_grid[rve_tuples_i[0]][rve_tuples_i[1]] for rve_tuples_i in rve_tuples]

        boundary_x = [x_grid[boundary_tuples_i[0]][boundary_tuples_i[1]] for boundary_tuples_i in boundary_tuples]
        boundary_y = [y_grid[boundary_tuples_i[0]][boundary_tuples_i[1]] for boundary_tuples_i in boundary_tuples]

        # generate pandas Dataframe of coordinates and grain IDs
        rve_dict = {'x': rve_x, 'y': rve_y, 'GrainID': rve_array[rve_array > 0]}
        rve = pd.DataFrame(rve_dict)
        rve['box_size'] = box_size
        rve['n_pts'] = n_pts

        return rve

    def repair_periodicity_3D_new(self, rve_array: np.ndarray):
        """this function is used to mirror the three masterfaces on the three slave faces of the rve
        in order to achieve exact periodicity"""
        # load some variables
        box_size = RveInfo.box_size

        if RveInfo.box_size_y is None and RveInfo.box_size_z is None:
            rve = np.zeros((rve_array.shape[0] + 1, rve_array.shape[1] + 1, rve_array.shape[2] + 1))
            rve[0:-1, 0:-1, 0:-1] = rve_array
            rve[-1, :, :] = rve[0, :, :]
            rve[:, -1, :] = rve[:, 0, :]
            rve[:, :, -1] = rve[:, :, 0]
            rve_x = np.linspace(0, RveInfo.box_size, rve_array.shape[0] + 1, endpoint=True)
            rve_y = np.linspace(0, RveInfo.box_size, rve_array.shape[1] + 1, endpoint=True)
            rve_z = np.linspace(0, RveInfo.box_size, rve_array.shape[2] + 1, endpoint=True)

        elif RveInfo.box_size_y is not None and RveInfo.box_size_z is None:
            rve = np.zeros((rve_array.shape[0] + 1, rve_array.shape[1], rve_array.shape[2] + 1))
            rve[0:-1, :, 0:-1] = rve_array
            rve[-1, :, :] = rve[0, :, :]
            rve[:, :, -1] = rve[:, :, 0]
            rve_x = np.linspace(0, RveInfo.box_size, rve_array.shape[0] + 1, endpoint=True)
            rve_y = np.linspace(0, RveInfo.box_size_y, rve_array.shape[1], endpoint=True)
            rve_z = np.linspace(0, RveInfo.box_size, rve_array.shape[2] + 1, endpoint=True)

        elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
            rve = np.zeros((rve_array.shape[0] + 1, rve_array.shape[2] + 1, rve_array.shape[2]))
            rve[0:-1, 0:-1, :] = rve_array
            rve[-1, :, :] = rve[0, :, :]
            rve[:, -1, :] = rve[:, 0, :]
            rve_x = np.linspace(0, RveInfo.box_size, rve_array.shape[0] + 1, endpoint=True)
            rve_y = np.linspace(0, RveInfo.box_size, rve_array.shape[1] + 1, endpoint=True)
            rve_z = np.linspace(0, RveInfo.box_size_z, rve_array.shape[2] , endpoint=True)

        else:
            rve = rve_array
            rve_x = np.linspace(0, RveInfo.box_size, rve_array.shape[0], endpoint=True)
            rve_y = np.linspace(0, RveInfo.box_size_y, rve_array.shape[1] , endpoint=True)
            rve_z = np.linspace(0, RveInfo.box_size_z, rve_array.shape[2] , endpoint=True)

        xx, yy, zz = np.meshgrid(rve_x, rve_y, rve_z, indexing='ij')
        rve_dict = {'x': xx.flatten(), 'y': yy.flatten(), 'z': zz.flatten(), 'GrainID': rve.flatten()}
        rve_df = pd.DataFrame(rve_dict)
        rve_df['box_size'] = box_size
        rve_df['n_pts'] = RveInfo.n_pts

        return rve_df, rve
    def repair_periodicity_3D(self, rve_array: np.ndarray):
        """this function is used to mirror the three masterfaces on the three slave faces of the rve
        in order to achieve exact periodicity"""
        # load some variables
        box_size = RveInfo.box_size
        n_pts = RveInfo.n_pts
        print('####repair####')
        print(np.unique(rve_array))
        print('####repair####')
        # define first boundary row/column with grainID Values of row and column #0
        if RveInfo.box_size_y is None and RveInfo.box_size_z is None:
            start1 = int(rve_array.shape[0] / 4)
            stop1 = int(rve_array.shape[0] / 4 + rve_array.shape[0] / 4 * 2)+1
            start2 = int(rve_array.shape[1] / 4)
            stop2 = int(rve_array.shape[1] / 4 + rve_array.shape[1] / 4 * 2)+1
            start3 = int(rve_array.shape[2] / 4)
            stop3 = int(rve_array.shape[2] / 4 + rve_array.shape[2] / 4 * 2)+1
            rve = rve_array[start1:stop1, start2:stop2, start3:stop3]

            rve[-1, :, :] = rve[0, :, :]
            rve[:, -1, :] = rve[:, 0, :]
            rve[:, :, -1] = rve[:, :, 0]
            rve_x = np.linspace(0, RveInfo.box_size, RveInfo.n_pts+1, endpoint=True)
            rve_y = np.linspace(0, RveInfo.box_size, RveInfo.n_pts+1, endpoint=True)
            rve_z = np.linspace(0, RveInfo.box_size, RveInfo.n_pts+1, endpoint=True)

        elif RveInfo.box_size_y is not None and RveInfo.box_size_z is None:
            start1 = int(rve_array.shape[0] / 4)
            stop1 = int(rve_array.shape[0] / 4 + rve_array.shape[0] / 4 * 2)+1
            start2 = int(rve_array.shape[1] / 4)
            stop2 = int(rve_array.shape[1] / 4 + rve_array.shape[1] / 4 * 2)
            start3 = int(rve_array.shape[2] / 4)
            stop3 = int(rve_array.shape[2] / 4 + rve_array.shape[2] / 4 * 2)+1
            rve = rve_array[start1:stop1, start2:stop2, start3:stop3]

            rve[-1, :, :] = rve[0, :, :]
            rve[:, :, -1] = rve[:, :, 0]

            rve_x = np.linspace(0, RveInfo.box_size, RveInfo.n_pts+1, endpoint=True)
            rve_y = np.linspace(0, RveInfo.box_size_y, RveInfo.n_pts_y, endpoint=True)
            rve_z = np.linspace(0, RveInfo.box_size, RveInfo.n_pts+1, endpoint=True)

        elif RveInfo.box_size_y is None and RveInfo.box_size_z is not None:
            start1 = int(rve_array.shape[0] / 4)
            stop1 = int(rve_array.shape[0] / 4 + rve_array.shape[0] / 4 * 2)+1
            start2 = int(rve_array.shape[1] / 4)
            stop2 = int(rve_array.shape[1] / 4 + rve_array.shape[1] / 4 * 2)+1
            start3 = int(rve_array.shape[2] / 4)
            stop3 = int(rve_array.shape[2] / 4 + rve_array.shape[2] / 4 * 2)
            rve = rve_array[start1:stop1, start2:stop2, start3:stop3]

            rve[-1, :, :] = rve[0, :, :]
            rve[:, -1, :] = rve[:, 0, :]

            rve_x = np.linspace(0, RveInfo.box_size, RveInfo.n_pts+1, endpoint=True)
            rve_y = np.linspace(0, RveInfo.box_size, RveInfo.n_pts+1, endpoint=True)
            rve_z = np.linspace(0, RveInfo.box_size_z, RveInfo.n_pts_z, endpoint=True)

        else:
            start1 = int(rve_array.shape[0] / 4)
            stop1 = int(rve_array.shape[0] / 4 + rve_array.shape[0] / 4 * 2)
            start2 = int(rve_array.shape[1] / 4)
            stop2 = int(rve_array.shape[1] / 4 + rve_array.shape[1] / 4 * 2)
            start3 = int(rve_array.shape[2] / 4)
            stop3 = int(rve_array.shape[2] / 4 + rve_array.shape[1] / 4 * 2)
            rve = rve_array[start1:stop1, start2:stop2, start3:stop3]

            rve_x = np.linspace(0, RveInfo.box_size, RveInfo.n_pts, endpoint=True)
            rve_y = np.linspace(0, RveInfo.box_size_y, RveInfo.n_pts_y, endpoint=True)
            rve_z = np.linspace(0, RveInfo.box_size_z, RveInfo.n_pts_z, endpoint=True)

        xx, yy, zz = np.meshgrid(rve_x, rve_y, rve_z, indexing='ij')
        rve_dict = {'x': xx.flatten(), 'y': yy.flatten(), 'z': zz.flatten(), 'GrainID': rve.flatten()}
        rve_df = pd.DataFrame(rve_dict)
        rve_df['box_size'] = box_size
        rve_df['n_pts'] = n_pts

        return rve_df, rve

    def ellipse(self, a, b, x_0, y_0, alpha=0):
        x_grid, y_grid = self.gen_grid2d()
        # without rotation
        """ellipse = np.sqrt((x_grid - x_0) ** 2 / (a ** 2) + (y_grid - y_0) ** 2 / (b ** 2))"""

        """ellipse = 1 / a ** 2 * ((self.x_grid - x_0) * np.cos(np.deg2rad(alpha))
                                        - (self.y_grid - y_0) * np.sin(np.deg2rad(alpha))) ** 2 +\
                  1 / b ** 2 * ((self.x_grid - x_0) * np.sin(np.deg2rad(alpha))
                                          + (self.y_grid - y_0) * np.cos(np.deg2rad(alpha))) ** 2"""

        ellipse = 1 / (a ** 2) * ((x_grid - x_0) * np.cos(np.deg2rad(alpha+RveInfo.slope_offset))
                                  + (y_grid - y_0) * np.sin(np.deg2rad(alpha+RveInfo.slope_offset))) ** 2 + \
                  1 / (b ** 2) * (-(x_grid - x_0) * np.sin(np.deg2rad(alpha+RveInfo.slope_offset))
                                  + (y_grid - y_0) * np.cos(np.deg2rad(alpha+RveInfo.slope_offset))) ** 2

        return ellipse

    def ellipsoid(self, shape, a, b, c, alpha=0):

        x_grid, y_grid, z_grid = self.gen_grid_new(shape)

        x_0 = int(float(RveInfo.box_size)/2)
        y_0 = int(float(RveInfo.box_size) / 2)
        z_0 = int(float(RveInfo.box_size) / 2)

        if RveInfo.box_size_y is not None:
            y_0 = int(float(RveInfo.box_size_y) / 2)
        if RveInfo.box_size_z is not None:
            z_0 = int(float(RveInfo.box_size_z) / 2)

        # rotation around z-axis
        ellipsoid = 1 / a ** 2 * ((x_grid - x_0) * np.cos(np.deg2rad(alpha+RveInfo.slope_offset)) +
                                  (y_grid - y_0) * np.sin(np.deg2rad(alpha+RveInfo.slope_offset))) ** 2 + \
                    1 / b ** 2 * (-(x_grid - x_0) * np.sin(np.deg2rad(alpha+RveInfo.slope_offset)) +
                                  (y_grid - y_0) * np.cos(np.deg2rad(alpha+RveInfo.slope_offset))) ** 2 + \
                    1 / c ** 2 * (z_grid - z_0) ** 2

        return ellipsoid

    def process_df(self, df, shrink_factor: float) -> pd.DataFrame:
        discrete_vol = list()
        df.reset_index(inplace=True, drop=True)
        df['GrainID'] = df.index
        a = df['a'].tolist()
        b = df['b'].tolist()
        c = df['c'].tolist()

        final_conti_volume = [(4 / 3 * a[i] * b[i] * c[i] * np.pi) for i in range(len(a))]

        for i in range(len(df)): # TODO: @Manuel, Max, Niklas: Ich glaub das ist super langsam
            discrete_vol.append(self.convert_volume_3D(df['a'][i], df['b'][i], df['c'][i]))
        df['final_discrete_volume'] = discrete_vol


        df['a'] = shrink_factor * df['a']
        df['b'] = shrink_factor * df['b']
        df['c'] = shrink_factor * df['c']

        df['final_conti_volume'] = final_conti_volume
        # Sortiert und resetet Index bereits
        df.sort_values(by='final_conti_volume', inplace=True, ascending=False)
        df.reset_index(inplace=True, drop=True)
        df['GrainID'] = df.index
        return df

    def process_df_2D(self, df, shrink_factor: float) -> pd.DataFrame:
        discrete_vol = list()
        df.reset_index(inplace=True, drop=True)
        df['GrainID'] = df.index
        a = df['a'].tolist()
        b = df['b'].tolist()

        final_conti_volume = [(a[i] * b[i] * np.pi) for i in range(len(a))]

        for i in range(len(df)):
            discrete_vol.append(self.convert_volume_2D(df['a'][i], df['b'][i]))
        df['final_discrete_volume'] = discrete_vol

        a_shrinked = [a_i * shrink_factor for a_i in a]
        b_shrinked = [b_i * shrink_factor for b_i in b]

        df['a'] = a_shrinked
        df['b'] = b_shrinked

        df['final_conti_volume'] = final_conti_volume
        # Sortiert und resetet Index bereits
        df.sort_values(by='final_conti_volume', inplace=True, ascending=False)
        df.reset_index(inplace=True, drop=True)
        df['GrainID'] = df.index
        return df

    def rearange_grain_ids_bands(self, bands_df, grains_df, rsa):
        """
        Rearanges the grains in an RVE in ascending order
        Needed because the band_grains are placed with ID lower -1000, but there is no need for this
        Band-Grains are added after the normal Grains and are labeled from 1 to xx
        """
        start = grains_df['GrainID'].max() + 1  # First occupied value

        rsa = rsa.copy()

        for i in bands_df['GrainID']:
            j = i + 1
            rsa[np.where(rsa == -(1000 + j))] = start + j
        return rsa

    def get_final_disc_vol_3D(self, grains_df: pd.DataFrame, rve: np.ndarray) -> pd.DataFrame:
        grains_df.sort_values(by=['GrainID'], inplace=True)
        disc_vols = np.zeros((1, grains_df.shape[0])).flatten().tolist()
        for i in range(len(grains_df)):
            disc_vols[i] = np.count_nonzero(rve == i+1) * RveInfo.bin_size**3

        grains_df['final_discrete_volume'] = disc_vols
        grains_df.sort_values(by='final_conti_volume', inplace=True, ascending=False)

        return grains_df

    def get_final_disc_vol_2D(self, grains_df: pd.DataFrame, rve: np.ndarray) -> pd.DataFrame:
        grains_df.sort_values(by=['GrainID'], inplace=True)
        disc_vols = np.zeros((1, grains_df.shape[0])).flatten().tolist()
        for i in range(len(grains_df)):
            disc_vols[i] = np.count_nonzero(rve == i+1) * RveInfo.bin_size**2

        grains_df['final_discrete_volume'] = disc_vols
        grains_df.sort_values(by='final_conti_volume', inplace=True, ascending=False)
        return grains_df

    def upsampling_rsa(self, rsa):
        repeat_factor = 2
        # repeat array in each direction
        resampled_array = np.repeat(rsa, repeat_factor, axis=0)
        resampled_array = np.repeat(resampled_array, repeat_factor, axis=1)
        resampled_array = np.repeat(resampled_array, repeat_factor, axis=2)
        return resampled_array

    def write_setup_file(self):
        members = [attr for attr in dir(RveInfo()) if
                   not callable(getattr(RveInfo(), attr)) and not attr.startswith("__")]

        setup_file = open(RveInfo.store_path + '/setup_file.txt', 'w')
        for member in members:
            setup_file.write(f'{member}: {str(RveInfo().__getattribute__(member))} \n')
        setup_file.close()
