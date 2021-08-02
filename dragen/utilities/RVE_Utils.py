import sys
import pandas as pd
import numpy as np
import logging
import datetime
from tkinter import messagebox


class RVEUtils:
    """Common Representative Volume Element (RVE) operations."""

    def __init__(self, box_size, n_pts, x_grid=None, y_grid=None, z_grid=None,
                 bandwidth=None, debug=False) -> None:
        self.logger = logging.getLogger("RVE-Gen")
        self.box_size = box_size
        self.n_pts = n_pts
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_grid = z_grid
        self.bandwidth = bandwidth
        self.debug = debug

        self.bin_size = box_size / n_pts
        self.step_half = self.bin_size / 2

    def gen_array(self) -> np.zeros:
        array = np.zeros((2 * self.n_pts, 2 * self.n_pts, 2 * self.n_pts))
        return array

    def gen_array_2D(self) -> np.zeros:
        array = np.zeros((2 * self.n_pts, 2 * self.n_pts))
        return array

    def gen_grid(self):
        xyz = np.linspace(-self.box_size / 2, self.box_size + self.box_size / 2, 2 * self.n_pts, endpoint=True)
        x_grid, y_grid, z_grid = np.meshgrid(xyz, xyz, xyz)
        return x_grid, y_grid, z_grid

    def gen_grid2D(self):
        xy = np.linspace(-self.box_size / 2, self.box_size + self.box_size / 2, 2 * self.n_pts, endpoint=True)
        x_grid, y_grid = np.meshgrid(xy, xy)
        return x_grid, y_grid

    def read_input(self, file_name, dimension) -> pd.DataFrame:
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
            print('ERROR: No "a" in given .csv-Inputfile!')
            messagebox.showinfo(message='No "a" in given .csv-Inputfile! RVE-Generation was canceled!', title='ERROR')
            self.logger.info('ERROR: No "a" in given .csv-Inputfile! RVE-Generation was canceled!')
            sys.exit()

        if 'b' in data.head(0) and data['b'].count() != 0:
            for rad in data['b']:
                radius_b.append(rad)
        else:
            radius_b = radius_a
            self.logger.info('No "b" in given .csv-Inputfile! Assumption: b = a')

        if 'c' in data.head(0) and data['c'].count() != 0:
            for rad in data['c']:
                radius_c.append(rad)
        else:
            radius_c = radius_a
            self.logger.info('No "c" in given .csv-Inputfile! Assumption: c = a')

        if 'alpha' in data.head(0) and data['alpha'].count() != 0:
            for ang in data['alpha']:
                alpha.append(ang)
        else:
            alpha = [0] * len(radius_a)
            self.logger.info('No "alpha" in given .csv-Inputfile! Assumption: alpha = 0, no rotation')

        if 'phi1' in data.head(0) and data['phi1'].count() != 0 and 'PHI' in data.head(0) and data['PHI'].count() != 0 \
                and 'phi2' in data.head(0) and data['phi2'].count() != 0:
            for tex in data['phi1']:
                tex_phi1.append(tex)
            for tex in data['PHI']:
                tex_PHI.append(tex)
            for tex in data['phi2']:
                tex_phi2.append(tex)
        else:
            self.logger.info(
                'No texture parameters (phi1, PHI, phi2) in given .csv-Inputfile! Assumption: random texture')
            i = 0
            while i < len(radius_a):
                tex_phi1.append(round((np.random.rand() * 360), 2))
                tex_PHI.append(round((np.random.rand() * 360), 2))
                tex_phi2.append(round((np.random.rand() * 360), 2))
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

    def sample_input_3D(self, data, bs) -> pd.DataFrame:
        max_volume = bs * bs * bs
        grain_vol = 0
        data = data.copy()
        inp_list = list()
        while grain_vol < max_volume:
            idx = np.random.randint(0, data.__len__())
            grain = data[["a", "b", "c", "alpha", "phi1", "PHI", "phi2"]].iloc[idx].tolist()
            data = data.drop(labels=data.index[idx], axis=0)
            vol = 4 / 3 * np.pi * grain[0] * grain[1] * grain[2]
            grain_vol += vol
            inp_list.append([grain[0], grain[1], grain[2], grain[3], grain[4], grain[5], grain[6]])
            if len(data) == 0:
                break

        # Del last if to big and more than one value: causes big errors for small dataframes
        #if grain_vol > max_volume and inp_list.__len__() > 1:
        #    inp_list.pop(-1)

        header = ["a", "b", "c", "alpha", "phi1", "PHI", "phi2"]
        df = pd.DataFrame(inp_list, columns=header)

        return df

    def sample_input_2D(self, data, bs) -> pd.DataFrame:
        max_volume = bs*bs
        grain_vol = 0
        data = data.copy()
        inp_list = list()
        while grain_vol < max_volume:
            idx = np.random.randint(0, data.__len__())
            grain = data[["a", "b", "alpha", "phi1", "PHI", "phi2"]].iloc[idx].tolist()
            data = data.drop(labels=data.index[idx], axis=0)
            vol = np.pi * grain[0] * grain[1]
            grain_vol += vol
            inp_list.append([grain[0], grain[1], grain[2], grain[3], grain[4], grain[5]])
            if len(data) == 0:
                break

        # Del last if to big and more than one value:
        if grain_vol >= max_volume and inp_list.__len__() > 1:
            inp_list.pop(-1)

        header = ["a", "b", "alpha", "phi1", "PHI", "phi2"]
        df = pd.DataFrame(inp_list, columns=header)

        return df

    def convert_volume_3D(self, radius_a, radius_b, radius_c):
        """Compute the volume for the given radii.
        Parameters :
        radius_a : Integer, radius along x-axis
        radius_b : Integer, radius along y-axis
        radius_b : Integer, radius along z-axis
        """
        array = self.gen_array()
        ellipsoid = self.ellipsoid(radius_a, radius_b, radius_c, 0, 0, 0)
        inside = ellipsoid <= 1
        array[inside] = 1
        d_vol = np.count_nonzero(array)*self.bin_size**3
        #self.logger.info("Volume for the given radii: {}".format(d_vol))
        return d_vol

    def convert_volume_2D(self, radius_a, radius_b):
        """Compute the volume for the given radii.
        Parameters :
        radius_a : Integer, radius along x-axis
        radius_b : Integer, radius along y-axis
        """
        array = self.gen_array_2D()
        ellipse = self.ellipse(radius_a, radius_b,  0, 0)
        inside = ellipse <= 1
        array[inside] = 1
        d_vol = np.count_nonzero(array)*self.bin_size**2
        self.logger.info("Volume for the given radii: {}".format(d_vol))
        return d_vol

    def band_generator(self, store_path, band_array: np.array, bandwidth=None, plane: str = 'xz', center=None):
        """Creates a band of given bandwidth for given points in interval [step_half, box_size)
        with bin_size spacing along the axis.
        Parameters :
        xyz_grid : Array, list of points in interval [step_half, box_size) with bin_size spacing
        plane : String, default is 'xy'
        Bandidentifier will be -200 in rve_array
        """
        band_is_placed = False
        if bandwidth == None:
            band_half = self.bandwidth / 2
        else:
            band_half = bandwidth/2

        empty_array = band_array.copy()
        band_array_backup = band_array.copy()
        empty_array[empty_array == -200] = 0

        if plane == 'xy':
            r = self.gen_grid()[2]
        elif plane == 'yz':
            r = self.gen_grid()[0]
        elif plane == 'xz':
            r = self.gen_grid()[1]
        else:
            self.logger.error("Error: plane must be defined as xy, yz or xz! Default: xy")
            sys.exit(1)

        while not band_is_placed:

            # band_ center doesnt necessarily need to be an integer
            band_center = int(self.bin_size + np.random.rand() * (self.box_size - self.bin_size))
            if center is not None:
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

            # get total volume of ALL bands
            rve_band_vol_new = np.count_nonzero(band_array == -200)

            # compare real volume and theoretical volume of current band if bands are exactly on top of
            # each other band_vol_0_theo = 0 which must be avoided
            # NEW: Bands should not intersect
            """print('Altes Volumen: ', rve_band_vol_old)
            print('Neues Volumen: ', rve_band_vol_new)
            print('Theoretisches Neues Vol: ', band_vol_0_theo)
            print(rve_band_vol_old + band_vol_0_theo)
            print(rve_band_vol_old + rve_band_vol_new)"""
            if ((rve_band_vol_old + band_vol_0_theo) == (rve_band_vol_new)) and not band_vol_0_theo == 0:
                band_is_placed = True
                self.logger.info("Band generator - Bandwidth: {}, Left bound: {} and Right bound: {}"
                                 .format(self.bandwidth, left_bound, right_bound))
                with open(store_path + '/rve.sta', 'a') as sta:
                    sta.writelines("Band generator - Bandwidth: {}, Left bound: {} and Right bound: {}\n"
                                   .format(bandwidth, left_bound, right_bound))
            else:
                band_array = band_array_backup.copy()
                empty_array = band_array.copy()


        return band_array

    def make_periodic_2D(self, points_array, ellipse_points, iterator) -> np.ndarray:
        points_array_mod = np.zeros(points_array.shape)
        points_array_mod[points_array == iterator] = iterator

        for i in range(1, 9):
            points_array_copy = np.zeros(points_array.shape)
            points_array_copy[(ellipse_points <= 1) & (points_array == -1 * i)] = -100 - i
            if i % 2 != 0:
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
            elif (i == 2) | (i == 6):
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
            else:
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
        return points_array_mod

    def make_periodic_3D(self, points_array, ellipse_points, iterator) -> np.ndarray:
        points_array_mod = np.zeros(points_array.shape)
        points_array_mod[points_array == iterator] = iterator
        t_0 = datetime.datetime.now()
        for i in range(1, 27):  # move points in x,y and z dir
            points_array_copy = np.zeros(points_array.shape)
            points_array_copy[(ellipse_points <= 1) & (points_array == -1 * i)] = -100 - i
            if (i == 1) | (i == 3) | (i == 7) | (i == 9) | \
                    (i == 18) | (i == 20) | (i == 24) | (i == 26):  # move points in x,y and z direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=2)
                points_array_mod[points_array_copy == -100 - i] = iterator
            elif (i == 10) | (i == 12) | (i == 15) | (i == 17):  # move points in x and y direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_mod[points_array_copy == -100 - i] = iterator
            elif (i == 4) | (i == 6) | (i == 21) | (i == 23):  # move points in x and z direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=2)
                points_array_mod[points_array_copy == -100 - i] = iterator
            elif (i == 2) | (i == 8) | (i == 19) | (i == 25):  # move points in y and z direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=2)
                points_array_mod[points_array_copy == -100 - i] = iterator
            elif (i == 13) | (i == 14):  # move points in x direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_mod[points_array_copy == -100 - i] = iterator
            elif (i == 11) | (i == 16):  # move points in y direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_mod[points_array_copy == -100 - i] = iterator
            elif (i == 5) | (i == 22):  # move points in z direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=2)
                points_array_mod[points_array_copy == -100 - i] = iterator
        time_elapse = datetime.datetime.now() - t_0
        if self.debug:
            self.logger.info('time spent on periodicity for grain {}: {}'.format(iterator, time_elapse.total_seconds()))
        return points_array_mod

    def gen_boundaries_2D(self, points_array) -> np.ndarray:
        box_size = self.box_size
        x_grid, y_grid = self.gen_grid2D()
        points_array[np.where((x_grid > box_size) & (y_grid > box_size))] = -1
        points_array[(x_grid < box_size) & (y_grid > box_size)] = -2
        points_array[(x_grid < 0) & (y_grid > box_size)] = -3
        points_array[(x_grid < 0) & (y_grid < box_size)] = -4
        points_array[(x_grid > box_size) & (y_grid < box_size)] = -8
        points_array[(x_grid > box_size) & (y_grid < 0)] = -7
        points_array[(x_grid < box_size) & (y_grid < 0)] = -6
        points_array[(x_grid < 0) & (y_grid < 0)] = -5
        return points_array

    def gen_boundaries_3D(self, points_array) -> np.ndarray:
        t_0 = datetime.datetime.now()
        box_size = self.box_size
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
        """

        points_array[(x_grid < 0) & (y_grid < 0) & (z_grid < 0)] = -1
        points_array[(x_grid > 0) & (y_grid < 0) & (z_grid < 0)] = -2
        points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid < 0)] = -3

        points_array[(x_grid < 0) & (y_grid > 0) & (z_grid < 0)] = -4
        points_array[(x_grid > 0) & (y_grid > 0) & (z_grid < 0)] = -5
        points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid < 0)] = -6

        points_array[(x_grid < 0) & (y_grid > box_size) & (z_grid < 0)] = -7
        points_array[(x_grid > 0) & (y_grid > box_size) & (z_grid < 0)] = -8
        points_array[(x_grid > box_size) & (y_grid > box_size) & (z_grid < 0)] = -9

        """
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
        """
        points_array[(x_grid < 0) & (y_grid < 0) & (z_grid > 0)] = -10
        points_array[(x_grid > 0) & (y_grid < 0) & (z_grid > 0)] = -11
        points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid > 0)] = -12

        points_array[(x_grid < 0) & (y_grid > 0) & (z_grid > 0)] = -13
        points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid > 0)] = -14

        points_array[(x_grid < 0) & (y_grid > box_size) & (z_grid > 0)] = -15
        points_array[(x_grid > 0) & (y_grid > box_size) & (z_grid > 0)] = -16
        points_array[(x_grid > box_size) & (y_grid > box_size) & (z_grid > 0)] = -17

        """
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

        """
        points_array[(x_grid < 0) & (y_grid < 0) & (z_grid > box_size)] = -18
        points_array[(x_grid > 0) & (y_grid < 0) & (z_grid > box_size)] = -19
        points_array[(x_grid > box_size) & (y_grid < 0) & (z_grid > box_size)] = -20

        points_array[(x_grid < 0) & (y_grid > 0) & (z_grid > box_size)] = -21
        points_array[(x_grid > 0) & (y_grid > 0) & (z_grid > box_size)] = -22
        points_array[(x_grid > box_size) & (y_grid > 0) & (z_grid > box_size)] = -23

        points_array[(x_grid < 0) & (y_grid > box_size) & (z_grid > box_size)] = -24
        points_array[(x_grid > 0) & (y_grid > box_size) & (z_grid > box_size)] = -25
        points_array[(x_grid > box_size) & (y_grid > box_size) & (z_grid > box_size)] = -26
        time_elapse = datetime.datetime.now() - t_0
        if self.debug:
            self.logger.info('time spent on gen_boundaries: {}'.format(time_elapse.total_seconds()))
        return points_array

    def repair_periodicity_2D(self, rve_array: np.ndarray) -> pd.DataFrame:

        # load some variables
        box_size = self.box_size
        n_pts = self.n_pts

        # Transform np.array to coordinates
        xy = np.linspace(-box_size / 2, box_size + box_size / 2, 2 * self.n_pts, endpoint=True)
        x_grid, y_grid = np.meshgrid(xy, xy)

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

        boundary_dict = {'x': boundary_x, 'y': boundary_y, 'GrainID': rve_array[rve_array < 0]}
        boundary = pd.DataFrame(boundary_dict)

        # Extract points that are supposed to be added to the rve
        new_max_x = min(boundary[boundary['GrainID'] == -1].x)
        new_max_y = min(boundary[boundary['GrainID'] == -2].y)

        additional_pts = boundary[(boundary['GrainID'] == -1) |
                                  (boundary['GrainID'] == -2) |
                                  (boundary['GrainID'] == -8)].copy()

        drop_idx = additional_pts[(additional_pts['GrainID'] == -1) &
                                  ((additional_pts['x'] > new_max_x) |
                                   (additional_pts['y'] > new_max_y))].index
        additional_pts.drop(drop_idx, inplace=True)

        drop_idx = additional_pts[((additional_pts['GrainID'] == -2) &
                                   (additional_pts['y'] > new_max_y))].index
        additional_pts.drop(drop_idx, inplace=True)

        drop_idx = additional_pts[((additional_pts['GrainID'] == -8) &
                                   (additional_pts['x'] > new_max_x))].index
        additional_pts.drop(drop_idx, inplace=True)

        rve = pd.concat([rve, additional_pts])

        max_x = max(rve.x)
        min_x = min(rve.x)
        max_y = max(rve.y)
        min_y = min(rve.y)

        # fixing grain IDs of corners
        rve_corners = rve.loc[((rve['x'] == min_x) & (rve['y'] == min_y)) |
                              ((rve['x'] == max_x) & (rve['y'] == max_y)) |
                              ((rve['x'] == min_x) & (rve['y'] == max_y)) |
                              ((rve['x'] == max_x) & (rve['y'] == min_y))].copy()

        cornersGrainID = rve_corners[(rve_corners['x'] == min_x) & (rve_corners['y'] == min_y)].GrainID.values

        rve.loc[rve_corners.index, 'GrainID'] = cornersGrainID

        # fixing grain IDs of Edges
        rve_edges = rve.loc[(rve['x'] == max_x) | (rve['x'] == min_x) |
                            (rve['y'] == max_y) | (rve['y'] == min_y)].copy()
        # Left Edge
        e_left = rve_edges.loc[rve_edges['x'] == min_x].copy()
        # Bottom Edge
        e_bottom = rve_edges.loc[rve_edges['y'] == min_y].copy()
        # Right Edge
        e_right = rve_edges.loc[rve_edges['x'] == max_x].copy()
        # Top Edge
        e_top = rve_edges.loc[rve_edges['y'] == max_y].copy()

        rve.loc[e_right.index, 'GrainID'] = e_left.GrainID.values
        rve.loc[e_top.index, 'GrainID'] = e_bottom.GrainID.values

        return rve

    def repair_periodicity_3D(self, rve_array: np.ndarray) -> pd.DataFrame:
        """this function is used to mirror the three masterfaces on the three slave faces of the rve
        in order to achieve exact periodicity"""
        # load some variables
        box_size = self.box_size
        n_pts = self.n_pts

        # Transform np.array to coordinates
        xyz = np.linspace(-box_size / 2, box_size + box_size / 2, 2 * self.n_pts, endpoint=True)
        x_grid, y_grid, z_grid = np.meshgrid(xyz, xyz, xyz)

        rve_x_idx, rve_y_idx, rve_z_idx = np.where(
            (rve_array > 0) | (rve_array == -200) | (rve_array < -200))  # Added for the inclusions
        boundary_x_idx, boundary_y_idx, boundary_z_idx = np.where(
            (rve_array < 0) & (rve_array > -200))  # Zwischen -200 und 0

        rve_tuples = [*zip(rve_x_idx, rve_y_idx, rve_z_idx)]
        boundary_tuples = [*zip(boundary_x_idx, boundary_y_idx, boundary_z_idx)]

        rve_x = [x_grid[rve_tuples_i[0]][rve_tuples_i[1]][rve_tuples_i[2]] for rve_tuples_i in rve_tuples]
        rve_y = [y_grid[rve_tuples_i[0]][rve_tuples_i[1]][rve_tuples_i[2]] for rve_tuples_i in rve_tuples]
        rve_z = [z_grid[rve_tuples_i[0]][rve_tuples_i[1]][rve_tuples_i[2]] for rve_tuples_i in rve_tuples]

        boundary_x = [x_grid[boundary_tuples_i[0]][boundary_tuples_i[1]][boundary_tuples_i[2]]
                      for boundary_tuples_i in boundary_tuples]
        boundary_y = [y_grid[boundary_tuples_i[0]][boundary_tuples_i[1]][boundary_tuples_i[2]]
                      for boundary_tuples_i in boundary_tuples]
        boundary_z = [z_grid[boundary_tuples_i[0]][boundary_tuples_i[1]][boundary_tuples_i[2]]
                      for boundary_tuples_i in boundary_tuples]

        # generate pandas Dataframe of coordinates and grain IDs
        rve_dict = {'x': rve_x, 'y': rve_y, 'z': rve_z, 'GrainID': rve_array[
            (rve_array > 0) | (rve_array == -200) | (rve_array < -200)]}  # Also smaller -200 for inclusions
        rve = pd.DataFrame(rve_dict)
        rve['box_size'] = box_size
        rve['n_pts'] = n_pts

        boundary_dict = {'x': boundary_x, 'y': boundary_y, 'z': boundary_z,
                         'GrainID': rve_array[(rve_array < 0) & (rve_array > -200)]}
        boundary = pd.DataFrame(boundary_dict)

        # Extract points that are supposed to be added to the rve
        new_max_x = min(boundary[boundary['GrainID'] == -26].x)  # several numbers are possible -26 just works
        new_max_y = min(boundary[boundary['GrainID'] == -26].y)  # for all three directions
        new_max_z = min(boundary[boundary['GrainID'] == -26].z)  # therefore it's the most intuitive choice

        # these boundaries contain the new cornerpoints:
        additional_corner_pts = boundary[(boundary['GrainID'] == -26)  # top_front_right (-26)
        ].copy()

        # these boundaries contain the new edgepoints:
        additional_edge_pts = boundary[(boundary['GrainID'] == -17) |  # bottom_to_top_front_right (-17)
                                       (boundary['GrainID'] == -23) |  # left_to_right_top_front (-23)
                                       (boundary['GrainID'] == -25)  # rear_to_front_top_right (-25)
                                       ].copy()

        # these boundaries contain the new edgepoints:
        additional_face_pts = boundary[(boundary['GrainID'] == -14) |  # left_to_right_bottom_front (-6)
                                       (boundary['GrainID'] == -16) |  # rear_to_front_bottom_right (-8)
                                       (boundary['GrainID'] == -22)  # bottom_to_top_front_left (-12)

                                       ].copy()

        drop_idx_corners = additional_corner_pts[(additional_corner_pts['x'] != new_max_x) |
                                                 (additional_corner_pts['y'] != new_max_y) |
                                                 (additional_corner_pts['z'] != new_max_z)].index

        additional_corner_pts.drop(drop_idx_corners, inplace=True)

        drop_idx_edges = additional_edge_pts[((additional_edge_pts['GrainID'] == -17) &
                                              ((additional_edge_pts['x'] > new_max_x) |
                                               (additional_edge_pts['y'] > new_max_y))) |
                                             ((additional_edge_pts['GrainID'] == -23) &
                                              ((additional_edge_pts['x'] > new_max_x) |
                                               (additional_edge_pts['z'] > new_max_z))) |
                                             ((additional_edge_pts['GrainID'] == -25) &
                                              ((additional_edge_pts['y'] > new_max_y) |
                                               (additional_edge_pts['z'] > new_max_z)))].index

        additional_edge_pts.drop(drop_idx_edges, inplace=True)

        drop_idx_faces = additional_face_pts[((additional_face_pts['GrainID'] == -14) &
                                              (additional_face_pts['x'] > new_max_x)) |
                                             ((additional_face_pts['GrainID'] == -16) &
                                              (additional_face_pts['y'] > new_max_y)) |
                                             ((additional_face_pts['GrainID'] == -22) &
                                              (additional_face_pts['z'] > new_max_z))].index
        additional_face_pts.drop(drop_idx_faces, inplace=True)

        rve = pd.concat([rve, additional_face_pts, additional_edge_pts, additional_corner_pts])

        max_x = max(rve.x)
        min_x = min(rve.x)
        max_y = max(rve.y)
        min_y = min(rve.y)
        max_z = max(rve.z)
        min_z = min(rve.z)

        # fixing faces
        rve_faces = rve.loc[(((rve['x'] == max_x) | (rve['x'] == min_x)) &
                             ((rve['y'] != max_y) & (rve['y'] != min_y)) &
                             ((rve['z'] != max_z) & (rve['z'] != min_z))) |

                            (((rve['x'] != max_x) & (rve['x'] != min_x)) &
                             ((rve['y'] != max_y) & (rve['y'] != min_y)) &
                             ((rve['z'] == max_z) | (rve['z'] == min_z))) |

                            (((rve['x'] != max_x) & (rve['x'] != min_x)) &
                             ((rve['y'] == max_y) | (rve['y'] == min_y)) &
                             ((rve['z'] != max_z) & (rve['z'] != min_z)))].copy()

        # front set
        RearSet = rve_faces.loc[rve_faces['x'] == min_x].copy()
        # rear set
        FrontSet = rve_faces.loc[rve_faces['x'] == max_x].copy()
        # left set
        LeftSet = rve_faces.loc[rve_faces['y'] == min_y].copy()
        # right set
        RightSet = rve_faces.loc[rve_faces['y'] == max_y].copy()
        # bottom set
        BottomSet = rve_faces.loc[rve_faces['z'] == min_z].copy()
        # top set
        TopSet = rve_faces.loc[rve_faces['z'] == max_z].copy()

        rve.loc[RightSet.index, 'GrainID'] = LeftSet.GrainID.values
        rve.loc[TopSet.index, 'GrainID'] = BottomSet.GrainID.values
        rve.loc[FrontSet.index, 'GrainID'] = RearSet.GrainID.values

        rve_hull = rve.loc[(rve.x == min_x) | (rve.y == min_y) | (rve.z == min_z) |
                           (rve.x == max_x) | (rve.y == max_y) | (rve.z == max_z)]

        # fixing Edges
        rve_edges = rve.loc[(
                                    ((rve['x'] == min_x) & (rve['z'] == min_z)) |
                                    ((rve['x'] == max_x) & (rve['z'] == min_z)) |
                                    ((rve['x'] == min_x) & (rve['z'] == max_z)) |
                                    ((rve['x'] == max_x) & (rve['z'] == max_z))) |
                            (
                                    ((rve['y'] == min_y) & (rve['z'] == min_z)) |
                                    ((rve['y'] == max_y) & (rve['z'] == min_z)) |
                                    ((rve['y'] == min_y) & (rve['z'] == max_z)) |
                                    ((rve['y'] == max_y) & (rve['z'] == max_z))) |
                            (
                                    ((rve['x'] == min_x) & (rve['y'] == min_y)) |
                                    ((rve['x'] == max_x) & (rve['y'] == min_y)) |
                                    ((rve['x'] == min_x) & (rve['y'] == max_y)) |
                                    ((rve['x'] == max_x) & (rve['y'] == max_y)))].copy()

        # bottom_to_top_rear_left is mirrored on bottom_to_top_front_left,
        # bottom_to_top_front_right and bottom_to_top_rear_right
        bottom_to_top_rear_left = rve_edges.loc[(rve_edges['x'] == min_x) & (rve_edges['y'] == min_y)].copy()
        bottom_to_top_front_left = rve_edges.loc[(rve_edges['x'] == max_x) & (rve_edges['y'] == min_y)].copy()
        bottom_to_top_front_right = rve_edges.loc[(rve_edges['x'] == max_x) & (rve_edges['y'] == max_y)].copy()
        bottom_to_top_rear_right = rve_edges.loc[(rve_edges['x'] == min_x) & (rve_edges['y'] == max_y)].copy()

        rve.loc[bottom_to_top_front_left.index, 'GrainID'] = bottom_to_top_rear_left.GrainID.values
        rve.loc[bottom_to_top_front_right.index, 'GrainID'] = bottom_to_top_rear_left.GrainID.values
        rve.loc[bottom_to_top_rear_right.index, 'GrainID'] = bottom_to_top_rear_left.GrainID.values

        # left_to_right_bottom_rear is mirrored on left_to_right_top_rear,
        # left_to_right_top_front and left_to_right_bottom_front
        left_to_right_bottom_rear = rve_edges.loc[(rve_edges['x'] == min_x) & (rve_edges['z'] == min_z)].copy()
        left_to_right_top_rear = rve_edges.loc[(rve_edges['x'] == min_x) & (rve_edges['z'] == max_z)].copy()
        left_to_right_top_front = rve_edges.loc[(rve_edges['x'] == max_x) & (rve_edges['z'] == max_z)].copy()
        left_to_right_bottom_front = rve_edges.loc[(rve_edges['x'] == max_x) & (rve_edges['z'] == min_z)].copy()

        rve.loc[left_to_right_top_rear.index, 'GrainID'] = left_to_right_bottom_rear.GrainID.values
        rve.loc[left_to_right_top_front.index, 'GrainID'] = left_to_right_bottom_rear.GrainID.values
        rve.loc[left_to_right_bottom_front.index, 'GrainID'] = left_to_right_bottom_rear.GrainID.values

        # rear_to_front_bottom_left is mirrored on rear_to_front_top_left,
        # rear_to_front_top_right and rear_to_front_bottom_right
        rear_to_front_bottom_left = rve_edges.loc[(rve_edges['y'] == min_y) & (rve_edges['z'] == min_z)].copy()
        rear_to_front_top_left = rve_edges.loc[(rve_edges['y'] == min_y) & (rve_edges['z'] == max_z)].copy()
        rear_to_front_top_right = rve_edges.loc[(rve_edges['y'] == max_y) & (rve_edges['z'] == max_z)].copy()
        rear_to_front_bottom_right = rve_edges.loc[(rve_edges['y'] == max_y) & (rve_edges['z'] == min_z)].copy()

        rve.loc[rear_to_front_top_left.index, 'GrainID'] = rear_to_front_bottom_left.GrainID.values
        rve.loc[rear_to_front_top_right.index, 'GrainID'] = rear_to_front_bottom_left.GrainID.values
        rve.loc[rear_to_front_bottom_right.index, 'GrainID'] = rear_to_front_bottom_left.GrainID.values

        # fixing corners
        corner1 = rve.loc[(rve['x'] == min_x) & (rve['y'] == min_y) & (rve['z'] == min_z)]
        corners = rve.loc[((rve['x'] == min_x) & (rve['y'] == min_y) & (rve['z'] == min_z)) |
                          ((rve['x'] == min_x) & (rve['y'] == max_y) & (rve['z'] == min_z)) |
                          ((rve['x'] == max_x) & (rve['y'] == min_y) & (rve['z'] == min_z)) |
                          ((rve['x'] == max_x) & (rve['y'] == max_y) & (rve['z'] == min_z)) |
                          ((rve['x'] == min_x) & (rve['y'] == min_y) & (rve['z'] == max_z)) |
                          ((rve['x'] == min_x) & (rve['y'] == max_y) & (rve['z'] == max_z)) |
                          ((rve['x'] == max_x) & (rve['y'] == min_y) & (rve['z'] == max_z)) |
                          ((rve['x'] == max_x) & (rve['y'] == max_y) & (rve['z'] == max_z))]
        rve.loc[corners.index, 'GrainID'] = corner1.GrainID.values
        return rve

    def ellipse(self, a, b, x_0, y_0, alpha=0):
        x_grid, y_grid = self.gen_grid2D()
        # without rotation
        """ellipse = np.sqrt((x_grid - x_0) ** 2 / (a ** 2) + (y_grid - y_0) ** 2 / (b ** 2))"""

        """ellipse = 1 / a ** 2 * ((self.x_grid - x_0) * np.cos(np.deg2rad(alpha))
                                        - (self.y_grid - y_0) * np.sin(np.deg2rad(alpha))) ** 2 +\
                  1 / b ** 2 * ((self.x_grid - x_0) * np.sin(np.deg2rad(alpha))
                                          + (self.y_grid - y_0) * np.cos(np.deg2rad(alpha))) ** 2"""

        ellipse = 1 / (a ** 2) * ((x_grid - x_0) * np.cos(np.deg2rad(alpha))
                                  + (y_grid - y_0) * np.sin(np.deg2rad(alpha))) ** 2 + \
                  1 / (b ** 2) * (-(x_grid - x_0) * np.sin(np.deg2rad(alpha))
                                  + (y_grid - y_0) * np.cos(np.deg2rad(alpha))) ** 2

        return ellipse

    def ellipsoid(self, a, b, c, x_0, y_0, z_0, alpha=0):

        x_grid, y_grid, z_grid = self.gen_grid()
        # rotation around x-axis
        """ellipsoid = 1/a**2 * (self.x_grid - x_0) ** 2 + \
                    1/b**2 * ((self.y_grid - y_0) * np.cos(np.deg2rad(slope)) -
                                 (self.z_grid - z_0) * np.sin(np.deg2rad(slope))) ** 2 + \
                    1/c**2 * ((self.y_grid - y_0) * np.sin(np.deg2rad(slope)) +
                                 (self.z_grid - z_0) * np.cos(np.deg2rad(slope))) ** 2"""

        # rotation around y-axis; with a=c no influence
        """ellipsoid = 1/a**2 * ((self.x_grid - x_0) * np.cos(np.deg2rad(slope)) +
                              (self.z_grid - z_0) * np.sin(np.deg2rad(slope))) ** 2 + \
                    1/b**2 * (self.y_grid - y_0) ** 2 + \
                    1/c**2 * ((self.x_grid - x_0) * -np.sin(np.deg2rad(slope)) +
                              (self.z_grid - z_0) * np.cos(np.deg2rad(slope))) ** 2"""

        # rotation around z-axis
        ellipsoid = 1 / a ** 2 * ((x_grid - x_0) * np.cos(np.deg2rad(alpha)) +
                                  (y_grid - y_0) * np.sin(np.deg2rad(alpha))) ** 2 + \
                    1 / b ** 2 * (-(x_grid - x_0) * np.sin(np.deg2rad(alpha)) +
                                  (y_grid - y_0) * np.cos(np.deg2rad(alpha))) ** 2 + \
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

        for i in range(len(df)):
            discrete_vol.append(self.convert_volume_3D(df['a'][i], df['b'][i], df['c'][i]))
        df['final_discrete_volume'] = discrete_vol

        a_shrinked = [a_i * shrink_factor for a_i in a]
        b_shrinked = [b_i * shrink_factor for b_i in b]
        c_shrinked = [c_i * shrink_factor for c_i in c]

        df['a'] = a_shrinked
        df['b'] = b_shrinked
        df['c'] = c_shrinked
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
        print(grains_df)

        print(bands_df)
        rsa = rsa.copy()
        print(start)
        for i in bands_df['GrainID']:
            j = i + 1
            rsa[np.where(rsa == -(1000 + j))] = start + j

        print(np.asarray(np.unique(rsa, return_counts=True)).T)
        return rsa

    def get_final_disc_vol_3D(self, grains_df: pd.DataFrame, rve: np.ndarray) -> pd.DataFrame:
        grains_df.sort_values(by=['GrainID'], inplace=True)
        disc_vols = np.zeros((1, grains_df.shape[0])).flatten().tolist()
        for i in range(len(grains_df)):
            #grainID = grains_df.GrainID[i]

            disc_vols[i] = np.count_nonzero(rve == i+1) * self.bin_size**3

        grains_df['final_discrete_volume'] = disc_vols
        grains_df.sort_values(by='final_conti_volume', inplace=True, ascending=False)

        return grains_df




