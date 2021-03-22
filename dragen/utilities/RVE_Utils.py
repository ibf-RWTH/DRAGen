import sys
import pandas as pd
import numpy as np
import logging
import datetime


class RVEUtils:
    """Common Representative Volume Element (RVE) operations."""

    def __init__(self, box_size, n_pts, x_grid=None, y_grid=None, z_grid=None, bandwidth=0, debug=False):
        self.logger = logging.getLogger("RVE-Gen")
        self.box_size = box_size
        self.n_pts = n_pts
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_grid = z_grid
        self.bandwidth = bandwidth
        self.debug = debug

        self.step_size = box_size / n_pts
        self.step_half = self.step_size / 2

    def read_input(self, file_name, dimension):
        """Reads the given input file and returns the volume along with radii list.
        Parameter :
        file_name : String, name of the input file
        """
        data = pd.read_csv(file_name)
        radius_a, radius_b, radius_c = ([] for i in range(3))
        if 'a' in data.head(0):
            for rad in data['a']:
                radius_a.append(rad)
        if 'b' in data.head(0):
            for rad in data['b']:
                radius_b.append(rad)
        if 'c' in data.head(0):
            for rad in data['c']:
                radius_c.append(rad)
        if dimension == 3:
            return radius_a, radius_b, radius_c
        elif dimension == 2:
            return radius_a, radius_b

    def convert_volume(self, radius_a, radius_b, radius_c):
        """Compute the volume for the given radii.
        Parameters :
        radius_a : Integer, radius along x-axis
        radius_b : Integer, radius along y-axis
        radius_b : Integer, radius along z-axis
        """
        grid = np.around(np.arange(-self.box_size + self.step_half, self.box_size, self.step_size))
        min_grid = min([n for n in grid if n > 0])
        x0 = list(grid).index(min_grid)
        grainx, grainy, grainz = np.meshgrid(grid, grid, grid)
        A0 = (1. / radius_a) ** 2
        B0 = (1. / radius_b) ** 2
        C0 = (1. / radius_c) ** 2
        r = np.sqrt(A0 * (grainx - grid[x0]) ** 2 + B0 * (grainy - grid[x0]) ** 2 + C0 * (grainz - grid[x0]) ** 2)
        inside = r <= 1
        self.logger.info("Volume for the given radii: {}".format(len(grainx[inside])))
        return len(grainx[inside])

    def periodicity_RSA(self, coordinate, points):
        """Compute the list of periodic coordinates for the given grain coordinates and list of points.
        Parameters :
        coordinate : List, list of grain coordinates
        points : Array, list of points in interval [step_half-box_size, box_size*2) with step_size spacing
        """
        points = list(points)
        periodic_coord = []
        for j, point in enumerate(coordinate):
            if point > self.box_size:
                idx = points.index(point)
                periodic_point = points[idx - self.n_pts]

            elif point < self.step_half:
                idx = points.index(point)
                periodic_point = points[idx + self.n_pts]

            else:
                periodic_point = point
            periodic_coord.append(periodic_point)

        return periodic_coord

    def periodicity_DT(self, coordinate, points):
        """Computes the list of periodic points and periodic identifiers for the given coordinates
        and list of points.
        Parameters :
        coordinate : List, list of grain coordinates
        points : Array, list of points in interval [step_half-box_size, box_size*2) with step_size spacing
        """
        points = list(points)
        periodic_pt, periodic_identifier_list = ([] for i in range(2))
        for j, point in enumerate(coordinate):
            if point > self.box_size:
                idx = points.index(point)
                periodic_point = points[idx - self.n_pts]
                periodic_pt.append(periodic_point)
                periodic_identifier_list.append(1)

            elif point < self.step_half:
                idx = points.index(point)
                periodic_point = points[idx + self.n_pts]
                periodic_pt.append(periodic_point)
                periodic_identifier_list.append(-1)

            else:
                periodic_pt.append(point)
                periodic_identifier_list.append(0)

        return periodic_pt, periodic_identifier_list

    def band_generator(self, xyz_grid, plane='xy'):
        """Creates a band of given bandwidth for given points in interval [step_half, box_size)
        with step_size spacing along the axis.
        Parameters :
        xyz_grid : Array, list of points in interval [step_half, box_size) with step_size spacing
        plane : String, default is 'xy'
        """
        band_half = self.bandwidth / 2
        rand_idx = int(np.random.rand() * len(xyz_grid))
        band_center = xyz_grid[rand_idx]
        x, y, z = np.meshgrid(xyz_grid, xyz_grid, xyz_grid)
        if plane == 'xy':
            r = z
        elif plane == 'yz':
            r = x
        elif plane == 'xz':
            r = y
        else:
            self.logger.error("Error: plane must be defined as xy, yz or xz! Default: xy")
            sys.exit(1)
        left_bound = r >= band_center - band_half
        right_bound = r <= band_center + band_half
        self.logger.info("Band generator - Bandwidth: {}, Left bound: {} and Right bound: {}"
                         .format(self.bandwidth, left_bound, right_bound))
        left = set([a for a in zip(x[right_bound], y[right_bound], z[right_bound])])
        right = set([a for a in zip(x[left_bound], y[left_bound], z[left_bound])])

        return left.intersection(right)

    def make_periodic_2D(self, points_array, ellipse_points, iterator):
        points_array_mod = np.zeros(points_array.shape)
        points_array_mod[points_array == iterator] = iterator

        for i in range(1, 9):
            points_array_copy = np.zeros(points_array.shape)
            points_array_copy[(ellipse_points <= 1) & (points_array == -1*i)] = -100-i
            if i % 2 != 0:
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_mod[np.where(points_array_copy == -100-i)] = iterator
            elif (i == 2) | (i == 6):
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
            else:
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
        return points_array_mod

    def make_periodic_3D(self, points_array, ellipse_points, iterator):
        points_array_mod = np.zeros(points_array.shape)
        points_array_mod[points_array == iterator] = iterator
        t_0 = datetime.datetime.now()
        for i in range(1, 27): #move points in x,y and z dir
            points_array_copy = np.zeros(points_array.shape)
            points_array_copy[(ellipse_points <= 1) & (points_array == -1*i)] = -100-i
            if (i == 1) | (i == 3) | (i == 7) | (i == 9) | \
                    (i == 18) | (i == 20) | (i == 24) | (i == 26):  # move points in x,y and z direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=2)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 10) | (i == 12) | (i == 15) | (i == 17):  # move points in x and y direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 4) | (i == 6) | (i == 21) | (i == 23):  # move points in x and z direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=2)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 2) | (i == 8) | (i == 19) | (i == 25):  # move points in y and z direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=2)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 13) | (i == 14) :  # move points in x direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=1)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 11) | (i == 16):  # move points in y direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=0)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 5) | (i == 22):  # move points in z direction
                points_array_copy = np.roll(points_array_copy, self.n_pts, axis=2)
                points_array_mod[points_array_copy == -100-i] = iterator
        time_elapse = datetime.datetime.now()-t_0
        if self.debug:
            self.logger.info('time spent on periodicity for grain {}: {}'.format(iterator, time_elapse.total_seconds()) )
        return points_array_mod

    def gen_boundaries_2D(self, points_array):
        box_size = self.box_size
        x_grid = self.x_grid
        y_grid = self.y_grid
        points_array[np.where((x_grid > box_size) & (y_grid > box_size))] = -1
        points_array[(x_grid < box_size) & (y_grid > box_size)] = -2
        points_array[(x_grid < 0) & (y_grid > box_size)] = -3
        points_array[(x_grid < 0) & (y_grid < box_size)] = -4
        points_array[(x_grid > box_size) & (y_grid < box_size)] = -8
        points_array[(x_grid > box_size) & (y_grid < 0)] = -7
        points_array[(x_grid < box_size) & (y_grid < 0)] = -6
        points_array[(x_grid < 0) & (y_grid < 0)] = -5
        return points_array

    def gen_boundaries_3D(self, points_array):
        t_0 = datetime.datetime.now()
        box_size = self.box_size
        x_grid = self.x_grid
        y_grid = self.y_grid
        z_grid = self.z_grid

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