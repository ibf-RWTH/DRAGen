import sys
import pandas as pd
import numpy as np
import logging
import datetime

import matplotlib.pyplot as plt

class RVEUtils:
    """Common Representative Volume Element (RVE) operations."""

    def __init__(self, box_size, n_pts, x_grid=None, y_grid=None, z_grid=None, bandwidth=0, debug=False) -> None:
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

    @staticmethod
    def read_input(file_name, dimension) -> tuple:
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

    def make_periodic_2D(self, points_array, ellipse_points, iterator) -> np.ndarray:
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

    def make_periodic_3D(self, points_array, ellipse_points, iterator) -> np.ndarray:
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

    def gen_boundaries_2D(self, points_array) -> np.ndarray:
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

    def gen_boundaries_3D(self, points_array) -> np.ndarray:
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

    def repair_periodicity_2D(self, rve_array: np.ndarray) -> pd.DataFrame:

        # load some variables
        box_size = self.box_size
        n_pts = self.n_pts

        xy = np.linspace(-box_size / 2, box_size + box_size / 2, 2 * self.n_pts, endpoint=True)
        x_grid, y_grid = np.meshgrid(xy, xy)
        rve_xx, rve_yy = np.where(rve_array >= 1)
        boundary_xx, boundary_yy = np.where(rve_array < 0)

        rve_tuples = [*zip(rve_xx, rve_yy)]
        boundary_tuples = [*zip(boundary_xx, boundary_yy)]

        rve_x = [x_grid[rve_tuples_i[0]][rve_tuples_i[1]] for rve_tuples_i in rve_tuples]
        rve_y = [y_grid[rve_tuples_i[0]][rve_tuples_i[1]] for rve_tuples_i in rve_tuples]

        boundary_x = [x_grid[boundary_tuples_i[0]][boundary_tuples_i[1]] for boundary_tuples_i in boundary_tuples]
        boundary_y = [y_grid[boundary_tuples_i[0]][boundary_tuples_i[1]] for boundary_tuples_i in boundary_tuples]

        rve_dict = {'x': rve_x, 'y': rve_y, 'GrainID': rve_array[rve_array > 0]}
        rve = pd.DataFrame(rve_dict)
        rve['box_size'] = box_size
        rve['n_pts'] = n_pts

        boundary_dict = {'x': boundary_x, 'y': boundary_y, 'GrainID': rve_array[rve_array < 0]}
        boundary = pd.DataFrame(boundary_dict)

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

        # fixing corners
        rve_corners = rve.loc[((rve['x'] == min_x) & (rve['y'] == min_y)) |
                              ((rve['x'] == max_x) & (rve['y'] == max_y)) |
                              ((rve['x'] == min_x) & (rve['y'] == max_y)) |
                              ((rve['x'] == max_x) & (rve['y'] == min_y))].copy()

        cornersGrainID = rve_corners[(rve_corners['x'] == min_x) & (rve_corners['y'] == min_y)].GrainID.values

        rve.loc[rve_corners.index, 'GrainID'] = cornersGrainID

        # fixing Edges
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

        # load some variables
        box_size = self.box_size
        n_pts = self.n_pts
        dim = rve_array.shape[1]

        xyz = np.linspace(-box_size / 2, box_size + box_size / 2, 2 * self.n_pts, endpoint=True)
        x_grid, y_grid, z_grid = np.meshgrid(xyz, xyz, xyz)
        rve_xx, rve_yy, rve_zz = np.where(rve_array >= 1)
        grain_tuples = [*zip(rve_xx, rve_yy, rve_zz)]

        rve_x = [x_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]
        rve_y = [y_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]
        rve_z = [z_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]

        rve_dict = {'x': rve_x, 'y': rve_y, 'z': rve_z, 'grainID': rve_array[rve_array > 0]}
        rve = pd.DataFrame(rve_dict)
        rve['box_size'] = box_size
        rve['n_pts'] = n_pts

        max_x = max(rve.x)
        min_x = min(rve.x)
        max_y = max(rve.y)
        min_y = min(rve.y)
        max_z = max(rve.z)
        min_z = min(rve.z)

        # fixing corners
        rve_corners = rve.loc[((rve['x'] == min_x) &
                               (rve['y'] == min_y) & (rve['z'] == max_z)) |
                              ((rve['x'] == max_x) &
                               (rve['y'] == min_y) & (rve['z'] == max_z)) |
                              ((rve['x'] == max_x) &
                               (rve['y'] == max_y) & (rve['z'] == max_z)) |
                              ((rve['x'] == min_x) &
                               (rve['y'] == max_y) & (rve['z'] == max_z)) |
                              ((rve['x'] == min_x) &
                               (rve['y'] == min_y) & (rve['z'] == min_z)) |
                              ((rve['x'] == max_x) &
                               (rve['y'] == min_y) & (rve['z'] == min_z)) |
                              ((rve['x'] == max_x) &
                               (rve['y'] == max_y) & (rve['z'] == min_z)) |
                              ((rve['x'] == min_x) &
                               (rve['y'] == max_y) & (rve['z'] == min_z))]

        # cornersGrainID = rve_corners.GrainID.mode()[0] #find most common grainID on all corners and advise it to each corner
        # rve.loc[rve_corners.index,'GrainID'] = cornersGrainID

        # fixing Edges
        rve_edges = rve.loc[(((rve['x'] == max_x) | (rve['x'] == min_x)) &
                             ((rve['y'] == max_y) | (rve['y'] == min_y)) &
                             ((rve['z'] != max_z) & (rve['z'] != min_z))) |

                            (((rve['x'] == max_x) | (rve['x'] == min_x)) &
                             ((rve['y'] != max_y) & (rve['y'] != min_y)) &
                             ((rve['z'] == max_z) | (rve['z'] == min_z))) |

                            (((rve['x'] != max_x) & (rve['x'] != min_x)) &
                             ((rve['y'] == max_y) | (rve['y'] == min_y)) &
                             ((rve['z'] == max_z) | (rve['z'] == min_z)))]
        # Top front Edge
        E_T1 = rve_edges.loc[(rve_edges['y'] == max_y) & (rve_edges['z'] == max_z)].copy()
        # Top right Edge
        E_T2 = rve_edges.loc[(rve_edges['x'] == max_x) & (rve_edges['y'] == max_y)].copy()
        # Top back Edge
        E_T3 = rve_edges.loc[(rve_edges['y'] == max_y) & (rve_edges['z'] == min_z)].copy()
        # Top left Edge
        E_T4 = rve_edges.loc[(rve_edges['x'] == min_x) & (rve_edges['y'] == max_y)].copy()
        # bottm front edge
        E_B1 = rve_edges.loc[(rve_edges['y'] == min_y) & (rve_edges['z'] == max_z)].copy()
        # bottm right edge
        E_B2 = rve_edges.loc[(rve_edges['x'] == max_x) & (rve_edges['y'] == min_y)].copy()
        # bottm back edge
        E_B3 = rve_edges.loc[(rve_edges['y'] == min_y) & (rve_edges['z'] == min_z)].copy()
        # bottm left edge
        E_B4 = rve_edges.loc[(rve_edges['x'] == min_x) & (rve_edges['y'] == min_y)].copy()
        # left front edge
        E_M1 = rve_edges.loc[(rve_edges['x'] == min_x) & (rve_edges['z'] == max_z)].copy()
        # right front edge
        E_M2 = rve_edges.loc[(rve_edges['x'] == max_x) & (rve_edges['z'] == max_z)].copy()
        # left rear edge
        E_M4 = rve_edges.loc[(rve_edges['x'] == min_x) & (rve_edges['z'] == min_z)].copy()
        # right rear edge
        E_M3 = rve_edges.loc[(rve_edges['x'] == max_x) & (rve_edges['z'] == min_z)].copy()

        E_T1.reset_index(inplace=True)
        E_T3.reset_index(inplace=True)
        E_B1.reset_index(inplace=True)
        E_B3.reset_index(inplace=True)
        E_T1['GrainID'] = np.where(E_B1['GrainID'] != E_T1['GrainID'], E_B1['GrainID'], E_T1['GrainID'])
        E_T3['GrainID'] = np.where(E_B1['GrainID'] != E_T3['GrainID'], E_B1['GrainID'], E_T3['GrainID'])
        E_B3['GrainID'] = np.where(E_B1['GrainID'] != E_B3['GrainID'], E_B1['GrainID'], E_B3['GrainID'])
        E_T1.set_index('index', inplace=True)
        E_T3.set_index('index', inplace=True)
        E_B3.set_index('index', inplace=True)

        E_B4.reset_index(inplace=True)
        E_B2.reset_index(inplace=True)
        E_T4.reset_index(inplace=True)
        E_T2.reset_index(inplace=True)
        E_B2['GrainID'] = np.where(E_B4['GrainID'] != E_B2['GrainID'], E_B4['GrainID'], E_B2['GrainID'])
        E_T4['GrainID'] = np.where(E_B4['GrainID'] != E_T4['GrainID'], E_B4['GrainID'], E_T4['GrainID'])
        E_T2['GrainID'] = np.where(E_B4['GrainID'] != E_T2['GrainID'], E_B4['GrainID'], E_T2['GrainID'])
        E_B2.set_index('index', inplace=True)
        E_T4.set_index('index', inplace=True)
        E_T2.set_index('index', inplace=True)

        E_M1.reset_index(inplace=True)
        E_M2.reset_index(inplace=True)
        E_M3.reset_index(inplace=True)
        E_M4.reset_index(inplace=True)
        E_M2['GrainID'] = np.where(E_M1['GrainID'] != E_M2['GrainID'], E_M1['GrainID'], E_M2['GrainID'])
        E_M3['GrainID'] = np.where(E_M1['GrainID'] != E_M3['GrainID'], E_M1['GrainID'], E_M3['GrainID'])
        E_M4['GrainID'] = np.where(E_M1['GrainID'] != E_M4['GrainID'], E_M1['GrainID'], E_M4['GrainID'])
        E_M2.set_index('index', inplace=True)
        E_M3.set_index('index', inplace=True)
        E_M4.set_index('index', inplace=True)

        rve.loc[E_B2.index, 'GrainID'] = E_B2
        rve.loc[E_B3.index, 'GrainID'] = E_B3
        rve.loc[E_T1.index, 'GrainID'] = E_T1
        rve.loc[E_T2.index, 'GrainID'] = E_T2
        rve.loc[E_T3.index, 'GrainID'] = E_T3
        rve.loc[E_T4.index, 'GrainID'] = E_T4
        rve.loc[E_M2.index, 'GrainID'] = E_M2
        rve.loc[E_M3.index, 'GrainID'] = E_M3
        rve.loc[E_M4.index, 'GrainID'] = E_M4

        rve_faces = rve.loc[(((rve['x'] == max_x) | (rve['x'] == min_x)) &
                             ((rve['y'] != max_y) & (rve['y'] != min_y)) &
                             ((rve['z'] != max_z) & (rve['z'] != min_z))) |

                            (((rve['x'] != max_x) & (rve['x'] != min_x)) &
                             ((rve['y'] != max_y) & (rve['y'] != min_y)) &
                             ((rve['z'] == max_z) | (rve['z'] == min_z))) |

                            (((rve['x'] != max_x) & (rve['x'] != min_x)) &
                             ((rve['y'] == max_y) | (rve['y'] == min_y)) &
                             ((rve['z'] != max_z) & (rve['z'] != min_z)))]

        # left set
        LeftSet = rve_faces.loc[rve_faces['x'] == min_x].copy()
        # right set
        RightSet = rve_faces.loc[rve_faces['x'] == max_x].copy()
        # bottom set
        BottomSet = rve_faces.loc[rve_faces['y'] == min_y].copy()
        # top set
        TopSet = rve_faces.loc[rve_faces['y'] == max_y].copy()
        # front set
        RearSet = rve_faces.loc[rve_faces['z'] == min_z].copy()
        # rear set
        FrontSet = rve_faces.loc[rve_faces['z'] == max_z].copy()

        LeftSet.reset_index(inplace=True)
        BottomSet.reset_index(inplace=True)
        FrontSet.reset_index(inplace=True)
        RightSet.reset_index(inplace=True)
        TopSet.reset_index(inplace=True)
        RearSet.reset_index(inplace=True)
        RightSet['GrainID'] = np.where(LeftSet['GrainID'] != RightSet['GrainID'], LeftSet['GrainID'],
                                       RightSet['GrainID'])
        TopSet['GrainID'] = np.where(BottomSet['GrainID'] != TopSet['GrainID'], BottomSet['GrainID'], TopSet['GrainID'])
        RearSet['GrainID'] = np.where(FrontSet['GrainID'] != RearSet['GrainID'], FrontSet['GrainID'],
                                      RearSet['GrainID'])
        RightSet.set_index('index', inplace=True)
        TopSet.set_index('index', inplace=True)
        RearSet.set_index('index', inplace=True)
        rve.loc[RightSet.index, 'GrainID'] = RightSet
        rve.loc[TopSet.index, 'GrainID'] = TopSet
        rve.loc[RearSet.index, 'GrainID'] = RearSet

        return rve