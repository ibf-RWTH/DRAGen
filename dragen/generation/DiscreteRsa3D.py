import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import sys
import datetime
import logging

from dragen.utilities.RVE_Utils import RVEUtils


class DiscreteRsa3D:

    def __init__(self, box_size, n_pts, a, b, c, slope, store_path, debug=False):

        self.box_size = box_size
        self.n_pts = n_pts
        self.a = a
        self.b = b
        self.c = c
        self.slope = slope
        self.store_path = store_path
        self.debug = debug

        self.logger = logging.getLogger("RVE-Gen")
        xyz = np.linspace(-box_size / 2, box_size + box_size / 2, 2 * self.n_pts, endpoint=True)
        self.x_grid, self.y_grid, self.z_grid = np.meshgrid(xyz, xyz, xyz)
        self.n_grains = len(a)
        self.rve_utils_object = RVEUtils(box_size, n_pts, self.x_grid, self.y_grid, self.z_grid, debug=debug)

    def gen_ellipsoid(self, array, iterator):
        t_0 = datetime.datetime.now()
        x_grid = self.x_grid
        y_grid = self.y_grid
        z_grid = self.z_grid
        a = self.a
        b = self.b
        c = self.c
        slope = self.slope
        unoccupied_pts_x, unoccupied_pts_y, unoccupied_pts_z = np.where(array == 0)
        unoccupied_tuples = [*zip(unoccupied_pts_x, unoccupied_pts_y, unoccupied_pts_z)]
        unoccupied_area_x = [self.x_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]][unoccupied_tuples_i[2]]
                             for unoccupied_tuples_i in unoccupied_tuples]
        unoccupied_area_y = [self.y_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]][unoccupied_tuples_i[2]]
                             for unoccupied_tuples_i in unoccupied_tuples]
        unoccupied_area_z = [self.z_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]][unoccupied_tuples_i[2]]
                             for unoccupied_tuples_i in unoccupied_tuples]

        idx = random.choice(range(len(unoccupied_area_x)))
        x_0 = unoccupied_area_x[idx]
        y_0 = unoccupied_area_y[idx]
        z_0 = unoccupied_area_z[idx]
        print('x_0_{}: {}, y_0_{}: {}, z_0_{}: {}'.format(iterator, x_0, iterator, y_0, iterator, z_0))
        a = a[iterator]
        b = b[iterator]
        c = c[iterator]
        slope = slope[iterator]
        """ellipse = (x_grid - x_0) ** 2 / (a ** 2) + \
                  (y_grid - y_0) ** 2 / (b ** 2) + \
                  (z_grid - z_0) ** 2 / (c ** 2)"""
        ellipsoid = self.rve_utils_object.ellipsoid(a, b, c, x_0, y_0, z_0, slope=slope)

        time_elapse = datetime.datetime.now() - t_0
        if self.debug:
            self.logger.info('time spent on ellipsoid{}: {}'.format(iterator, time_elapse.total_seconds()))
        return ellipsoid, x_0, y_0, z_0

    def rsa_plotter(self, array, iterator, attempt):
        plt.ioff()
        t_0 = datetime.datetime.now()
        n_grains = self.n_grains
        rve_x, rve_y, rve_z = np.where((array >= 1) | (array == -200))
        grain_tuples = [*zip(rve_x, rve_y, rve_z)]
        grains_x = [self.x_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]
        grains_y = [self.y_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]
        grains_z = [self.z_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(grains_x, grains_y, grains_z, c=array[np.where((array > 0) | (array == -200))], s=1, vmin=-20,  vmax=n_grains, cmap='seismic')
        #ax.scatter(grains_x, grains_y, grains_z, c='r', s=1)

        ax.set_xlim(-5, self.box_size + 5)
        ax.set_ylim(-5, self.box_size + 5)
        ax.set_zlim(-5, self.box_size + 5)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_zlabel('z (µm)')
        #ax.view_init(90, 270)
        #plt.show()
        plt.savefig(self.store_path + '/Figs/3D_Epoch_{}_{}.png'.format(iterator, attempt))
        plt.close(fig)
        time_elapse = datetime.datetime.now() - t_0
        if self.debug:
            self.logger.info('time spent on plotter for grain {}: {}'.format(iterator, time_elapse.total_seconds()))

    def run_rsa(self, banded_rsa_array=None, animation=False):
        status = False
        bandratio = 0.3

        if banded_rsa_array is None:
            rsa = np.zeros((2 * self.n_pts, 2 * self.n_pts, 2 * self.n_pts), dtype=np.int32)
            rsa = self.rve_utils_object.gen_boundaries_3D(rsa)

        else:
            rsa = banded_rsa_array

        band_vol_0 = np.count_nonzero(rsa == -200)
        rsa_boundaries = rsa.copy()
        x_0_list = list()
        y_0_list = list()
        z_0_list = list()

        i = 1
        attempt = 0
        while i < self.n_grains + 1 | attempt < 1000:
            t_0 = datetime.datetime.now()
            free_points_old = np.count_nonzero(rsa == 0)
            band_points_old = np.count_nonzero(rsa == -200)
            grain = rsa_boundaries.copy()
            backup_rsa = rsa.copy()
            ellipsoid, x0, y0, z0 = self.gen_ellipsoid(rsa, iterator=i-1)
            grain[(ellipsoid <= 1) & ((grain == 0) | (grain == -200))] = i
            periodic_grain = self.rve_utils_object.make_periodic_3D(grain, ellipsoid, iterator=i)
            rsa[(periodic_grain == i) & ((rsa == 0) | (rsa == -200))] = i

            if animation:
                self.rsa_plotter(rsa, iterator=i, attempt=attempt)

            free_points = np.count_nonzero(rsa == 0)
            band_points = np.count_nonzero(rsa == -200)
            if band_points_old>0:
                if (free_points_old + band_points_old - free_points -band_points != np.count_nonzero(periodic_grain)) | \
                        (band_points/band_vol_0 < (1-bandratio)):
                    print('difference: ', free_points_old - free_points != np.count_nonzero(periodic_grain))
                    print('ratio:', (band_points/band_vol_0 < bandratio))
                    rsa = backup_rsa.copy()
                    attempt = attempt + 1

                else:
                    x_0_list.append(x0)
                    y_0_list.append(y0)
                    z_0_list.append(z0)
                    i = i + 1
                    attempt = 0
                    if self.debug:
                        time_elapse = datetime.datetime.now() - t_0
                        self.logger.info('total time needed for placement of grain {}: {}'.format(i, time_elapse.total_seconds()) )
            else:
                if (free_points_old + band_points_old - free_points - band_points != np.count_nonzero(periodic_grain)):
                    print('difference: ', free_points_old - free_points != np.count_nonzero(periodic_grain))
                    rsa = backup_rsa.copy()
                    attempt = attempt + 1

                else:
                    x_0_list.append(x0)
                    y_0_list.append(y0)
                    z_0_list.append(z0)
                    i = i + 1
                    attempt = 0
                    if self.debug:
                        time_elapse = datetime.datetime.now() - t_0
                        self.logger.info(
                            'total time needed for placement of grain {}: {}'.format(i, time_elapse.total_seconds()))
        if len(x_0_list) == self.n_grains:
            status = True
        else:
            self.logger.info("Not all grains could be placed please decrease shrinkfactor!")
        return rsa, x_0_list, y_0_list, z_0_list, status


if __name__ == '__main__':
    a = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    b = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    c = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    animation_test = True
    debug_test = True
    # Define Box-dimension
    box_size = 100
    # Define resolution of Grid
    n_pts_test = 50
    rsa_obj = DiscreteRsa3D(box_size, n_pts_test, a, b, c, debug_test)
    rsa, x_0_list, y_0_list, z_0_list, status = rsa_obj.run_rsa(np.array, animation_test)
    np.save('./3D_x0_list', x_0_list, allow_pickle=True, fix_imports=True)
    np.save('./3D_y0_list', y_0_list, allow_pickle=True, fix_imports=True)
    np.save('./3D_z0_list', z_0_list, allow_pickle=True, fix_imports=True)









