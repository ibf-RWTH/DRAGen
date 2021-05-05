import numpy as np
import matplotlib.pyplot as plt
import random
import logging

from dragen.utilities.RVE_Utils import *

class DiscreteRsa2D:
    def __init__(self, box_size, n_pts, a, b, alpha, store_path):

        self.box_size = box_size
        self.n_pts = n_pts
        self.a = a
        self.b = b
        self.alpha = alpha
        self.store_path = store_path

        self.n_grains = len(a)
        self.logger = logging.getLogger("RVE-Gen")
        xy = np.linspace(-self.box_size / 2, self.box_size + self.box_size / 2, 2 * self.n_pts, endpoint=True)
        self.x_grid, self.y_grid = np.meshgrid(xy, xy)
        self.rve_utils_object = RVEUtils()

    def gen_ellipsoid(self, array, iterator):
        x_grid = self.x_grid
        y_grid = self.y_grid
        a = self.a
        b = self.b
        alpha = self.alpha
        unoccupied_pts_x, unoccupied_pts_y = np.where(array == 0)
        unoccupied_tuples = [*zip(unoccupied_pts_x, unoccupied_pts_y)]
        unoccupied_area_x = [self.x_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]] for unoccupied_tuples_i in
                             unoccupied_tuples]
        unoccupied_area_y = [self.y_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]] for unoccupied_tuples_i in
                             unoccupied_tuples]

        idx = random.choice(range(len(unoccupied_area_x)))
        x_0 = unoccupied_area_x[idx]
        y_0 = unoccupied_area_y[idx]
        print('x_0_{}: {}, y_0_{}: {}'.format(iterator, x_0, iterator, y_0))
        a = a[iterator]
        b = b[iterator]
        alpha = alpha[iterator]

        ellipse = self.rve_utils_object.ellipse(a, b, x_0, y_0, alpha=alpha)

        return ellipse, x_0, y_0

    def rsa_plotter(self, array, n_grains, iterator, attempt):
        rsa_x, rsa_y = np.where(array >= 1)
        outside_x, outside_y = np.where(array < 0)
        unoccupied_pts_x, unoccupied_pts_y = np.where(array == 0)

        grain_tuples = [*zip(rsa_x, rsa_y)]
        grains_x = [self.x_grid[grain_tuples_i[0]][grain_tuples_i[1]] for grain_tuples_i in grain_tuples]
        grains_y = [self.y_grid[grain_tuples_i[0]][grain_tuples_i[1]] for grain_tuples_i in grain_tuples]

        outside_tuples = [*zip(outside_x, outside_y)]
        boundary_x = [self.x_grid[outside_tuples_i[0]][outside_tuples_i[1]] for outside_tuples_i in outside_tuples]
        boundary_y = [self.y_grid[outside_tuples_i[0]][outside_tuples_i[1]] for outside_tuples_i in outside_tuples]

        unoccupied_tuples = [*zip(unoccupied_pts_x, unoccupied_pts_y)]
        unoccupied_area_x = [self.x_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]] for unoccupied_tuples_i in
                             unoccupied_tuples]
        unoccupied_area_y = [self.y_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]] for unoccupied_tuples_i in
                             unoccupied_tuples]

        fig = plt.figure()
        plt.scatter(unoccupied_area_x, unoccupied_area_y, c='gray', s=1)
        plt.scatter(grains_x, grains_y, c=array[np.where(array > 0)], s=1, vmin=0, vmax=n_grains, cmap='seismic')
        # plt.scatter(boundary_x, boundary_y, c='r')
        # plt.scatter(ellipse_outside_x, ellipse_outside_y, c='k')
        plt.xlim(-5, self.box_size + 5)
        plt.ylim(-5, self.box_size + 5)
        plt.savefig(self.store_path+'/Figs/2D_RSA_Epoch_{}_{}.png'.format(iterator, attempt))
        plt.close(fig)

    def run_rsa(self, animation=False):

        #define some variables
        status = False
        x_0_list = list()
        y_0_list = list()
        i = 1
        attempt = 0
        rsa = np.zeros((2 * self.n_pts, 2 * self.n_pts), dtype=np.int32)
        rsa = self.rve_utils_object.gen_boundaries_2D(rsa)
        rsa_boundaries = rsa.copy()

        free_points = np.count_nonzero(rsa == 0)
        while i < self.n_grains + 1 | attempt < free_points:
            free_points_old = np.count_nonzero(rsa == 0)
            grain = rsa_boundaries.copy()
            ellipse, x0, y0 = self.gen_ellipsoid(rsa, iterator=i - 1)
            grain[(ellipse <= 1) & (grain == 0)] = i
            periodic_grain = self.rve_utils_object.make_periodic_2D(grain, ellipse, iterator=i)
            rsa[(periodic_grain == i) & (rsa == 0)] = i

            if animation:
                self.rsa_plotter(rsa, self.n_grains, iterator=i, attempt=attempt)

            free_points = np.count_nonzero(rsa == 0)
            if free_points_old - free_points != np.count_nonzero(periodic_grain):
                rsa[np.where(rsa == i)] = 0
                attempt = attempt + 1

            else:
                i = i + 1
                x_0_list.append(x0)
                y_0_list.append(y0)
                attempt = 0

        if len(x_0_list) == self.n_grains:
            status = True
        else:
            self.logger.info("Not all grains could be placed please decrease shrinkfactor!")

        return rsa, x_0_list, y_0_list, status


if __name__ == '__main__':
    a = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    b = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    # Define Box-dimension
    box_size = 100
    # Define resolution of Grid
    n_pts = 100
    rsa_obj = DiscreteRsa2D(box_size, n_pts, a, b)
    rsa, x_0_list, y_0_list, status = rsa_obj.run_rsa(animation=True)
    print(x_0_list)
    print(y_0_list)
    np.save('./2D_x_0', x_0_list, allow_pickle=True, fix_imports=True)
    np.save('./2D_y_0', y_0_list, allow_pickle=True, fix_imports=True)
    np.save('./2D_rve', rsa, allow_pickle=True, fix_imports=True)






