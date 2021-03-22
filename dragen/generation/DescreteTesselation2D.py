import matplotlib.pyplot as plt
import numpy as np
import logging

import sys

from dragen.utilities.RVE_Utils import RVEUtils


class Tesselation2D:
    def __init__(self, box_size, n_pts, a, b, x_0, y_0, shrinkfactor, storepath):

        self.box_size = box_size
        self.n_pts = n_pts
        self.a = a
        self.b = b
        self.x_0 = x_0
        self.y_0 = y_0
        self.shrinkfactor = shrinkfactor
        self.storepath = storepath

        self.logger = logging.getLogger("RVE-Gen")
        self.bin_size = box_size / n_pts
        self.a_max = max(a)
        self.b_max = max(b)
        self.final_volume = [np.pi * a[i] * b[i] / shrinkfactor for i in range(len(a))]
        xy = np.linspace(-self.box_size / 2, self.box_size + self.box_size / 2, 2 * self.n_pts, endpoint=True)
        self.x_grid, self.y_grid = np.meshgrid(xy, xy)
        self.rve_utils_object = RVEUtils(box_size, n_pts, self.x_grid, self.y_grid)

    def grow(self, iterator, a, b):
        x_grid = self.x_grid
        y_grid = self.y_grid
        x_0 = self.x_0[iterator-1]
        y_0 = self.y_0[iterator-1]
        a_i = a[iterator - 1]
        b_i = b[iterator - 1]
        a_i = a_i + b_i/self.a_max*self.bin_size
        b_i = b_i + b_i/self.b_max*self.bin_size
        if iterator == 1:
            print(a_i)
            print(b_i)
        a[iterator - 1] = a_i
        b[iterator - 1] = b_i
        ellipse = (x_grid - x_0) ** 2 / (a_i ** 2) + (y_grid - y_0) ** 2 / (b_i ** 2)

        return ellipse, a, b

    def tesselation_plotter(self, array, storepath, epoch):
        n_grains = len(self.a)
        rve_x, rve_y = np.where(array >= 1)
        unoccupied_pts_x, unoccupied_pts_y = np.where(array == 0)

        grain_tuples = [*zip(rve_x, rve_y)]
        grains_x = [self.x_grid[grain_tuples_i[0]][grain_tuples_i[1]] for grain_tuples_i in grain_tuples]
        grains_y = [self.y_grid[grain_tuples_i[0]][grain_tuples_i[1]] for grain_tuples_i in grain_tuples]

        unoccupied_tuples = [*zip(unoccupied_pts_x, unoccupied_pts_y)]
        unoccupied_area_x = [self.x_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]] for unoccupied_tuples_i in
                             unoccupied_tuples]
        unoccupied_area_y = [self.y_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]] for unoccupied_tuples_i in
                             unoccupied_tuples]

        fig = plt.figure()
        plt.scatter(unoccupied_area_x, unoccupied_area_y, c='gray', s=1)
        plt.scatter(grains_x, grains_y, c=array[np.where(array > 0)], s=1, vmin=0, vmax=n_grains, cmap='seismic')
        plt.xlim(-5, self.box_size + 5)
        plt.ylim(-5, self.box_size + 5)
        plt.savefig(storepath+'/Figs/2D_Tesselation_Epoch_{}.png'.format(epoch))
        plt.close(fig)

    def run_tesselation(self, rsa):

        # define some variables
        status = False
        repeat = False
        packingratio = 0
        a = self.a
        b = self.b
        epoch = 0
        n_grains = len(self.a)

        # load some variables
        rve = rsa
        empty_rve = np.zeros((2 * self.n_pts, 2 * self.n_pts), dtype=np.int32)
        empty_rve = self.rve_utils_object.gen_boundaries_2D(empty_rve)
        rve_boundaries = empty_rve.copy()  # empty rve grid with defined boundaries

        vol_0 = np.count_nonzero(empty_rve == 0)
        freepoints = np.count_nonzero(rve == 0)

        grain_idx = [i for i in range(1, n_grains+1)]
        grain_idx_backup = grain_idx.copy()
        while freepoints > 0:
            i = 0
            np.random.shuffle(grain_idx)
            while i < len(grain_idx):
                idx = grain_idx[i]
                grain = rve_boundaries.copy()
                ellipse, a, b = self.grow(idx, a, b)
                grain[(ellipse <= 1) & (grain == 0)] = idx
                periodic_grain = self.rve_utils_object.make_periodic_2D(grain, ellipse, iterator=idx)
                rve[(periodic_grain == idx) & (rve == 0)] = idx
                freepoints = np.count_nonzero(rve == 0)
                grain_vol = np.count_nonzero(rve == idx)*self.bin_size**2
                if freepoints == 0:
                    break
                if grain_vol > self.final_volume[idx-1] and not repeat:
                    self.logger.info('grain_{} reached final volume and was deleted '
                                     'from growth list'.format(grain_idx[i]))
                    del grain_idx[i]
                i += 1
            if len(grain_idx) == 0:
                print('restore')
                grain_idx = grain_idx_backup.copy()
                repeat = True
                self.logger.info('grain_idx was restored since all grains reached final volume'
                                 'Boxsize should probably be decreased or shrinkfactor increased'
                                 'use animation to check if how much space is used by the grains')

            self.tesselation_plotter(rve, self.storepath, epoch)
            epoch += 1
            packingratio = (1-freepoints/vol_0)*100
            print('packingratio:', packingratio, '%')
        if packingratio == 1:
            status = True
        return rve, status


if __name__ == '__main__':
    a_test = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    b_test = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
    # Define Box-dimension
    box_size = 100
    # Define resolution of Grid
    n_pts = 256
    shrinkfactor = 0.42
    x0_path = './2D_x_0.npy'
    y0_path = './2D_y_0.npy'
    x_0 = np.load(x0_path)
    y_0 = np.load(y0_path)
    tesselation_obj = Tesselation2D(box_size, n_pts, a_test, b_test, x_0, y_0, shrinkfactor)
    tesselation_obj.run_tesselation()



