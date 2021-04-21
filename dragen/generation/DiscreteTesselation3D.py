import matplotlib.pyplot as plt
import numpy as np
import datetime
import logging
import sys

from dragen.utilities.RVE_Utils import RVEUtils


class Tesselation3D:

    def __init__(self, box_size, n_pts, a, b, c, alpha, x_0, y_0, z_0, final_volume, shrinkfactor, band_ratio, store_path, debug=False):

        self.box_size = box_size
        self.n_pts = n_pts
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0
        self.shrinkfactor = shrinkfactor
        self.band_ratio = band_ratio
        self.store_path = store_path
        self.debug = debug

        self.logger = logging.getLogger("RVE-Gen")
        self.n_grains = len(a)
        self.bin_size = box_size / n_pts
        self.a_max = max(a)
        self.b_max = max(b)
        self.c_max = max(c)
        self.final_volume = final_volume
        xyz = np.linspace(-self.box_size / 2, self.box_size + self.box_size / 2, 2 * self.n_pts, endpoint=True)
        self.x_grid, self.y_grid, self.z_grid = np.meshgrid(xyz, xyz, xyz)
        self.rve_utils_object = RVEUtils(box_size, n_pts, self.x_grid, self.y_grid, self.z_grid, debug=debug)

    def grow(self, iterator, a, b, c):
        x_grid = self.x_grid
        y_grid = self.y_grid
        z_grid = self.z_grid
        alpha = self.alpha[iterator-1]
        x_0 = self.x_0[iterator-1]
        y_0 = self.y_0[iterator-1]
        z_0 = self.z_0[iterator-1]
        a_i = a[iterator - 1]
        b_i = b[iterator - 1]
        c_i = c[iterator - 1]
        a_i = a_i + a_i / self.a_max * self.bin_size
        b_i = b_i + b_i / self.b_max * self.bin_size
        c_i = c_i + c_i / self.c_max * self.bin_size
        a[iterator - 1] = a_i
        b[iterator - 1] = b_i
        c[iterator - 1] = c_i

        """ellipsoid = (x_grid - x_0) ** 2 / (a_i ** 2) + \
                  (y_grid - y_0) ** 2 / (b_i ** 2) + \
                  (z_grid - z_0) ** 2 / (c_i ** 2)"""

        ellipsoid = self.rve_utils_object.ellipsoid(a_i, b_i, c_i, x_0, y_0, z_0, alpha=alpha)

        return ellipsoid, a, b, c

    def tesselation_plotter(self, array, epoch):
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
        ax.scatter(grains_x, grains_y, grains_z, c=array[np.where((array >= 1) | (array == -200))], s=1, vmin=-20,
                   vmax=n_grains, cmap='seismic')

        ax.set_xlim(-5, self.box_size + 5)
        ax.set_ylim(-5, self.box_size + 5)
        ax.set_zlim(-5, self.box_size + 5)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_zlabel('z (µm)')
        # plt.show()
        plt.savefig(self.store_path + '/Figs/3D_Tesselation_Epoch_{}.png'.format(epoch))
        plt.close(fig)
        time_elapse = datetime.datetime.now() - t_0
        if self.debug:
            self.logger.info('time spent on plotter for epoch {}: {}'.format(epoch, time_elapse.total_seconds()))

    def run_tesselation(self, rsa, animation=True):

        # set some variables
        status = False
        repeat = False
        packingratio = 0
        epoch = 0
        band_vol_0 = np.count_nonzero(rsa == -200)  # This volume is already affected by the First band ratio
        # So total Band ratio is band_ratio_rsa * band_ratio_tesselator

        # load some variables
        a = self.a
        b = self.b
        c = self.c
        n_grains = len(self.a)
        rve = rsa

        # define boundaries and empty rve array
        empty_rve = np.zeros((2 * self.n_pts, 2 * self.n_pts, 2 * self.n_pts), dtype=np.int32)
        empty_rve = self.rve_utils_object.gen_boundaries_3D(empty_rve)
        rve_boundaries = empty_rve.copy()  # empty rve grid with defined boundaries
        vol_0 = np.count_nonzero(empty_rve == 0)

        freepoints = np.count_nonzero(rve == 0)
        grain_idx = [i for i in range(1, n_grains + 1)]
        grain_idx_backup = grain_idx.copy()
        while freepoints > 0:
            i = 0
            np.random.shuffle(grain_idx)
            while i < len(grain_idx):
                idx = grain_idx[i]
                ellipsoid, a, b, c = self.grow(idx, a, b, c)
                grain = rve_boundaries.copy()
                grain[(ellipsoid <= 1) & (grain == 0)] = idx
                periodic_grain = self.rve_utils_object.make_periodic_3D(grain, ellipsoid, iterator=idx)
                band_vol = np.count_nonzero(rve == -200)

                if band_vol_0 > 0:
                    band_ratio = band_vol / band_vol_0
                    if band_ratio > self.band_ratio:  # Class property
                        rve[((periodic_grain == idx) & (rve == 0)) | ((periodic_grain == idx) & (rve == -200))] = idx
                    else:
                        rve[((periodic_grain == idx) & (rve == 0))] = idx
                else:
                    rve[((periodic_grain == idx) & (rve == 0))] = idx

                freepoints = np.count_nonzero(rve == 0)
                grain_vol = np.count_nonzero(rve == idx) * self.bin_size ** 3
                if freepoints == 0:
                    break

                if grain_vol > self.final_volume[idx-1] and not repeat:
                    del grain_idx[i]

                i += 1
            if not grain_idx:
                repeat = True
                grain_idx = grain_idx_backup.copy()
            if animation:
                self.tesselation_plotter(rve, epoch)
            epoch += 1
            packingratio = (1 - freepoints / vol_0) * 100
            print('packingratio:', packingratio, '%')

        if packingratio == 100:
            status = True

        # Save for further usage
        print(np.asarray(np.unique(rve, return_counts=True)).T)
        np.save(self.store_path + '/' + 'RVE_Numpy.npy', rve)
        return rve, status


if __name__ == '__main__':
    a = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    b = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
    c = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    # Define Box-dimension
    box_size = 100
    # Define resolution of Grid
    n_pts = 100

    shrinkfactor = 0.01

    x0_path = './3D_x0_list.npy'
    y0_path = './3D_y0_list.npy'
    z0_path = './3D_z0_list.npy'
    x_0 = np.load(x0_path)
    y_0 = np.load(y0_path)
    z_0 = np.load(z0_path)

    tesselation_obj = Tesselation3D(box_size, n_pts, a, b, c, x_0, y_0, z_0, shrinkfactor)
    tesselation_obj.run_tesselation()



