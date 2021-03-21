import matplotlib.pyplot as plt
import numpy as np

import sys

class Tesselation:
    def __init__(self, box_size, n_pts, a, b, x_0, y_0, shrinkfactor):
        self.box_size = box_size
        self.bin_size = box_size/n_pts
        self.n_pts = n_pts
        self.a = a
        self.b = b
        self.final_volume = [np.pi * a[i] * b[i]/shrinkfactor for i in range(len(a))]
        xy = np.linspace(-self.box_size / 2, self.box_size + self.box_size / 2, 2 * self.n_pts, endpoint=True)
        self.x_grid, self.y_grid = np.meshgrid(xy, xy)
        self.x_0 = x_0
        self.y_0 = y_0

    def gen_boundaries(self, points_array):
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

    def grow(self, iterator, frame):
        x_grid = self.x_grid
        y_grid = self.y_grid
        x_0 = self.x_0[iterator-1]
        y_0 = self.y_0[iterator-1]
        a = self.a[iterator-1]+(frame+1)*self.bin_size
        b = self.b[iterator-1]+(frame+1)*self.bin_size

        ellipse = (x_grid - x_0) ** 2 / (a ** 2) + (y_grid - y_0) ** 2 / (b ** 2)
        return ellipse

    @staticmethod
    def make_periodic(points_array, ellipse_points, iterator):
        points_array_mod = np.zeros(points_array.shape)
        points_array_mod[points_array == iterator] = iterator

        for i in range(1, 9):
            points_array_copy = np.zeros(points_array.shape)
            points_array_copy[(ellipse_points <= 1) & (points_array == -1 * i)] = -100 - i
            if i % 2 != 0:
                points_array_copy = np.roll(points_array_copy, n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, n_pts, axis=1)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
            elif (i == 2) | (i == 6):
                points_array_copy = np.roll(points_array_copy, n_pts, axis=0)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
            else:
                points_array_copy = np.roll(points_array_copy, n_pts, axis=1)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
        return points_array_mod

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
        plt.savefig(storepath+'/2D_Tesselation_Epoch_{}.png'.format(epoch))
        plt.close(fig)

    def run_tesselation(self):
        rve = np.zeros((2 * n_pts, 2 * n_pts), dtype=np.int32)
        rve = tesselation_obj.gen_boundaries(rve)
        rve_boundaries = rve.copy()  # empty rve grid with defined boundaries
        vol_0 = np.count_nonzero(rve == 0)
        epoch = 0
        storepath = './'
        freepoints = np.count_nonzero(rve == 0)
        n_grains = len(self.a)
        grain_idx = [i for i in range(1, n_grains+1)]
        grain_idx_backup = grain_idx.copy()
        while freepoints > 0:
            i = 0
            np.random.shuffle(grain_idx)
            while i < len(grain_idx):
                idx = grain_idx[i]
                freepoints_old = freepoints
                ellipse = tesselation_obj.grow(idx, epoch)
                grain = rve_boundaries.copy()
                grain[(ellipse <= 1) & (grain == 0)] = idx
                periodic_grain = tesselation_obj.make_periodic(grain, ellipse, iterator=idx)
                rve[(periodic_grain == idx) & (rve == 0)] = idx
                freepoints = np.count_nonzero(rve == 0)
                grain_vol = np.count_nonzero(rve == idx)*self.bin_size**2

                if not grain_idx:
                    grain_idx = grain_idx_backup.copy()
                    'grain_idx was restored since all grains reached final volume'
                if (freepoints_old == freepoints) | (grain_vol > self.final_volume[idx-1]):
                    print('!')
                    del grain_idx[i]
                i += 1

            tesselation_obj.tesselation_plotter(rve, storepath, epoch)
            epoch += 1
            packingratio = (1-freepoints/vol_0)*100
            print('packingratio:', packingratio, '%')

if __name__ == '__main__':
    a = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    b = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
    # Define Box-dimension
    box_size = 100
    # Define resolution of Grid
    n_pts = 256
    shrinkfactor = 0.42
    x0_path = './2D_x_0.npy'
    y0_path = './2D_y_0.npy'
    x_0 = np.load(x0_path)
    y_0 = np.load(y0_path)
    tesselation_obj = Tesselation(box_size, n_pts, a, b, x_0, y_0, shrinkfactor)
    tesselation_obj.run_tesselation()



