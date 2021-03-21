import numpy as np
import matplotlib.pyplot as plt
import random

class DiscreteRsa2D:
    def __init__(self, box_size, n_pts, a, b):
        self.box_size = box_size
        self.n_pts = n_pts
        xy = np.linspace(-self.box_size / 2, self.box_size + self.box_size / 2, 2*self.n_pts, endpoint=True)
        self.x_grid, self.y_grid = np.meshgrid(xy, xy)
        self.a = a
        self.b = b
        self.n_grains = len(a)

    def gen_ellipsoid(self, array, iterator):
        x_grid = self.x_grid
        y_grid = self.y_grid
        a = self.a
        b = self.b
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
        ellipse = np.sqrt((x_grid - x_0) ** 2 / (a ** 2) + (y_grid - y_0) ** 2 / (b ** 2))
        return ellipse, x_0, y_0

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

    @staticmethod
    def make_periodic(points_array, ellipse_points, iterator):
        points_array_mod = np.zeros(points_array.shape)
        points_array_mod[points_array == iterator] = iterator

        for i in range(1, 9):
            points_array_copy = np.zeros(points_array.shape)
            points_array_copy[(ellipse_points <= 1) & (points_array == -1*i)] = -100-i
            if i % 2 != 0:
                points_array_copy = np.roll(points_array_copy, n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, n_pts, axis=1)
                points_array_mod[np.where(points_array_copy == -100-i)] = iterator
            elif (i == 2) | (i == 6):
                points_array_copy = np.roll(points_array_copy, n_pts, axis=0)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
            else:
                points_array_copy = np.roll(points_array_copy, n_pts, axis=1)
                points_array_mod[np.where(points_array_copy == -100 - i)] = iterator
        return points_array_mod

    def rsa_plotter(self, array, n_grains, storepath, iterator, attempt):
        rve_x, rve_y = np.where(array >= 1)
        outside_x, outside_y = np.where(array < 0)
        unoccupied_pts_x, unoccupied_pts_y = np.where(array == 0)

        grain_tuples = [*zip(rve_x, rve_y)]
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
        plt.savefig(storepath+'/Epoch_{}_{}.png'.format(iterator, attempt))
        plt.close(fig)

    def run_rsa(self, animation=False):
        rve = np.zeros((2 * n_pts, 2 * n_pts), dtype=np.int32)
        rve = self.gen_boundaries(rve)
        x_0_list = list()
        y_0_list = list()

        i = 1
        attempt = 0
        while i < self.n_grains + 1 | attempt < 1000:
            free_points_old = np.count_nonzero(rve == 0)
            grain = np.zeros((2 * n_pts, 2 * n_pts), dtype=np.int32)
            grain = self.gen_boundaries(grain)
            ellipse, x0, y0 = self.gen_ellipsoid(rve, iterator=i - 1)
            grain[(ellipse <= 1) & (grain == 0)] = i
            periodic_grain = self.make_periodic(grain, ellipse, iterator=i)
            rve[(periodic_grain == i) & (rve == 0)] = i

            if animation:
                self.rsa_plotter(rve, self.n_grains, storepath='./', iterator=i, attempt=attempt)

            free_points = np.count_nonzero(rve == 0)
            if free_points_old - free_points != np.count_nonzero(periodic_grain):
                rve[np.where(rve == i)] = 0
                attempt = attempt + 1

            else:
                i = i + 1
                x_0_list.append(x0)
                y_0_list.append(y0)
                attempt = 0
        return rve, x_0_list, y_0_list

if __name__ == '__main__':
    a = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    b = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
    # Define Box-dimension
    box_size = 100
    # Define resolution of Grid
    n_pts = 100
    rsa_obj = DiscreteRsa2D(box_size, n_pts, a, b)
    rve, x_0_list, y_0_list = rsa_obj.run_rsa(animation=True)
    print(x_0_list)
    print(y_0_list)
    np.save('./2D_x_0', x_0_list, allow_pickle=True, fix_imports=True)
    np.save('./2D_y_0', y_0_list, allow_pickle=True, fix_imports=True)
    np.save('./2D_rve', rve, allow_pickle=True, fix_imports=True)






