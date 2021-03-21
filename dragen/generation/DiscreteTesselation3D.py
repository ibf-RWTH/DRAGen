import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys

class Tesselation:
    def __init__(self, box_size, n_pts, a, b, c, x_0, y_0, z_0, shrinkfactor, debug=False):
        self.box_size = box_size
        self.bin_size = box_size/n_pts
        self.n_pts = n_pts
        self.a = a
        self.b = b
        self.c = c
        self.final_volume = [4 / 3 * np.pi * a[i] * b[i] * c[i]/shrinkfactor for i in range(len(a))]
        xyz = np.linspace(-self.box_size / 2, self.box_size + self.box_size / 2, 2 * self.n_pts, endpoint=True)
        self.x_grid, self.y_grid, self.z_grid = np.meshgrid(xyz, xyz, xyz)
        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0
        self.debug = debug
        self.n_grains = len(a)

    def gen_boundaries(self, points_array):
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
            print('time spent on gen_boundaries: {}'.format(time_elapse.total_seconds()))
        return points_array

    def grow(self, iterator, frame):
        x_grid = self.x_grid
        y_grid = self.y_grid
        z_grid = self.z_grid
        x_0 = self.x_0[iterator-1]
        y_0 = self.y_0[iterator-1]
        z_0 = self.z_0[iterator-1]
        a = self.a[iterator-1]+(frame+1)*self.bin_size
        b = self.b[iterator-1]+(frame+1)*self.bin_size
        c = self.c[iterator-1]+(frame+1)*self.bin_size

        ellipse = (x_grid - x_0) ** 2 / (a ** 2) + (y_grid - y_0) ** 2 / (b ** 2) + (z_grid - z_0) ** 2 / (c ** 2)
        return ellipse

    def make_periodic(self, points_array, ellipse_points, iterator):
        points_array_mod = np.zeros(points_array.shape)
        points_array_mod[points_array == iterator] = iterator
        t_0 = datetime.datetime.now()
        for i in range(1, 27): #move points in x,y and z dir
            points_array_copy = np.zeros(points_array.shape)
            points_array_copy[(ellipse_points <= 1) & (points_array == -1*i)] = -100-i
            if (i == 1) | (i == 3) | (i == 7) | (i == 9) | \
                    (i == 18) | (i == 20) | (i == 24) | (i == 26):  # move points in x,y and z direction
                points_array_copy = np.roll(points_array_copy, n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, n_pts, axis=1)
                points_array_copy = np.roll(points_array_copy, n_pts, axis=2)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 10) | (i == 12) | (i == 15) | (i == 17):  # move points in x and y direction
                points_array_copy = np.roll(points_array_copy, n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, n_pts, axis=1)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 4) | (i == 6) | (i == 21) | (i == 23):  # move points in x and z direction
                points_array_copy = np.roll(points_array_copy, n_pts, axis=1)
                points_array_copy = np.roll(points_array_copy, n_pts, axis=2)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 2) | (i == 8) | (i == 19) | (i == 25):  # move points in y and z direction
                points_array_copy = np.roll(points_array_copy, n_pts, axis=0)
                points_array_copy = np.roll(points_array_copy, n_pts, axis=2)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 13) | (i == 14) :  # move points in x direction
                points_array_copy = np.roll(points_array_copy, n_pts, axis=1)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 11) | (i == 16):  # move points in y direction
                points_array_copy = np.roll(points_array_copy, n_pts, axis=0)
                points_array_mod[points_array_copy == -100-i] = iterator
            elif (i == 5) | (i == 22):  # move points in z direction
                points_array_copy = np.roll(points_array_copy, n_pts, axis=2)
                points_array_mod[points_array_copy == -100-i] = iterator
        time_elapse = datetime.datetime.now()-t_0
        if self.debug:
            print('time spent on periodicity for grain {}: {}'.format(iterator, time_elapse.total_seconds()) )
        return points_array_mod

    def tesselation_plotter(self, array, storepath, epoch):
        t_0 = datetime.datetime.now()
        n_grains = self.n_grains
        rve_x, rve_y, rve_z = np.where(array >= 1)
        grain_tuples = [*zip(rve_x, rve_y, rve_z)]
        grains_x = [self.x_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]
        grains_y = [self.y_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]
        grains_z = [self.z_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(grains_x, grains_y, grains_z, c=array[np.where(array > 0)], s=1, vmin=0, vmax=n_grains,
                   cmap='seismic')

        ax.set_xlim(-5, self.box_size + 5)
        ax.set_ylim(-5, self.box_size + 5)
        ax.set_zlim(-5, self.box_size + 5)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_zlabel('z (µm)')
        # plt.show()
        plt.savefig(storepath + '/3D_Tesselation_Epoch_{}.png'.format(epoch))
        plt.close(fig)
        time_elapse = datetime.datetime.now() - t_0
        if self.debug:
            print('time spent on plotter for epoch {}: {}'.format(epoch, time_elapse.total_seconds()))

    def run_tesselation(self):
        rve = np.zeros((2 * n_pts, 2 * n_pts, 2 * n_pts), dtype=np.int32)
        rve = tesselation_obj.gen_boundaries(rve)
        rve_boundaries = rve.copy()  # empty rve grid with defined boundaries
        vol_0 = np.count_nonzero(rve == 0)
        epoch = 0
        storepath = './'
        freepoints = np.count_nonzero(rve == 0)
        n_grains = len(self.a)
        grain_idx = [i for i in range(1, n_grains + 1)]
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
                grain_vol = np.count_nonzero(rve == idx) * self.bin_size ** 3
                if freepoints == 0:
                    break
                if not grain_idx:
                    grain_idx = grain_idx_backup.copy()
                    'grain_idx was restored since all grains reached final volume'
                if (freepoints_old == freepoints) | (grain_vol > self.final_volume[idx - 1]):
                    print('freepoints_old:', freepoints_old)
                    print('freepoints:', freepoints)
                    print('current vol:', grain_vol)
                    print('final vol:', self.final_volume[idx - 1])
                    del grain_idx[i]
                i += 1

            tesselation_obj.tesselation_plotter(rve, storepath, epoch)
            epoch += 1
            packingratio = (1 - freepoints / vol_0) * 100
            print('packingratio:', packingratio, '%')

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

    tesselation_obj = Tesselation(box_size, n_pts, a, b, c, x_0, y_0, z_0, shrinkfactor)
    tesselation_obj.run_tesselation()



