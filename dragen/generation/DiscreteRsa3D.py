import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import sys
import datetime

class DiscreteRsa3D:
    def __init__(self, box_size, n_pts, a, b, c, debug):
        self.box_size = box_size
        self.n_pts = n_pts
        xyz = np.linspace(-box_size / 2, box_size + box_size / 2, 2*self.n_pts, endpoint=True)
        self.x_grid, self.y_grid, self.z_grid = np.meshgrid(xyz, xyz, xyz)
        self.a = a
        self.b = b
        self.c = c
        self.n_grains = len(a)
        self.debug = debug

    def gen_ellipsoid(self, array, iterator):
        t_0 = datetime.datetime.now()
        x_grid = self.x_grid
        y_grid = self.y_grid
        z_grid = self.z_grid
        a = self.a
        b = self.b
        c = self.c
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
        ellipse = np.sqrt((x_grid - x_0) ** 2 / (a ** 2) +
                          (y_grid - y_0) ** 2 / (b ** 2) +
                          (z_grid - z_0) ** 2 / (c ** 2))
        time_elapse = datetime.datetime.now() - t_0
        if self.debug:
            print('time spent on ellipse{}: {}'.format(iterator, time_elapse.total_seconds()))
        return ellipse, x_0, y_0, z_0

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

    def rsa_plotter(self, array, storepath, iterator, attempt):
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
        ax.scatter(grains_x, grains_y, grains_z, c=array[np.where(array > 0)], s=1, vmin=0, vmax=n_grains, cmap='seismic')
        #ax.scatter(grains_x, grains_y, grains_z, c='r', s=1)

        ax.set_xlim(-5, self.box_size + 5)
        ax.set_ylim(-5, self.box_size + 5)
        ax.set_zlim(-5, self.box_size + 5)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_zlabel('z (µm)')
        #plt.show()
        plt.savefig(storepath+'/3D_Epoch_{}_{}.png'.format(iterator, attempt))
        plt.close(fig)
        time_elapse = datetime.datetime.now() - t_0
        if self.debug:
            print('time spent on plotter for grain {}: {}'.format(iterator, time_elapse.total_seconds()))

    def run_rsa(self, animation=False):
        rve = np.zeros((2 * n_pts, 2 * n_pts, 2 * n_pts), dtype=np.int32)
        rve = self.gen_boundaries(rve)
        x0_list = list()
        y0_list = list()
        z0_list = list()
        rve_boundaries = rve.copy()
        i = 1
        attempt = 0
        while i < self.n_grains + 1 | attempt < 1000:
            t_0 = datetime.datetime.now()
            free_points_old = np.count_nonzero(rve == 0)
            grain = rve_boundaries.copy()
            ellipse, x0, y0, z0 = self.gen_ellipsoid(rve, iterator=i-1)
            grain[(ellipse <= 1) & (grain == 0)] = i
            periodic_grain = self.make_periodic(grain, ellipse, iterator=i)
            rve[(periodic_grain == i) & (rve == 0)] = i

            if animation:
                rsa_obj.rsa_plotter(rve, storepath='./', iterator=i, attempt=attempt)

            free_points = np.count_nonzero(rve == 0)
            if free_points_old - free_points != np.count_nonzero(periodic_grain):
                rve[np.where(rve == i)] = 0
                attempt = attempt + 1

            else:
                x0_list.append(x0)
                y0_list.append(y0)
                z0_list.append(z0)
                i = i + 1
                attempt = 0
                if self.debug:
                    time_elapse = datetime.datetime.now() - t_0
                    print('total time needed for placement of grain {}: {}'.format(i, time_elapse.total_seconds()) )
        return x0_list, y0_list, z0_list


if __name__ == '__main__':
    a = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    b = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    c = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    animation = True
    debug = False
    # Define Box-dimension
    box_size = 100
    # Define resolution of Grid
    n_pts = 50
    rsa_obj = DiscreteRsa3D(box_size, n_pts, a, b, c, debug)
    x_0_list, y_0_list, z_0_list = rsa_obj.run_rsa(animation)
    np.save('./3D_x0_list', x_0_list, allow_pickle=True, fix_imports=True)
    np.save('./3D_y0_list', y_0_list, allow_pickle=True, fix_imports=True)
    np.save('./3D_z0_list', z_0_list, allow_pickle=True, fix_imports=True)









