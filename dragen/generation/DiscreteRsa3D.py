import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import logging
from scipy.ndimage import convolve

from dragen.utilities.RVE_Utils import RVEUtils


class DiscreteRsa3D(RVEUtils):

    def __init__(self, box_size, n_pts, a, b, c, alpha, store_path, debug=False, infobox_obj=None, progress_obj=None):

        self.box_size = box_size
        self.n_pts = n_pts
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.store_path = store_path
        self.debug = debug
        self.infobox_obj = infobox_obj
        self.progress_obj = progress_obj
        self.logger = logging.getLogger("RVE-Gen")
        self.n_grains = len(a)

        super().__init__(box_size, n_pts)

        self.x_grid, self.y_grid, self.z_grid = super().gen_grid()

    def gen_ellipsoid(self, array, iterator):
        t_0 = datetime.datetime.now()
        a = self.a
        b = self.b
        c = self.c
        alpha = self.alpha

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
        # self.infobox_obj.add_text('x_0_{}: {}, y_0_{}: {}, z_0_{}: {}'.format(iterator, x_0, iterator, y_0, iterator, z_0))
        print('x_0_{}: {}, y_0_{}: {}, z_0_{}: {}'.format(iterator, x_0, iterator, y_0, iterator, z_0))
        # print('Iterator', iterator)
        # print(a)
        # print('Länge von a', a.__len__())
        a = a[iterator]
        b = b[iterator]
        c = c[iterator]
        alpha = alpha[iterator]

        """ellipse = (x_grid - x_0) ** 2 / (a ** 2) + \
                  (y_grid - y_0) ** 2 / (b ** 2) + \
                  (z_grid - z_0) ** 2 / (c ** 2)"""

        ellipsoid = super().ellipsoid(a, b, c, x_0, y_0, z_0, alpha=alpha)

        time_elapse = datetime.datetime.now() - t_0
        # with open(self.store_path + '/rve.sta', 'a') as sta:
        #    sta.writelines('x_0_{}: {}, y_0_{}: {}, z_0_{}: {} time: {} \n'.format(iterator, x_0, iterator, y_0,
        #                                                                           iterator, z_0, time_elapse.total_seconds()))
        if self.debug:
            self.logger.info('time spent on ellipsoid{}: {}'.format(iterator, time_elapse.total_seconds()))
        return ellipsoid, x_0, y_0, z_0

    def rsa_plotter(self, array, iterator, attempt):
        plt.ioff()
        t_0 = datetime.datetime.now()
        n_grains = self.n_grains
        rve_x, rve_y, rve_z = np.where((array >= 1) | (array == -1000) | (array <= -200))
        grain_tuples = [*zip(rve_x, rve_y, rve_z)]
        grains_x = [self.x_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]
        grains_y = [self.y_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]
        grains_z = [self.z_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(grains_x, grains_y, grains_z, c=array[np.where((array > 0) | (array == -200) | (array < -200))],
        #           s=1, vmin=-20,
        #           vmax=n_grains, cmap='seismic')  # lower -200 for band grains and inclusions
        #ax.scatter(grains_x, grains_y, grains_z, c='black', s=1)

        ax.scatter(grains_x, grains_y, grains_z, c=array[np.where((array > 0) | (array == -1000) | (array <= -200))],
                   s=1, vmin=-0,
                   vmax=n_grains, cmap='seismic')

        ax.set_xlim(-5, self.box_size + 5)
        ax.set_ylim(-5, self.box_size + 5)
        ax.set_zlim(-5, self.box_size + 5)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_zlabel('z (µm)')
        # ax.view_init(90, 270)
        # plt.show()

        # ax.view_init(270, 90)  # facing in z-direction (clockwise rotation)
        # ax.view_init(90, 270) #facing against z-direction (counterclockwise rotation)
        # plt.show()

        plt.savefig(self.store_path + '/Figs/3D_Epoch_{}_{}.png'.format(iterator, attempt))

        plt.close(fig)
        time_elapse = datetime.datetime.now() - t_0
        if self.debug:
            self.logger.info('time spent on plotter for grain {}: {}'.format(iterator, time_elapse.total_seconds()))

    def run_rsa(self, band_ratio_rsa=None, banded_rsa_array=None,
                animation=False, x0_alt=None, y0_alt=None, z0_alt=None, gui=True):
        if gui:
            self.infobox_obj.emit('starting RSA')
        status = False
        bandratio = band_ratio_rsa

        if banded_rsa_array is None:
            rsa = super().gen_array()
            rsa = super().gen_boundaries_3D(rsa)

        else:
            rsa = banded_rsa_array

        band_vol_0 = np.count_nonzero(rsa == -200)
        rsa_boundaries = rsa.copy()

        x_0_list = list()
        y_0_list = list()
        z_0_list = list()

        i = 1
        attempt = 0
        free_points = np.count_nonzero(rsa == 0)
        while i < self.n_grains + 1 | attempt < free_points:
            t_0 = datetime.datetime.now()
            free_points_old = np.count_nonzero(rsa == 0)
            band_points_old = np.count_nonzero(rsa == -200)
            grain = rsa_boundaries.copy()
            backup_rsa = rsa.copy()
            ellipsoid, x0, y0, z0 = self.gen_ellipsoid(rsa, iterator=i - 1)
            grain[(ellipsoid <= 1) & ((grain == 0) | (grain == -200))] = i
            periodic_grain = super().make_periodic_3D(grain, ellipsoid, iterator=i)
            rsa[(periodic_grain == i) & ((rsa == 0) | (rsa == -200))] = i

            if animation:
                self.rsa_plotter(rsa, iterator=i, attempt=attempt)

            free_points = np.count_nonzero(rsa == 0)
            band_points = np.count_nonzero(rsa == -200)
            if band_points_old > 0:
                if (free_points_old + band_points_old - free_points - band_points != np.count_nonzero(periodic_grain)) | \
                        (band_points / band_vol_0 < bandratio):
                    # print('difference: ', free_points_old - free_points != np.count_nonzero(periodic_grain))
                    # print('ratio:', (band_points / band_vol_0 > bandratio))
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
            progress = int((float(len(x_0_list)) / self.n_grains * 100))
            if gui:
                self.progress_obj.emit(progress)

        if (len(x_0_list) == self.n_grains) or (i - 1) == self.n_grains:
            status = True
        else:
            self.logger.info("Not all grains could be placed please decrease shrinkfactor!")

        # If a list from previous Band grains is given:
        if x0_alt is None and y0_alt is None and z0_alt is None:
            pass
        else:
            x_0_list.extend(x0_alt)
            y_0_list.extend(y0_alt)
            z_0_list.extend(z0_alt)

        return rsa, x_0_list, y_0_list, z_0_list, status

    def run_rsa_clustered(self, banded_rsa_array, animation=False):
        """
        Parameters:
            banded_rsa_array: banded area with -200 everywhere
            animation: Animation flag

        Idea: Start with an normal banded rsa (-200 everywhere) and than go on and place the
        martensite islands here till a given volume is reached (percentage). After that, change the -200 back to
        zero and hand this "rve" array over to the normal rsa

        The identifier for Martensite IN the bands is below -1000 (similar to the inclusions which are below -200)
        """
        status = False

        if banded_rsa_array is None:
            self.logger.info('This cluster-rsa needs a defined band ')
        else:
            rsa = banded_rsa_array

        # Change values:
        shadow_rsa = rsa.copy()
        rsa[np.where(shadow_rsa == -200)] = 0
        rsa[np.where(shadow_rsa == 0)] = -200

        # --------------------------------------------------
        fig = plt.figure(figsize=(30, 30))
        ax = fig.gca(projection='3d')
        ax.set_aspect('auto')
        ax.voxels(rsa == -200, edgecolor="k")
        fig.savefig(self.store_path + '/' + 'Cluster.png')
        # --------------------------------------------------

        # Init
        band_vol_0 = np.count_nonzero(rsa == -200)  # Zähle -200 für initiales Gefüge
        print('Initiales, nicht belegbares Volumen:', band_vol_0)
        rsa_boundaries = rsa.copy()
        x_0_list = list()
        y_0_list = list()
        z_0_list = list()

        i = 1
        attempt = 0
        sum_attempts = 0

        # While-loop
        while (i < self.n_grains + 1) & (attempt < 30000):
            # Im Prinzip kann man Manuels while-loop kopieren, da es jetzt quasi ein riesiges, Dickes Band gibt,
            # was später wieder rückgängig gemacht wird
            t_0 = datetime.datetime.now()
            free_points_old = np.count_nonzero(rsa == 0)
            band_points_old = np.count_nonzero(rsa == -200)  # Same as band_vol_0
            grain = rsa_boundaries.copy()
            backup_rsa = rsa.copy()
            ellipsoid, x0, y0, z0 = self.gen_ellipsoid(rsa, iterator=i - 1)
            grain[(ellipsoid <= 1) & ((grain == 0) | (grain == -200))] = -(1000 + i)
            # print(np.unique(grain))
            periodic_grain = super().make_periodic_3D(grain, ellipsoid, iterator=-(1000 + i))
            # print(np.unique(periodic_grain))
            rsa[(periodic_grain == -(1000 + i)) & ((rsa == 0) | (rsa == -200))] = -(1000 + i)
            #print(np.asarray(np.unique(periodic_grain, return_counts=True)).T)
            #print(free_points_old)
            #print(band_points_old)

            if animation:
                self.rsa_plotter(rsa, iterator=-(1000 + i), attempt=attempt)

            free_points = np.count_nonzero(rsa == 0)
            band_points = np.count_nonzero(rsa == -200)
            #print(free_points)
            #print(band_points)
            #print(free_points_old+band_points_old-free_points-band_points)
            if band_points_old > 0:
                # Schnitte zulassen...Es geht wohl nicht anders
                # > xy x grain_points heißt, dass mindestens XX% des Korns im freien Raum platziert werden müssen
                if ((free_points_old + band_points_old - free_points - band_points) != 1.0 * np.count_nonzero(periodic_grain)) | \
                        (band_points / band_vol_0 < 0.9):  # Prozentbereich nach außen muss möglich sein (90%)
                    print('Attempt: ', attempt)
                    rsa = backup_rsa.copy()
                    attempt = attempt + 1

                else:
                    x_0_list.append(x0)
                    y_0_list.append(y0)
                    z_0_list.append(z0)
                    i = i + 1
                    sum_attempts = sum_attempts + attempt
                    attempt = 0
                    if self.debug:
                        time_elapse = datetime.datetime.now() - t_0
                        self.logger.info(
                            'total time needed for placement of grain {}: {}'.format(i, time_elapse.total_seconds()))
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
                    sum_attempts = sum_attempts + attempt
                    attempt = 0
                    if self.debug:
                        time_elapse = datetime.datetime.now() - t_0
                        self.logger.info(
                            'total time needed for placement of grain {}: {}'.format(i, time_elapse.total_seconds()))

        # Mindestens 90% der Körner müssen platziert werden.
        if len(x_0_list) >= 0.9*self.n_grains:
            status = True
        else:
            self.logger.info("Not all grains could be placed please decrease shrinkfactor!")

        # Change -200 in rsa_array back to 0
        print(np.asarray(np.unique(rsa, return_counts=True)).T)
        rsa[np.where(rsa == -200)] = 0
        print(np.asarray(np.unique(rsa, return_counts=True)).T)
        with open(self.store_path + '/rve.sta', 'a') as sta:
            sta.writelines('Total number of attempts needed: {}\n\n'.format(sum_attempts))
        return rsa, x_0_list, y_0_list, z_0_list, status

    def run_rsa_inclusions(self, rve, animation=False):
        """
        RSA-Algorithm to place Inclusions in the RVE: The main difference between the inclusions and "normal" (e.g.
        ferrite-grains is, that the inclusions don't grow and cannot be placed on grain boundaries, only directly in a
        grain. This is because the inclusions serve as a initiation site for grains during the recrystallization.

        The inclusions are sampled from the passed GAN-object directly and are assigned to one phase, which is purely
        elastic (so identifier smaller -200 and therefore phaseID = 2 at the moment (14.04.2021)

        Parameters:
            -rve: completely tesselated rve after the tesselation and before the mesher
            -animation: Flag for animation

        Behavior very similar to the "vanilla" rsa. The gen-ellipsoid method places where the array is 0. So create an
        array which has 0's everywhere except for the grain boundaries. Use this for place-determination, place in the
        normal rve

        Creation of this 0/1 rve either with sobel-Filter/Convolution of the 3D-Array or with sets

        The vanilla-RSA works with free_points_old - free_points == np.count_nonzero(grain_points)
        """

        # Startup
        status = False
        fil = np.asarray([[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                          [[-1, -1, -1], [-1, 26, -1], [-1, -1, -1]],
                          [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]])
        coords = np.where(convolve(rve, fil, mode='reflect') > 0)  # Edge kernel for grain boundary detection
        new_rve = super().gen_array()
        new_rve[coords] = 1000  # High value - Has 1000 for edges, and 0 elsewhere
        new_rve = super().gen_boundaries_3D(new_rve)
        inc_rve = rve.copy()

        i = 1
        attempt = 0
        while (i < self.n_grains + 1) & (attempt < 5000):
            grain = new_rve.copy()
            backup = inc_rve.copy()
            ellipsoid, x0, y0, z0 = self.gen_ellipsoid(new_rve, iterator=i - 1)
            grain[(ellipsoid <= 1) & ((grain == 0) | (grain == -200))] = -(200 + i)
            periodic_grain = super().make_periodic_3D(grain, ellipsoid, iterator=-(200 + i))

            inc_rve[(periodic_grain == -(200 + i)) & ((new_rve == 0) | (new_rve == -200))] = -(
                    200 + i)  # -for Inclusions

            # Checking
            check = set(rve[np.where(inc_rve == -(200 + i))])

            if check.__len__() > 1:
                print('Inclusion cuts grain boundary! - Cutted Grains: {}'.format(check))
                inc_rve = backup.copy()
                attempt = attempt + 1
            else:
                print('Placed inclusion successfully in grain {}'.format(check))
                i += 1
                rve = inc_rve.copy()  # To recognize inclusions

        status = True
        return inc_rve, status


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
