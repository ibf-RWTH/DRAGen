import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import random
import datetime


from scipy.ndimage import convolve
from tqdm import tqdm
from dragen.utilities.Helpers import HelperFunctions
from dragen.utilities.InputInfo import RveInfo


class DiscreteRsa3D(HelperFunctions):

    def __init__(self, a, b, c, alpha):

        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.n_grains = len(a)

        super().__init__()
        """n_x = RveInfo.n_pts
        n_y = RveInfo.n_pts
        n_z = RveInfo.n_pts
        if RveInfo.n_pts_y is not None:
            n_y = RveInfo.n_pts_y
        if RveInfo.n_pts_z is not None:
            n_y = RveInfo.n_pts_z
        shape = (n_x, n_y, n_z)"""

        self.x_grid, self.y_grid, self.z_grid = super().gen_grid_new()
        if not RveInfo.gui_flag:
            self.pbar = tqdm(total=self.n_grains)

    def gen_ellipsoid(self, array, iterator):
        t_0 = datetime.datetime.now()
        a = self.a
        b = self.b
        c = self.c
        alpha = self.alpha

        unoccupied_pts_x, unoccupied_pts_y, unoccupied_pts_z = np.where(array == 0)
        unoccupied_tuples = [*zip(unoccupied_pts_x, unoccupied_pts_y, unoccupied_pts_z)]


        """unoccupied_area_x = [self.x_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]][unoccupied_tuples_i[2]]
                             for unoccupied_tuples_i in unoccupied_tuples]
        unoccupied_area_y = [self.y_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]][unoccupied_tuples_i[2]]
                             for unoccupied_tuples_i in unoccupied_tuples]
        unoccupied_area_z = [self.z_grid[unoccupied_tuples_i[0]][unoccupied_tuples_i[1]][unoccupied_tuples_i[2]]
                             for unoccupied_tuples_i in unoccupied_tuples]
        """
        #print(unoccupied_pts_z)
        #sys.exit()
        x_0 = random.choice(unoccupied_pts_x)
        y_0 = random.choice(unoccupied_pts_y)
        z_0 = random.choice(unoccupied_pts_z)

        #x_0 = int(unoccupied_area_x[idx])
        #y_0 = int(unoccupied_area_y[idx])
        #z_0 = int(unoccupied_area_z[idx])

        a = a[iterator]
        b = b[iterator]
        c = c[iterator]
        alpha = alpha[iterator]

        ellipsoid = super().ellipsoid(a, b, c, alpha=alpha)

        inside = ellipsoid <= 1
        ellipsoid_array = np.zeros_like(ellipsoid, dtype='int16')
        ellipsoid_array[inside] = 1
        time_elapse = datetime.datetime.now() - t_0
        if RveInfo.debug:
            RveInfo.LOGGER.info('time spent on ellipsoid{}: {}'.format(iterator, time_elapse.total_seconds()))
        return ellipsoid, x_0, y_0, z_0

    def rsa_plotter(self, array, iterator, attempt):
        plt.ioff()
        t_0 = datetime.datetime.now()
        n_grains = self.n_grains
        rve_x, rve_y, rve_z = np.where((array >= 1) | (array == -1000) | (array < -200))
        grain_tuples = [*zip(rve_x, rve_y, rve_z)]
        grains_x = [self.x_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]
        grains_y = [self.y_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]
        grains_z = [self.z_grid[grain_tuples_i[0]][grain_tuples_i[1]][grain_tuples_i[2]]
                    for grain_tuples_i in grain_tuples]

        free_rve_x, free_rve_y, free_rve_z = np.where((array == -26))
        free_space_tuples = [*zip(free_rve_x, free_rve_y, free_rve_z)]

        free_space_x = [self.x_grid[free_space_tuples_i[0]][free_space_tuples_i[1]][free_space_tuples_i[2]]
                        for free_space_tuples_i in free_space_tuples]
        free_space_y = [self.y_grid[free_space_tuples_i[0]][free_space_tuples_i[1]][free_space_tuples_i[2]]
                        for free_space_tuples_i in free_space_tuples]
        free_space_z = [self.z_grid[free_space_tuples_i[0]][free_space_tuples_i[1]][free_space_tuples_i[2]]
                        for free_space_tuples_i in free_space_tuples]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(grains_x, grains_y, grains_z, c=array[np.where((array > 0) | (array == -1000) | (array < -200))],
                   s=1, vmin=-0,
                   vmax=n_grains, cmap='seismic')  # lower -200 for band grains and inclusions
        ax.scatter(free_space_x, free_space_y, free_space_z, color='grey', alpha=0.01)

        ax.set_xlim(-5, RveInfo.box_size + 5)
        ax.set_ylim(-5, RveInfo.box_size + 5)
        ax.set_zlim(-5, RveInfo.box_size + 5)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_zlabel('z (µm)')
        # ax.view_init(90, 270)
        # plt.show()

        # ax.view_init(270, 90)  # facing in z-direction (clockwise rotation)
        # ax.view_init(90, 270) #facing against z-direction (counterclockwise rotation)
        #plt.show()

        plt.savefig(RveInfo.store_path + '/Figs/3D_Epoch_{}_{}.png'.format(iterator, attempt))
        plt.close(fig)
        time_elapse = datetime.datetime.now() - t_0
        if RveInfo.debug:
            RveInfo.LOGGER.info('time spent on plotter for grain {}: {}'.format(iterator, time_elapse.total_seconds()))

    def run_rsa(self, band_ratio_rsa=None, banded_rsa_array=None, x0_alt=None, y0_alt=None, z0_alt=None):
        if RveInfo.gui_flag:
            RveInfo.infobox_obj.emit('starting RSA')
        status = False
        bandratio = band_ratio_rsa

        if banded_rsa_array is None:
            rsa = super().gen_array_new()

        else:
            rsa = banded_rsa_array

        band_vol_0 = np.count_nonzero(rsa == -200)
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
            backup_rsa = rsa.copy()
            ellipsoid, x0, y0, z0 = self.gen_ellipsoid(rsa, iterator=i - 1)
            grain = np.zeros_like(ellipsoid, dtype='int16')
            grain[ellipsoid <= 1] = i
            # check that at least one element is not 0
            if np.count_nonzero(grain) == 0:
                # if there is no element set to the value i give set one element at x_0, y_0, z_0 to i manually
                grain[x0, y0, z0] = i

            periodic_grain = super().make_periodic_3D_new(grain, x0, y0, z0)

            rsa[(periodic_grain == i) & ((rsa == 0) | (rsa == -200))] = i
            if RveInfo.anim_flag:
                self.rsa_plotter(rsa, iterator=i, attempt=attempt)

            free_points = np.count_nonzero(rsa == 0)
            band_points = np.count_nonzero(rsa == -200)

            if band_points_old > 0:
                intersecting_pts = np.count_nonzero(periodic_grain) - (free_points_old + band_points_old - free_points - band_points)
                intersecting_ratio = intersecting_pts/np.count_nonzero(periodic_grain)
                if intersecting_ratio > RveInfo.allowed_intersection_ratio:
                    rsa = backup_rsa.copy()
                    attempt = attempt + 1

                else:
                    x_0_list.append(x0)
                    y_0_list.append(y0)
                    z_0_list.append(z0)
                    i = i + 1
                    attempt = 0
                    if RveInfo.debug:
                        time_elapse = datetime.datetime.now() - t_0
                        RveInfo.LOGGER.info(
                            'total time needed for placement of grain {}: {}'.format(i, time_elapse.total_seconds()))
            else:
                # free points old - free points should equal non zero in periodic grain
                intersecting_pts = np.count_nonzero(periodic_grain) - (free_points_old - free_points)
                intersecting_ratio = intersecting_pts / np.count_nonzero(periodic_grain)

                if intersecting_ratio > 0.01:
                    rsa = backup_rsa.copy()
                    attempt = attempt + 1
                else:
                    x_0_list.append(x0)
                    y_0_list.append(y0)
                    z_0_list.append(z0)
                    i = i + 1
                    attempt = 0
                    if RveInfo.debug:
                        time_elapse = datetime.datetime.now() - t_0
                        RveInfo.LOGGER.info(
                            'total time needed for placement of grain {}: {}'.format(i, time_elapse.total_seconds()))
            progress = int((float(len(x_0_list))/self.n_grains * 100))
            if RveInfo.gui_flag:
                RveInfo.progress_obj.emit(progress)
            else:
                inc = i - self.pbar.n
                self.pbar.update(n=inc)

        if (len(x_0_list) == self.n_grains) or (i - 1) == self.n_grains:
            status = True
        else:
            RveInfo.LOGGER.info("Not all grains could be placed please decrease shrinkfactor!")

        # If a list from previous Band grains is given:
        if x0_alt is None and y0_alt is None and z0_alt is None:
            pass
        else:
            x_0_list.extend(x0_alt)
            y_0_list.extend(y0_alt)
            z_0_list.extend(z0_alt)

        return rsa, x_0_list, y_0_list, z_0_list, status

    def run_rsa_clustered(self, previous_rsa, band_array, animation=True, startindex=0):
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

        if previous_rsa is None:
            RveInfo.LOGGER.info('This cluster-rsa needs a defined band ')
        else:
            # Use this array for checking, because here is only the band "free"
            rsa = band_array.copy()
            # Place in this Array
            placement_rsa = previous_rsa.copy()

        # Change values:
        shadow_rsa = rsa.copy()
        rsa[np.where(shadow_rsa == -200)] = 0
        rsa[np.where(shadow_rsa == 0)] = -200
        # Init
        band_vol_0 = np.count_nonzero(rsa == -200)  # Zähle -200 für initiales Gefüge
        print('Initiales, nicht belegbares Volumen:', band_vol_0)
        x_0_list = list()
        y_0_list = list()
        z_0_list = list()

        rsa_boundaries = rsa.copy()

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
            grain = np.zeros_like(ellipsoid, dtype='int16')
            grain[ellipsoid <= 1] = i
            periodic_grain = super().make_periodic_3D_new(grain, x0, y0, z0)

            rsa2 = rsa.copy()
            rsa[(periodic_grain == -(1000 + i + startindex)) & ((rsa == 0) | (rsa == -200))] = -(1000 + i + startindex)

            free_points = np.count_nonzero(rsa == 0)
            band_points = np.count_nonzero(rsa == -200)

            if band_points_old > 0:
                # > xy x grain_points heißt, dass mindestens XX% des Korns im freien Raum platziert werden müssen
                ##if ((free_points_old + band_points_old - free_points - band_points) != 1.0 * np.count_nonzero(periodic_grain)) | \
                ##        (band_points / band_vol_0 < 0.90):  # Prozentbereich nach außen muss möglich sein (90%)
                if ((free_points_old + band_points_old - free_points - band_points) != 1.0 * np.count_nonzero(periodic_grain)) and \
                        (band_points / band_vol_0 < 0.90):  # Prozentbereich nach außen muss möglich sein (90%)
                    print('Attempt: ', attempt)
                    rsa = backup_rsa.copy()
                    attempt = attempt + 1
                else:
                    # Place now the grain in the "real" rsa
                    placement_rsa[(periodic_grain == -(1000 + i + startindex)) & ((rsa2 == 0) | (rsa2 == -200))] = -(1000 + i + startindex)
                    x_0_list.append(x0)
                    y_0_list.append(y0)
                    z_0_list.append(z0)
                    i = i + 1
                    sum_attempts = sum_attempts + attempt
                    if RveInfo.anim_flag:
                        self.rsa_plotter(placement_rsa, iterator=-(1000 + i + startindex), attempt=attempt)
                    attempt = 1


        # Mindestens 90% der Körner müssen platziert werden.
        if len(x_0_list) >= 0.9 * self.n_grains:
            status = True
        else:
            RveInfo.LOGGER.info("Not all grains could be placed please decrease shrinkfactor!")

        # Change -200 in rsa_array back to 0
        print(np.asarray(np.unique(placement_rsa, return_counts=True)).T)
        placement_rsa[np.where(placement_rsa == -200)] = 0
        print(np.asarray(np.unique(placement_rsa, return_counts=True)).T)

        rsa[np.where(placement_rsa == -200)] = 0

        with open(RveInfo.store_path + '/rve.log', 'a') as log:
            log.writelines('Total number of attempts needed: {}\n\n'.format(sum_attempts))

        return placement_rsa, x_0_list, y_0_list, z_0_list, status

    def run_rsa_inclusions(self, rve):
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
        fil = np.asarray([[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                          [[-1, -1, -1], [-1, 26, -1], [-1, -1, -1]],
                          [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]])
        coords = np.where(convolve(rve, fil, mode='reflect') > 0)  # Edge kernel for grain boundary detection
        new_rve = super().gen_array_new()
        new_rve[coords] = 1000  # High value - Has 1000 for edges, and 0 elsewhere
        #new_rve = super().gen_boundaries_3D(new_rve)
        inc_rve = rve.copy()

        i = 1
        attempt = 0
        while (i < self.n_grains + 1) & (attempt < 5000):
            grain = new_rve.copy()
            backup = inc_rve.copy()
            ellipsoid, x0, y0, z0 = self.gen_ellipsoid(new_rve, iterator=i - 1)

            grain = np.zeros_like(ellipsoid, dtype='int16')
            grain[ellipsoid <= 1] = i
            periodic_grain = super().make_periodic_3D_new(grain, x0, y0, z0)

            inc_rve[(periodic_grain == -(200 + i)) & ((new_rve == 0) | (new_rve == -200))] = \
                -(200 + i)  # -for Inclusions

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
