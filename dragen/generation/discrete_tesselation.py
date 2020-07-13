import sys
import os
import time
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd

from mpl_toolkits.mplot3d import axes3d, Axes3D
from tqdm import tqdm
from tqdm.auto import tqdm
from operator import itemgetter

from dragen.utilities.RVE_utils import RVE_utils

class DiscreteTesselation:

    def __init__(self, main_dir, box_size, points_on_edge, bandwidth = 0):
        # LOGGER initialization
        self.main_dir = main_dir
        self.box_size = box_size
        self.points_on_edge = points_on_edge
        self.step_size = box_size/points_on_edge
        self.step_half = self.step_size/2
        self.double_box = box_size*2
        self.bandwidth = bandwidth
        self.speed = 1
        self.radius = 1.0

    def tesselation(self, store_path, pt, rad, phase, convert_list, band = set(), animation = False):
        utils_obj = RVE_utils(self.box_size, self.points_on_edge, self.bandwidth)
        os.chdir(self.maindir)
        packing_ratio = 0
        boxxyz = np.arange(self.step_half, self.box_size, self.step_size)
        xyz = np.arange(-self.box_size + self.step_half, self.double_box, self.step_size)
        vol0 = len(boxxyz) * len(boxxyz) * len(boxxyz)
        print(vol0) # LOGGER
        N = len(boxxyz)
        bin = self.speed * self.box_size / (N)
        x, y, z = np.meshgrid(xyz, xyz, xyz)
        takenpoints = set()
        pid = dict() #periodic_identifier_dictionary containing coordinates of all points and translation to parent
        RSA_grains, tess_grains, idx_ist, volume = ([] for i in range(4))

        # Reconstructing RSA
        for i in tqdm(range(len(pt))):
            A1 = 1. / rad[i][0]
            B1 = 1. / rad[i][1]
            C1 = 1. / rad[i][2]
            r = np.sqrt(A1 ** 2 * (x - pt[i][0]) ** 2 + \
                        B1 ** 2 * (y - pt[i][1]) ** 2 + \
                        C1 ** 2 * (z - pt[i][2]) ** 2)
            inside = r <= self.radius
            x_per, pil_x = utils_obj.periodicity_DT(x[inside], xyz)
            y_per, pil_y = utils_obj.periodicity_DT(y[inside], xyz)
            z_per, pil_z = utils_obj.periodicity_DT(z[inside], xyz)
            grain = list(zip(x_per, y_per, z_per))
            pil_tuples = list(zip(pil_x,pil_y,pil_z))
            pid.update(dict(zip(grain, pil_tuples)))

            takenpoints.update(set(grain))
            RSA_grains.append(grain)
            tess_grains.append(grain)
            idx_list.append(i)
            volume.append(len(grain)*4/3)

        if len(band) > 0:
            takenpoints.update(band)
            print('bin in band if ')
            tess_grains.append(list(band))
            phase.append(2)

        print('Volumen RSAGrains', sum([len(tess_grain) for tess_grain in tess_grains])) # LOGGER
        pbar = tqdm(total=len(RSA_grains))
        RSA_grains_backup = RSA_grains
        idx_list_backup = idx_list
        volume_backup = volume
        rad = [list(rad_i) for rad_i in rad]
        print('Vol0: ', vol0) # LOGGER
        # Starting growth
        repeat = False
        step = 0
        fig, ax = plt.subplots()
        while packing_ratio < 1:
            if len(RSA_grains) > 0 and not repeat:
                shuffleList= list(zip(RSA_grains, idx_list, convert_list))
                np.random.shuffle(shuffleList)
                RSA_grains, idx_list, volume = zip(*shuffleList)
                RSA_grains = list(RSA_grains)
                idx_list = list(idx_list)
                volume = list(convert_list)

            for n, grain in enumerate(RSA_grains): ### iterate over grains for growth
                step = step + 1
                i = idx_list[n]
                rad[i][0] = rad[i][0] + bin
                rad[i][1] = rad[i][1] + bin
                rad[i][2] = rad[i][2] + bin
                A_max = 1. / (rad[i][0])
                B_max = 1. / (rad[i][1])
                C_max = 1. / (rad[i][2])
                r = np.sqrt(A_max ** 2 * (x - pt[i][0]) ** 2 + B_max ** 2 * (y - pt[i][1]) ** 2 + C_max ** 2 * (z - pt[i][2]) ** 2)
                inside = r <= 1
                x_pt, x_pil = utils_obj.periodicity_DT(x[inside], xyz)
                y_pt, y_pil = utils_obj.periodicity_DT(y[inside], xyz)
                z_pt, z_pil = utils_obj.periodicity_DT(z[inside], xyz)
                tuples = list(zip(x_pt, y_pt, z_pt))
                pil_tuples = list(zip(x_pil, y_pil, z_pil))
                temp_pid = dict(zip(tuples, pil_tuples))
                testellipse = set(tuples)
                hull = testellipse.difference(takenpoints)
                testlen = len(tess_grains[i])+len(hull)

                if len(hull)> 0 and testlen <= volume[n] and not repeat:
                    takenpoints.update(hull)
                    pid.update(zip(list(hull), list(map(temp_pid.get, list(hull)))))
                    tess_grains[i].extend(hull)

                elif (len(hull) == 0 or testlen > volume[n]) and not repeat:
                    maxhullsize = volume[n] - len(tess_grains[i])
                    if maxhullsize > 0:
                        hull = list(hull)
                        while len(hull) > maxhullsize:
                            del hull[-1]
                        hull = set(hull)
                        takenpoints.update(hull)
                        pid.update(zip(list(hull), list(map(temp_pid.get, list(hull)))))
                        tess_grains[i].extend(hull)

                    del RSA_grains[n]
                    del idx_list[n]
                    del volume[n]
                    pbar.update(1)

                if len(hull) ==0 and repeat:
                    del RSA_grains[n]
                    del idx_list[n]
                    del volume[n]

                elif repeat:
                    takenpoints.update(hull)
                    pid.update(zip(list(hull), list(map(temp_pid.get, list(hull)))))
                    tess_grains[i].extend(hull)

                if len(RSA_grains) == 0:
                    RSA_grains = RSA_grains_backup
                    idx_list = idx_list_backup
                    volume = volume_backup
                    repeat = True
                packing_ratio = len(takenpoints) / vol0

            if animation:
                rve = pd.DataFrame()
                graindf = [pd.DataFrame(grain, columns=['x', 'y', 'z']) for grain in tess_grains]
                for i in range(len(graindf)):
                    graindf[i]['GrainID'] = i
                    graindf[i]['vol'] = len(graindf[i])
                    graindf[i]['phaseID'] = phase[i]
                    rve = pd.concat([rve, graindf[i]])
                singleLayer = rve[rve['x'] == xyzmin]

                ax.set_xlim(0, boxsize)
                ax.set_ylim(0, boxsize)
                ax.set_xlabel('y (µm)')
                ax.set_ylabel('z (µm)')
                cm = plt.cm.get_cmap('gist_rainbow')
                ax.scatter(singleLayer['y'], singleLayer['z'], c= singleLayer['GrainID'], s=1, cmap='gist_rainbow')
                plt.savefig(storepath + '/Fig/2DRSA_'+str(step))
                plt.cla()

        # LOGGER
        #print(pid)
        #print('takenpoints',len(takenpoints))
        #print('vol0',vol0)

        volume_list = [len(TessGrain) for TessGrain in tess_grains]
        with open(storepath + '/tess_vol.csv', 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(volume_list)
        # LOGGER
        #print('len(tess_grains)',len(tess_grains))
        #print('len(volume_list)',len(volume_list))
        #print(packingratio)

        graindf = [pd.DataFrame(grain, columns=['x', 'y', 'z']) for grain in tess_grains]
        #print(graindf[0][:10])
        boxrve = pd.DataFrame()
        for i in range(len(graindf)):
            graindf[i]['GrainID'] = i
            graindf[i]['vol'] = len(graindf[i])
            graindf[i]['phaseID'] = phase[i]
        print('Storing RVE in dataframe')
        for i in tqdm(range(len(graindf))):
           boxrve = pd.concat([boxrve, graindf[i]])
        boxrve['rvesize'] = len(boxxyz)
        boxrve.sort_values(by=['x','y','z'],inplace=True)
        boxrve.reset_index(inplace=True, drop=True)
        #print(boxrve)
        boxrve.to_hdf(storepath + '/boxrve.h5', key='gb', mode='w')
        boxrve.to_csv(storepath + '/boxrve.csv', mode='w')
        flat_grain_list = [y for x in tess_grains for y in x]
        flat_grain_list.sort()
        rve = boxrve.copy()
        rve['trans'] = str(0)
        for index, row in rve.iterrows():
            point = (row['x'], row['y'], row['z'])
            trans = pid[point]
            pointx = point[0] + trans[0] * boxsize
            pointy = point[1] + trans[1] * boxsize
            pointz = point[2] + trans[2] * boxsize
            rve.at[index, 'x':'z'] = pointx, pointy, pointz
            rve.at[index, 'trans'] = str(trans)
            newpoint = (pointx, pointy, pointz)

        # LOGGER
        #print(boxrve)
        #print(rve)
        #print(max(rve['x']))
        #print(max(rve['y']))
        #print(max(rve['z']))
        #print(min(rve['x']))
        #print(min(rve['y']))
        #print(min(rve['z']))
        rve.to_hdf(storepath + '/rve.h5', key='gb', mode='w')
        rve.to_csv(storepath + '/rve.csv', mode='w')

        return rve

# TEST CODE
"""
if __name__ == '__main__':
    maindir = sys.argv[0][:-22]
    os.chdir(maindir)
    storepath = '../OutputData/' + str(datetime.datetime.now())[:10] + '_0'
    pt = [[0.5, 0.5, 0.5],[5.5, 5.5, 5.5]]
    rad = [[1,1,1],[1,1,1]]
    phase = [1,1]
    boxsize = 10
    stepsize = 1
    speed = 1
    convertlist = [5]
    boxhalf = boxsize/2
    stphalf = stepsize/2
    doublebox = 2*boxsize
    pointsonedge = 10
    tesselation(maindir, storepath, pt, rad, phase, boxsize, stepsize, speed, convertlist, boxhalf, stphalf, doublebox,
                    pointsonedge, band=set(), anim=False)
"""
