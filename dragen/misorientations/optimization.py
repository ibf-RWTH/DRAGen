import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
import random as r
import scipy.stats as stats
import damask
import matplotlib.pyplot as plt

def calc_miso1(grains,pairs,degrees):
    '''
    Function that calculates the misorientation data of a particular dataset
    Used only on the initial stage
    :param data: Array which contains grains' information
    :param pairs: Array of adjacent grains
    :param degrees: Input Euler Angles in degrees or not
    :return: Array of R-F vectors sorted by the pairs order
    '''
    start_time = time.time()
    #grainid = np.array([data['GrainID']]).reshape((-1, 1))
    grainid=grains[:,7]

    rodvecs=np.empty((0,3))
    #euler_angles=np.empty((0,3))
    for i in range(0, len(pairs)):
        x = int(pairs[i, 0])
        y = int(pairs[i, 1])
        id1 = int(np.where(grainid == x)[0])
        id2 = int(np.where(grainid == y)[0])
        o1 = np.array([grains[id1, 4], grains[id1, 5], grains[id1, 6]])
        o2 = np.array([grains[id2, 4], grains[id2, 5], grains[id2, 6]])

        if degrees:
            a = damask.Orientation.from_Euler_angles(phi=o1, degrees=True, family='cubic')
            b = damask.Orientation.from_Euler_angles(phi=o2, degrees=True, family='cubic')
        else:
            a = damask.Orientation.from_Euler_angles(phi=o1, degrees=False, family='cubic')
            b = damask.Orientation.from_Euler_angles(phi=o2, degrees=False, family='cubic')

        c = a.disorientation(b)
        r_v = damask.Orientation.as_Rodrigues_vector(c,compact=True)
        rodvecs=np.append(rodvecs,[r_v],0)
    print(" %s seconds " % (time.time() - start_time))

    return rodvecs

def disorientation(o1, o2, degrees):
    '''
    Simple function which calculates the disorientation angle and axis from
    2 orientation data
    o1= orientation of grain 1, expressed in Euler Angles
    o2= orientation of grain 2, expressed in Euler Angles
    degrees: Whether Euler Angles are expressed in degrees or not
    return:
    '''
    if degrees:
        a = damask.Orientation.from_Euler_angles(phi=o1, degrees=True, family='cubic')
        b = damask.Orientation.from_Euler_angles(phi=o2, degrees=True, family='cubic')
    else:
        a = damask.Orientation.from_Euler_angles(phi=o1, degrees=False, family='cubic')
        b = damask.Orientation.from_Euler_angles(phi=o2, degrees=False, family='cubic')

    c = a.disorientation(b)
    #a_a = damask.Orientation.as_axis_angle(c, degrees=True, pair=True)
    a_a=damask.Orientation.as_Rodrigues_vector(c,compact=True)

    ax = a_a[0]
    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]
    angle = np.float(a_a[1])

    return angle, ax1, ax2, ax3


def calc_miso(i, grains, pairs, degrees):
    '''
    Misorientation data calculation between two grains, derived from the pairs array
    i: iteration parameter
    grains: Grain EBSD data array
    pairs: Grains' pairs array
    degrees: Whether the Euler Angles in grains array are expressed in degrees or not
    Found it convinient to create this and disorientation function for the multiprocessing
    '''
    g1 = pairs[i, 0] - 1
    g2 = pairs[i, 1] - 1
    o1 = np.array([grains[g1, 4], grains[g1, 5], grains[g1, 6]])
    o2 = np.array([grains[g2, 4], grains[g2, 5], grains[g2, 6]])

    if degrees:
        angle, ax1, ax2, ax3 = disorientation(o1, o2, degrees=True)
    else:
        angle, ax1, ax2, ax3 = disorientation(o1, o2, degrees=False)

    return angle, ax1, ax2, ax3, i


def miso_pool(grains, pairs, degrees):
    '''

    '''
    #if __name__ == '__main__':
    start = time.time()
    result = []
    pool = mp.Pool()

    for i in range(0, len(pairs)):

        if degrees:
            pool.apply_async(calc_miso, args=(i, grains, pairs, True,), callback=result.append)
        else:
            pool.apply_async(calc_miso, args=(i, grains, pairs, False,), callback=result.append)

    pool.close()
    pool.join()
    result = np.array(result)
    result = result[result[:, 4].argsort()]

    angle = result[:, 0]
    ax = np.array([result[:, 1], result[:, 2], result[:, 3]])
    axis = np.empty((0, 3))
    for i in range(0, len(angle)):
        axis = np.append(axis, [ax[:, i]], 0)

    end = time.time()
    print(end - start)
    return angle, axis

def calc_mdf(data):
    '''
    Calculation of the MDF using Kernel Density Estimation Approach
    data: Misorientation data array
    returns: 3d grid with probabilities of a certain interval being present
    '''

    values = data.T

    kde = stats.gaussian_kde(values)

    xmin, xmax = 0, 0.42
    ymin, ymax = 0, 0.42
    zmin, zmax = 0, 0.42
    xi, yi, zi = np.mgrid[xmin:xmax:31j, ymin:ymax:31j, zmin:zmax:31j]

    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    density = kde(coords).reshape(xi.shape)
    sum = np.sum(density)
    fixed = density / sum
    return fixed


def swapping(grains):
    '''
    Random orientations swapping between two grains
    grains: grains array
    '''
    grains1 = np.copy(grains)
    lent = len(grains1) - 1
    x = r.randint(0, lent) #random selection of a grain
    y = r.randint(0, lent) #random selection of another grain

    while y == x: #safety while loop to avoid two identical grains selection
        y = r.randint(0, lent)

    array = np.array([grains1[x, 4], grains1[x, 5], grains1[x, 6]]) #selection of orientation parameters of the grain
    array1 = np.array([grains1[y, 4], grains1[y, 5], grains1[y, 6]])

    grains1[x, 4], grains1[x, 5], grains1[x, 6], grains1[y, 4], grains1[y, 5], grains1[y, 6] = array1[0], array1[1], \
                                                                                               array1[2], array[0], \
                                                                                               array[1], array[2]
    x += 1
    y += 1
    # print("ok")
    return grains1, x, y


def calc_error(mdf, mdf1):
    '''
    Error calculation
    '''
    error = np.sum((mdf - mdf1) ** 2)
    return error


def step(o, orientations, rodvecs, pairs, input_mdf):

    start_time = time.time()

    ori_opt, x, y = swapping(orientations)
    rodvecs_opt = np.copy(rodvecs)

    pairsx = np.where(pairs == x)[0]
    pairsy = np.where(pairs == y)[0]
    pairs4opt = np.append(pairsx, pairsy, 0)

    for i in pairs4opt:
        z = int(pairs[i, 0])
        k = int(pairs[i, 1])
        o1 = np.array([ori_opt[z - 1, 4], ori_opt[z - 1, 5], ori_opt[z - 1, 6]])
        o2 = np.array([ori_opt[k - 1, 4], ori_opt[k - 1, 5], ori_opt[k - 1, 6]])
        a = damask.Orientation.from_Euler_angles(phi=o1, degrees=True, family='cubic')
        b = damask.Orientation.from_Euler_angles(phi=o2, degrees=True, family='cubic')

        c = a.disorientation(b)
        rodvec = damask.Orientation.as_Rodrigues_vector(c, compact=True)
        #rodvec = rod_vec(an[1], an[0], degrees=True)
        rodvecs_opt[i] = rodvec

    opt_mdf = calc_mdf(rodvecs_opt)
    error = calc_error(input_mdf, opt_mdf)

    # print("ok")

    #print("New step:  %s seconds " % (time.time() - start_time))

    return error, x, y


def dir_swapping(grains, pairs, rodvecs, x, y):
    '''
    Accepted steps swaps
    grain: Grain Data array
    pairs: Pairs array
    rodves: Misorientation data in Rodriguez-Frank Vectors notation
    x: Grain 1 ID
    y: Grain 2 ID
    '''
    grains1 = np.copy(grains)
    array = np.array([grains1[x - 1, 4], grains1[x - 1, 5], grains1[x - 1, 6]])
    array1 = np.array([grains1[y - 1, 4], grains1[y - 1, 5], grains1[y - 1, 6]])

    grains1[x - 1, 4], grains1[x - 1, 5], grains1[x - 1, 6], grains1[y - 1, 4], grains1[y - 1, 5], grains1[
        y - 1, 6] = array1[0], array1[1], \
                    array1[2], array[0], \
                    array[1], array[2]
    rodvecs_opt = np.copy(rodvecs)

    pairsx = np.where(pairs == x)[0]
    pairsy = np.where(pairs == y)[0]
    pairs4opt = np.append(pairsx, pairsy, 0)

    for i in pairs4opt:
        z = int(pairs[i, 0])
        k = int(pairs[i, 1])
        o1 = np.array([grains1[z - 1, 4], grains1[z - 1, 5], grains1[z - 1, 6]])
        o2 = np.array([grains1[k - 1, 4], grains1[k - 1, 5], grains1[k - 1, 6]])
        a = damask.Orientation.from_Euler_angles(phi=o1, degrees=True, family='cubic')
        b = damask.Orientation.from_Euler_angles(phi=o2, degrees=True, family='cubic')

        c = a.disorientation(b)
        rodvec = damask.Orientation.as_Rodrigues_vector(c, compact=True)
        # rodvec = rod_vec(an[1], an[0], degrees=True)
        rodvecs_opt[i] = rodvec
        # print("ok")
    return grains1, rodvecs_opt


def multi_step(i, grains, rodvecs, pairs, mdf, error):
    '''
    Multi step function
    i: Iteration parameter, step no.
    grains: Grains data array
    rodvecs: Misorientation data in Rodriguez-Frank Vectors notation
    pairs: Pairs array
    mdf: EBSD data mdf
    error: input error
    '''
    start_time = time.time()
    n_cores = os.cpu_count()
    grains1 = np.copy(grains)
    rodvecs1 = np.copy(rodvecs)
    results = []
    # with mp.Pool() as pool:
    pool = mp.Pool()
    for o in range(n_cores):
        pool.apply_async(step, args=(o, grains1, rodvecs1, pairs, mdf,), callback=results.append)

        # retrieve the return value results
        # results = [ar.get() for ar in async_results]

    pool.close()
    pool.join()
    #print("New mutistep:  %s seconds " % (time.time() - start_time))
    results = np.array(results)
    eligible_swaps = (np.where(results[:, 0] < error))[0]
    if len(eligible_swaps) > 0:
        for o in eligible_swaps:
            grains1, rodvecs1 = dir_swapping(grains1, pairs, rodvecs1, int(results[o, 1]), int(results[o, 2]))

        mdf1 = calc_mdf(rodvecs1)
        error1 = calc_error(mdf, mdf1)
        #error1 = np.min(results[:, 0])
        i += n_cores
    else:
        grains1 = grains
        rodvecs1 = rodvecs
        error1 = error
        i += n_cores
    print("Step: " + str(i))
    print("Error: " + str(error1))
    print("New mutistep:  %s seconds " % (time.time() - start_time))

    return grains1, rodvecs1, error1, i

def mdf_plotting(values,in_probs,no_opt_probs,out_probs,storepath):
    plt.plot(values, in_probs)
    figname='experimental_angle_distribution.png'
    plt.title("Experimental's Angle of Misorientation Distribution")
    plt.xlabel("Misorientation Angle (degrees)")
    plt.ylabel("Probability")
    plt.savefig(storepath+figname)
    plt.close()

    plt.plot(values, no_opt_probs)
    figname = 'RVE_no_mdf_optimized_angle_distribution.png'
    plt.title("RVE's Angle of Misorientation Distribution (non-optimized)")
    plt.xlabel("Misorientation Angle (degrees)")
    plt.ylabel("Probability")
    plt.savefig(storepath + figname)
    plt.close()

    plt.plot(values, out_probs)
    figname = 'RVE_mdf_optimized_angle_distribution.png'
    plt.title("RVE's Angle of Misorientation Distribution (optimized)")
    plt.xlabel("Misorientation Angle (degrees)")
    plt.ylabel("Probability")
    plt.savefig(storepath + figname)
    plt.close()

    plt.plot(values, in_probs)
    plt.plot(values, out_probs)
    figname = 'mixed_experimental_RVE_mdf-opt_angle_distribution.png'
    plt.title("Experimental's & RVE's (optimized) Angle of Misorientation Distributions")
    plt.xlabel("Misorientation Angle (degrees)")
    plt.ylabel("Probability")
    plt.savefig(storepath + figname)
    plt.close()
