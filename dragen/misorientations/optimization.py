import numpy as np
import pandas as pd
import damask
import random as r
import time
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from dragen.misorientations.misofunctions import calc_miso

def values():
    '''
    Function which discretisize the possible results
    '''

    values = np.linspace(0, 65, 70)[:, np.newaxis]
    return values

def mdf_score_samples(angle,values):
    '''
    :param angle: Array of misorientation parameters
    :return: Array of probabilities
    '''
    mdf=KernelDensity(kernel='gaussian', bandwidth=0.6).fit(angle)
    probs=np.exp(mdf.score_samples(values))
    return probs

def swapping(grains):
    '''
    Orientation swapping between two grains
    :param grains: Array with grains' information
    :return: Array with grains' information after swapping
    '''
    grains1=np.copy(grains)
    lent = len(grains1) - 1
    x = r.randint(0, lent)
    y = r.randint(0, lent)
    while y == x:
        y = r.randint(0, lent)

    array = np.array([grains1[x, 4], grains1[x, 5], grains1[x, 6]])
    array1 = np.array([grains1[y, 4], grains1[y, 5], grains1[y, 6]])

    grains1[x, 4], grains1[x, 5], grains1[x, 6], grains1[y, 4], grains1[y, 5], grains1[y, 6] = array1[0], array1[1],array1[2], array[0], array[1], array[2]
    x+=1
    y+=1
    #print("ok")
    return grains1,x,y

def calc_error(input,output,bins):
    '''
    Error calculation between set of misorientation data
    :param input: Array of input mdf
    :param output: Array of optimized/output misorientation data
    :param bins: Values
    :return: Error
    '''
    error = 0
    for i in range(0, len(bins)):
        x = output[i]
        y=input[i]
        error += (x - y) ** 2
    return error

def step(grains1,angle1,pairs1,input_probs,values):
    '''
    Monte Carlo single step: Grains' orinetation swap and subsequent error calculation
    :param grains1: Array with grains' information
    :param angle1:  Array of misorientation information per pair
    :param pairs1:  Array of adjacent grains
    :param input_probs: Probabilities from the input set of data
    :return: Array with grains' information after swapping, Array of misorientation information per pair after swapping, error, MDF of output data
    '''
    #start_time = time.time()

    grains_opt,x,y=swapping(grains1)
    angle_opt=np.copy(angle1)

    pairsx=np.where(pairs1==x)[0]
    pairsy=np.where(pairs1==y)[0]
    pairs4opt=np.append(pairsx,pairsy,0)

    for i in pairs4opt:
        z = int(pairs1[i, 0])
        k = int(pairs1[i, 1])
        o1 = np.array([grains_opt[z - 1, 4], grains_opt[z - 1, 5], grains_opt[z - 1, 6]])
        o2 = np.array([grains_opt[k - 1, 4], grains_opt[k - 1, 5], grains_opt[k - 1, 6]])
        a = damask.Orientation.from_Euler_angles(phi=o1, degrees=True, family='cubic')
        b = damask.Orientation.from_Euler_angles(phi=o2, degrees=True, family='cubic')

        c = a.disorientation(b)
        an=damask.Orientation.as_axis_angle(c,degrees=True,pair=True)[1]
        angle_opt[i]=an
        #print("ok")

    opt_probs=mdf_score_samples(angle_opt,values)
    error1=calc_error(input_probs,opt_probs,values)
    #print("New step:  %s seconds " % (time.time() - start_time))
    #print("ok")
    #print(error)
    #print(error1)
    return grains_opt,angle_opt,error1

def mdf_opt(grains1, angle1,pairs1,error,input_probs,values):
    i = 0
    #swaps=np.empty((0,2))
    while i < 3000:
        grains_opt, angle_opt, error2 = step(grains1, angle1, pairs1,input_probs,values)
        i += 1
        print("Step: " + str(i))
        while error2 > error:
            grains_opt, angle_opt, error2 = step(grains1, angle1, pairs1,input_probs,values)
            i += 1
            print("Step: " + str(i))
        else:
            grains1, angle1, error = grains_opt, angle_opt, error2
            #swap = np.array([x, y])
            #swaps = np.append(swaps, [swap], 0)
            print(error)
    else:
        if error2>error:
            grains1, angle1, error = grains_opt, angle_opt, error2

    return grains1, angle1

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
