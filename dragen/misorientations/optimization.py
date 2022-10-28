import numpy as np
import random as r
from dragen.misorientations import misofunctions as f

def prob(a,lower,upper):
    newarray=np.empty([0,1])
    for i in a:
        if lower==0:
            condition=(i>=lower) & (i<=upper)
        else:
            condition=(i>lower) & (i<=upper)
        for y in condition:
            if y:
                newarray=np.append(newarray,[[y]],0)
        prob=len(newarray)/len(a)

    return prob

def discretisize():
    discrete=np.empty((0,2))
    for i in range(0,63):
        array=np.array([i,i+1])
        discrete=np.append(discrete,[array],0)
    return discrete

def calc_probs(array,bins):
    proba = np.empty((0, 1))

    for i in bins:
        p = prob(array, i[0], i[1])
        proba = np.append(proba, [[p]], 0)
    return proba

def calc_error(input,output,bins):
    error = 0
    for i in range(0, len(bins)):
        error = error + (output[i] - input[i]) ** 2
    return error

def swapping(grains):
    lent = len(grains) - 1
    x = r.randint(0, lent)
    y = r.randint(0, lent)
    while y == x:
        y = r.randint(0, lent)
    else:
        pass

    array = np.array([grains[x, 4], grains[x, 5], grains[x, 6]])
    array1 = np.array([grains[y, 4], grains[y, 5], grains[y, 6]])

    grains[x, 4], grains[x, 5], grains[x, 6], grains[y, 4], grains[y, 5], grains[y, 6] = array1[0], array1[1],array1[2], array[0], array[1], array[2]

    return grains

def local_opt(input,grains,pairs,bins,error):
    grains1 = swapping(grains)

    '''
    New error calculation
    '''
    miso2 = f.calc_miso(grains=grains1, pairs=pairs, degrees=True)
    angle2 = miso2[1]
    opt = calc_probs(angle2, bins)
    error1 = calc_error(input=input, output=opt, bins=bins)

    #print(error)
    #print(error1)

    while error1 >= error:
        grains1 = swapping(grains)
        miso2 = f.calc_miso(grains=grains1, pairs=pairs, degrees=True)
        angle2 = miso2[1]
        opt = calc_probs(angle2, bins)
        error1 = calc_error(input=input, output=opt, bins=bins)


    else:
        #print(error)
        #print(error1)
        error=error1

    return grains1,error

def opt(input,grains_noopt,pairs,bins,error_initial):
    grains2,error=local_opt(input,grains_noopt,pairs,bins,error_initial)
    print(error)

    '''
    Optimizing
    '''
    print('Optimizing......')
    print(error<=0.005)
    while error>=0.005:
        grains2, error = local_opt(input, grains2, pairs, bins, error)
        print(error)


    else:
        print("Final error: "+str(error))

    return grains2
