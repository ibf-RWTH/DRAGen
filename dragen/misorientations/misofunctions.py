import numpy as np
import damask
import time

def pairs2d(grid):
    npairs = np.empty((0, 2))
    for i in range(0, len(grid)):
        for j in range(0, len(grid)):

            x = (i, j)
            # print(x)
            y = grid[x]
            # print(y)
            # print(x)
            positions = np.array([[x[0] - 1, x[1] - 1],
                                  [x[0], x[1] - 1],
                                  [x[0] + 1, x[1] - 1],
                                  [x[0] - 1, x[1]],
                                  [x[0] + 1, x[1]],
                                  [x[0] - 1, x[1] + 1],
                                  [x[0], x[1] + 1],
                                  [x[0] + 1, x[1] + 1]])

            arguments = np.array([(x[0] - 1 >= 0) and (x[1] - 1 >= 0),
                                  (x[1] - 1 >= 0),
                                  x[0] + 1 <= (len(grid) - 1) and x[1] - 1 >= 0,
                                  x[0] - 1 >= 0,
                                  x[0] + 1 <= (len(grid) - 1),
                                  x[0] - 1 >= 0 and x[1] + 1 <= (len(grid) - 1),
                                  x[1] + 1 <= (len(grid) - 1),
                                  x[0] + 1 <= (len(grid) - 1) and x[1] + 1 <= (len(grid) - 1)])

            if y > 0:
                for k in range(0, 8):
                    if arguments[k] == True:
                        p = grid[positions[k, 0], positions[k, 1]]
                        # print(p)
                        if y != p:
                            pair = np.array((y, p))
                            npairs = np.append(npairs, [pair], 0)
                        else:
                            pass

    npairs = np.unique(npairs, axis=0)
    cnpairs = npairs.copy()
    pairs = np.empty((0, 2))

    for i in npairs:
        bool = np.empty([0])
        for u in cnpairs:
            condition = i[0] == u[1] and i[1] == u[0]
            bool = np.append(bool, [condition], 0)
            # print("break")
        if 1 in bool:
            pairs = np.append(pairs, [i], 0)
            cnpairs = np.delete(cnpairs, 0, 0)
            # print("break")
        else:
            cnpairs = np.delete(cnpairs, 0, 0)
    return pairs


def pairs3d(grid):
    start_time = time.time()
    print("Creating RVE's neighbour grains array...")
    npairs = np.empty((0, 2))

    if grid.ndim == 3:
        for g in range(0, len(grid)):
            for i in range(0, len(grid)):
                for j in range(0, len(grid)):

                    x = (g, i, j)
                    # print(x)

                    # print(x)
                    y = grid[x]
                    # print(y)
                    # print(x)
                    positions = np.array([

                        [x[0], x[1] - 1, x[2] - 1],
                        [x[0], x[1], x[2] - 1],
                        [x[0], x[1] + 1, x[2] - 1],
                        [x[0], x[1] - 1, x[2]],
                        [x[0], x[1] + 1, x[2]],
                        [x[0], x[1] - 1, x[2] + 1],
                        [x[0], x[1], x[2] + 1],
                        [x[0], x[1] + 1, x[2] + 1],

                        [x[0] + 1, x[1], x[2]],
                        [x[0] + 1, x[1] - 1, x[2] - 1],
                        [x[0] + 1, x[1], x[2] - 1],
                        [x[0] + 1, x[1] + 1, x[2] - 1],
                        [x[0] + 1, x[1] - 1, x[2]],
                        [x[0] + 1, x[1] + 1, x[2]],
                        [x[0] + 1, x[1] - 1, x[2] + 1],
                        [x[0] + 1, x[1], x[2] + 1],
                        [x[0] + 1, x[1] + 1, x[2] + 1]])

                    arguments = np.array([

                        (x[1] - 1 >= 0) and (x[2] - 1 >= 0),
                        (x[2] - 1 >= 0),
                        x[1] + 1 <= (len(grid) - 1) and x[2] - 1 >= 0,
                        x[1] - 1 >= 0,
                        x[1] + 1 <= (len(grid) - 1),
                        x[1] - 1 >= 0 and x[2] + 1 <= (len(grid) - 1),
                        x[2] + 1 <= (len(grid) - 1),
                        x[1] + 1 <= (len(grid) - 1) and x[2] + 1 <= (len(grid) - 1),

                        (x[0] + 1 <= len(grid) - 1),
                        (x[0] + 1 <= len(grid) - 1) and (x[1] - 1 >= 0) and (x[2] - 1 >= 0),
                        (x[0] + 1 <= len(grid) - 1) and (x[2] - 1 >= 0),
                        (x[0] + 1 <= len(grid) - 1) and x[1] + 1 <= (len(grid) - 1) and x[2] - 1 >= 0,
                        (x[0] + 1 <= len(grid) - 1) and x[1] - 1 >= 0,
                        (x[0] + 1 <= len(grid) - 1) and x[1] + 1 <= (len(grid) - 1),
                        (x[0] + 1 <= len(grid) - 1) and x[1] - 1 >= 0 and x[2] + 1 <= (len(grid) - 1),
                        (x[0] + 1 <= len(grid) - 1) and x[2] + 1 <= (len(grid) - 1),
                        (x[0] + 1 <= len(grid) - 1) and x[1] + 1 <= (len(grid) - 1) and x[2] + 1 <= (len(grid) - 1)])


                    for k in range(0, 17):
                        if arguments[k] == True:
                            p = grid[positions[k, 0], positions[k, 1], positions[k, 2]]
                            # print(p)
                            if y != p:
                                pair = np.array((y, p))
                                npairs = np.append(npairs, [pair], 0)

    npairs = np.sort(npairs)
    npairs = np.unique(npairs, axis=0)
    print("Finished! Total length: " + str(len(npairs)))
    end_time = time.time() - start_time
    if end_time>=60:
        print(str(end_time/60)+" minutes")
    else:
        print(str(end_time) + " seconds")
    return npairs

def calc_miso(grains,pairs,degrees):
    '''
    Function that calculates the mdf for a set of input data
    :param data: Array which contains grains' information
    :param pairs: Array of adjacent grains
    :param degrees: Input Euler Angles in degrees or not
    :return: Array of misorientation information per pair
    '''
    start_time = time.time()
    #grainid = np.array([data['GrainID']]).reshape((-1, 1))
    grainid=grains[:,7]

    angle = np.empty((0, 1))
    axis = np.empty((0, 3))

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
        a_a = damask.Orientation.as_axis_angle(c, degrees=True, pair=True)
        # e_a=damask.Orientation.as_Euler_angles(c,degrees=True)
        a = a_a[1]
        a1 = a_a[0]
        # quaternions = np.append(quaternions, [c], 0)
        angle = np.append(angle, [[a]], 0)
        axis = np.append(axis, [a1], 0)
        # euler_angles=np.append(euler_angles,[e_a],0)
        # cond=((len(quaternions) / len(pairs)) * 100)
        # print("Completion: "+str(cond))
    print(" %s seconds " % (time.time() - start_time))

    return angle,axis


