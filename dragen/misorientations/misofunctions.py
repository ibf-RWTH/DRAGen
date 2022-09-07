import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R

T = np.array([[[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]],

              [[0, 0, 1],
               [1, 0, 0],
               [0, 1, 0]],

              [[0, 1, 0],
               [0, 0, 1],
               [1, 0, 0]],

              [[0, -1, 0],
               [0, 0, 1],
               [-1, 0, 0]],

              [[0, -1, 0],
               [0, 0, -1],
               [1, 0, 0]],

              [[0, 1, 0],
               [0, 0, -1],
               [-1, 0, 0]],

              [[0, 0, -1],
               [1, 0, 0],
               [0, -1, 0]],

              [[0, 0, -1],
               [-1, 0, 0],
               [0, 1, 0]],

              [[0, 0, 1],
               [-1, 0, 0],
               [0, -1, 0]],

              [[-1, 0, 0],
               [0, 1, 0],
               [0, 0, -1]],

              [[-1, 0, 0],
               [0, -1, 0],
               [0, 0, 1]],

              [[1, 0, 0],
               [0, -1, 0],
               [0, 0, -1]],

              [[0, 0, -1],
               [0, -1, 0],
               [-1, 0, 0]],

              [[0, 0, 1],
               [0, -1, 0],
               [1, 0, 0]],

              [[0, 0, 1],
               [0, 1, 0],
               [-1, 0, 0]],

              [[0, 0, -1],
               [0, 1, 0],
               [1, 0, 0]],

              [[-1, 0, 0],
               [0, 0, -1],
               [0, -1, 0]],

              [[1, 0, 0],
               [0, 0, -1],
               [0, 1, 0]],

              [[1, 0, 0],
               [0, 0, 1],
               [0, -1, 0]],

              [[-1, 0, 0],
               [0, 0, 1],
               [0, 1, 0]],

              [[0, -1, 0],
               [-1, 0, 0],
               [0, 0, -1]],

              [[0, -1, 0],
               [1, 0, 0],
               [0, 0, 1]],

              [[0, 1, 0],
               [1, 0, 0],
               [0, 0, -1]],

              [[0, 1, 0],
               [-1, 0, 0],
               [0, 0, 1]]])

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
    npairs = np.empty((0, 2))
    for g in range(0, len(grid)):
        for i in range(0, len(grid)):
            for j in range(0, len(grid)):

                x = (g, i, j)
                # print(x)
                y = grid[x]
                # print(y)
                # print(x)
                positions = np.array([

                    [x[0] - 1, x[1] - 1, x[2] - 1],
                    [x[0] - 1, x[1], x[2] - 1],
                    [x[0] - 1, x[1] + 1, x[2] - 1],
                    [x[0] - 1, x[1] - 1, x[2]],
                    [x[0] - 1, x[1] + 1, x[2]],
                    [x[0] - 1, x[1] - 1, x[2] + 1],
                    [x[0] - 1, x[1], x[2] + 1],
                    [x[0] - 1, x[1] + 1, x[2] + 1],
                    [x[0] - 1, x[1], x[2]],

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
                    (x[0] - 1 >= 0) and (x[1] - 1 >= 0) and (x[2] - 1 >= 0),
                    (x[0] - 1 >= 0) and (x[2] - 1 >= 0),
                    (x[0] - 1 >= 0) and x[1] + 1 <= (len(grid) - 1) and x[2] - 1 >= 0,
                    (x[0] - 1 >= 0) and x[1] - 1 >= 0,
                    (x[0] - 1 >= 0) and x[1] + 1 <= (len(grid) - 1),
                    (x[0] - 1 >= 0) and x[1] - 1 >= 0 and x[2] + 1 <= (len(grid) - 1),
                    (x[0] - 1 >= 0) and x[2] + 1 <= (len(grid) - 1),
                    (x[0] - 1 >= 0) and x[1] + 1 <= (len(grid) - 1) and x[2] + 1 <= (len(grid) - 1),
                    (x[0] - 1 >= 0),

                    (x[1] - 1 >= 0) and (x[2] - 1 >= 0),
                    (x[2] - 1 >= 0),
                    x[1] + 1 <= (len(grid) - 1) and x[2] - 1 >= 0,
                    x[1] - 1 >= 0,
                    x[1] + 1 <= (len(grid) - 1),
                    x[1] - 1 >= 0 and x[2] + 1 <= (len(grid) - 1),
                    x[2] + 1 <= (len(grid) - 1),
                    x[1] + 1 <= (len(grid) - 1) and x[2] + 1 <= (len(grid) - 1),

                    (x[0] + 1 <= len(grid) - 1) and (x[1] - 1 >= 0) and (x[2] - 1 >= 0),
                    (x[0] + 1 <= len(grid) - 1) and (x[2] - 1 >= 0),
                    (x[0] + 1 <= len(grid) - 1) and x[1] + 1 <= (len(grid) - 1) and x[2] - 1 >= 0,
                    (x[0] + 1 <= len(grid) - 1) and x[1] - 1 >= 0,
                    (x[0] + 1 <= len(grid) - 1) and x[1] + 1 <= (len(grid) - 1),
                    (x[0] + 1 <= len(grid) - 1) and x[1] - 1 >= 0 and x[2] + 1 <= (len(grid) - 1),
                    (x[0] + 1 <= len(grid) - 1) and x[2] + 1 <= (len(grid) - 1),
                    (x[0] + 1 <= len(grid) - 1) and x[1] + 1 <= (len(grid) - 1) and x[2] + 1 <= (len(grid) - 1),
                    (x[0] + 1 <= len(grid) - 1)

                ])

                if y > 0:
                    for k in range(0, 8):
                        if arguments[k] == True:
                            p = grid[positions[k, 0], positions[k, 1],positions[k,2]]
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
                print(len(pairs))
            else:
                cnpairs = np.delete(cnpairs, 0, 0)
                print(len(pairs))
    return pairs


def orientation_matrix(x, degrees):
    if degrees == True:
        x = (np.radians(x[0]), np.radians(x[1]), np.radians(x[2]))
    else:
        pass

    cf1 = np.cos(x[0])

    sf1 = np.sin(x[0])

    cF = np.cos(x[1])

    sF = np.sin(x[1])

    cf2 = np.cos(x[2])

    sf2 = np.sin(x[2])

    ori_matrix = np.array([[cf1 * cf2 - sf1 * sf2 * cF, sf1 * cf2 + cf1 * sf2 * cF, sf2 * sF],
                           [-cf1 * sf2 - sf1 * cf2 * cF, -sf1 * sf2 + cf1 * cf2 * cF, cf2 * sF],
                           [sf1 * sF, -cf1 * sF, cF]])
    return ori_matrix

def misorientation(x, y, degrees, grains):
    '''
    function that calculates the misorientation angle between two grains
    Input:
        x=list, [phi1,PHI,ph2]
        y=list, [phi1,PHI,ph2]

    Output:
        misorientation angle in degrees
    '''
    x = (grains[x-1, 4], grains[x-1, 5], grains[x-1, 6])
    y = (grains[y-1, 4], grains[y-1, 5], grains[y-1, 6])

    print(x)
    print(y)

    if degrees == True:
        rot = (orientation_matrix(x, degrees=True))
        rot1 = (orientation_matrix(y, degrees=True))
    else:
        rot = (orientation_matrix(x, degrees=False))
        rot1 = (orientation_matrix(y, degrees=False))
    # print(rot)
    # print(rot1)
    Mis = np.matmul(rot, inv(rot1))
    # print(Mis)
    M = np.empty((0, 3, 3))
    a = np.empty(0)

    for i in range(0, 24):
        Mi = np.matmul(T[i], Mis)
        M = np.append(M, [Mi], axis=0)
        y = ((Mi.trace() - 1) / 2)
        # print(y)
        ai = np.arccos(y)
        ai = np.degrees(ai)
        a = np.append(a, [ai], axis=0)

    angle = np.min(a)
    h = np.where(angle == a)
    matrix = M[h]
    #print(matrix)
    # print(matrix)

    denominator = np.sqrt((matrix[0, 1, 2] - matrix[0, 2, 1]) ** 2 + (matrix[0, 2, 0] - matrix[0, 0, 2]) ** 2 + (
                matrix[0, 0, 1] - matrix[0, 1, 0]) ** 2)
    # denominator=2*np.sin(angle)
    r1 = (matrix[0, 1, 2] - matrix[0, 2, 1]) / denominator

    r2 = (matrix[0, 2, 0] - matrix[0, 0, 2]) / denominator
    r3 = (matrix[0, 0, 1] - matrix[0, 1, 0]) / denominator

    axis = (r1, r2, r3)
    return angle, axis
