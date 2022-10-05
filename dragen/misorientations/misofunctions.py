import numpy as np
import damask

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

def calc_miso(grains,pairs,degrees):
    quaternions=np.empty((0,4))
    angle=np.empty((0,1))

    for i in range(0, len(pairs)):
        x = int(pairs[i, 0])
        y = int(pairs[i, 1])
        o1 = np.array([grains[x - 1, 4], grains[x - 1, 5], grains[x - 1, 6]])
        o2 = np.array([grains[y - 1, 4], grains[y - 1, 5], grains[y - 1, 6]])

        if degrees==True:
            a = damask.Orientation.from_Euler_angles(phi=o1, degrees=True, family='cubic')
            b = damask.Orientation.from_Euler_angles(phi=o2, degrees=True, family='cubic')
        else:
            a = damask.Orientation.from_Euler_angles(phi=o1, degrees=False, family='cubic')
            b = damask.Orientation.from_Euler_angles(phi=o2, degrees=False, family='cubic')


        c = a.disorientation(b)
        a_a=damask.Orientation.as_axis_angle(c,degrees=True,pair=True)
        a=a_a[1]
        quaternions = np.append(quaternions, [c], 0)
        angle=np.append(angle,[[a]],0)
        cond=((len(quaternions) / len(pairs)) * 100)
        print("Completion: "+str(cond))

    return quaternions,angle

