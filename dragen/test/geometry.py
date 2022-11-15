from dragen.utilities.InputInfo import RveInfo
import pandas as pd
import numpy as np


class Cylinder:
    """
    a cylinder, the center point of 2 end sections are p1, p2 radius is r
    """

    def __init__(self, p1: np.ndarray, p2: np.ndarray, r: float):
        self.p1 = p1
        self.p2 = p2
        self.r = r

    def is_inside(self, p: np.ndarray) -> bool:
        """
        determine whether a point p is inside the cylinder
        """
        v1 = self.p2 - self.p1
        v2 = self.p2 - p
        v3 = p - self.p1

        if np.sum(v1 * v2) * np.sum(v1 * v3) >= 0:
            v3_project = np.sum(v1 * v3) / np.sum(v1 * v1) * v1
            dis = np.linalg.norm(v3 - v3_project)
            return dis <= self.r
        else:
            return False


def one_side(n: np.ndarray, p1: pd.DataFrame, p2: pd.DataFrame, rve: pd.DataFrame, packet_id:[int,float]) -> bool:
    """
    determine if the planes with normal n and pass p1, p2 belong to the same side in rve
    :param packet_id: the id of packet which have p1, p2
    :param n: norm of the planes
    :param p1: point on plane 1
    :param p2: point on plane 2
    :param rve: rve 
    :return: same side True otherwise False
    """
    d1 = -(p1['x'] * n[0, 0] + p1['y'] * n[0, 1] + p1['z'] *
           n[0, 2])
    d2 = -(p2['x'] * n[0, 0] + p2['y'] * n[0, 1] + p2['z'] *
           n[0, 2])
    section = rve.loc[(rve['x'] * n[0, 0] +
                       rve['y'] * n[0, 1] +
                       rve['z'] * n[0, 2] + float(d1)) *
                      (rve['x'] * n[0, 0] +
                       rve['y'] * n[0, 1] +
                       rve['z'] * n[0, 2] + float(d2)) <= 0].copy()
    pds = -(section['x'] * n[0, 0] +
            section['y'] * n[0, 1] +
            section['z'] * n[0, 2])
    section['pd'] = pds
    group = section.groupby('pd').apply(lambda data: len(data[data['packet_id'] == packet_id]))
    pid_counts = pd.DataFrame(columns=['pid_num'], data=group)

    return len(pid_counts[pid_counts['pid_num'] == 0]) == 0


def dis_in_rve(same_side: bool,
               p1: pd.DataFrame,
               p2: pd.DataFrame,
               x_moved: bool, y_moved: bool, z_moved: bool) -> float:
    """
    compute the distance between 2 points in rve
    :param same_side: if 2 points are on one sane side
    :param p1: point1
    :param p2: point2
    :param x_moved: if the rve is clipped along x-axis
    :param y_moved: if the rve is clipped along y-axis
    :param z_moved: if the rve is clipped along z-axis
    :return:
    """
    if same_side:
        x_dis = p1['x'] - p2['x']
        y_dis = p1['y'] - p2['y']
        z_dis = p1['z'] - p2['z']
    else:
        x_dis = p1['x'] - p2['x'] if not x_moved else RveInfo.box_size - abs(p1['x'] - p2['x'])
        y_dis = p1['y'] - p2['y'] if not y_moved else RveInfo.box_size_y - abs(p1['y'] - p2['y'])
        z_dis = p1['z'] - p2['z'] if not z_moved else RveInfo.box_size_z - abs(p1['z'] - p2['z'])
    return np.linalg.norm([x_dis, y_dis, z_dis])


def get_pedal_point(p1: pd.DataFrame, n: np.ndarray, d: [pd.DataFrame]) -> pd.DataFrame:
    """
    get the pedal point of p1 on the plane with norm n and intercept d
    :param p1: point
    :param n: norm of the plane
    :param d: intercept of the plane
    :return: pedal points of p1 on plane
    """
    unit_v = n / np.linalg.norm(n)  # get the unit vector
    dis = abs(p1['x'] * n[0, 0] +
              p1['y'] * n[0, 1] +
              p1['z'] * n[0, 2] + d) / np.linalg.norm(n)  # the distance of p1 to the plane
    # p2 - p1 = unit_v * dis
    pedal_points = pd.DataFrame(columns=['x', 'y', 'z'])
    pedal_points['x'] = p1['x'] + unit_v[0, 0] * dis
    pedal_points['y'] = p1['y'] + unit_v[0, 1] * dis
    pedal_points['z'] = p1['z'] + unit_v[0, 2] * dis
    return pedal_points


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = y = z = np.linspace(0, 3, num=30, endpoint=False)
    grids = xs, ys, zs = np.meshgrid(x, y, z)
    ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter(xs, ys, zs)
    # plt.show()
    points = np.vstack(map(np.ravel, grids)).T
    rve = pd.DataFrame(columns=['x', 'y', 'z'], data=points)
    print(rve)
    p1 = pd.DataFrame(columns=['x', 'y', 'z'], data=np.array([[0, 0, 0]]))
    p2 = pd.DataFrame(columns=['x', 'y', 'z'], data=np.array([[0.5, 0, 0]]))
    p3 = pd.DataFrame(columns=['x', 'y', 'z'], data=np.array([[2.5, 0, 0]]))
    rve['packet_id'] = 0
    rve.loc[rve['x'] <= 1.0, 'packet_id'] = 1
    rve.loc[(rve['x'] <= 2.0) & (rve['x'] > 1), 'packet_id'] = 2
    rve.loc[(rve['x'] <= 3.0) & (rve['x'] > 2), 'packet_id'] = 1
    n = np.array([1, 0, 0]).reshape(1, 3)
    d1 = -(p1['x'] * n[0, 0] + p1['y'] * n[0, 1] + p1['z'] *
           n[0, 2])
    d2 = -(p2['x'] * n[0, 0] + p2['y'] * n[0, 1] + p2['z'] *
           n[0, 2])
    section = rve.loc[(rve['x'] * n[0, 0] +
                       rve['y'] * n[0, 1] +
                       rve['z'] * n[0, 2] + float(d1)) *
                      (rve['x'] * n[0, 0] +
                       rve['y'] * n[0, 1] +
                       rve['z'] * n[0, 2] + float(d2)) <= 0].copy()

    pds = -(section['x'] * n[0, 0] +
           section['y'] * n[0, 1] +
           section['z'] * n[0, 2])

    section['pd'] = pds
    group = section.groupby('pd').apply(lambda data: len(data[data['packet_id'] == 1]))
    pid_counts = pd.DataFrame(columns=['pid_num'], data=group)
    print(len(pid_counts[pid_counts['pid_num'] == 0]))

    # cylinder = Cylinder(p1=p1.to_numpy(), p2=p2.to_numpy(), r=1.0)
    # inside = rve.apply(lambda p: cylinder.is_inside(p.to_numpy()), axis=1)
    #
    # cl = rve[inside]
    # print(len(cl))
    # packet = rve[rve['packet_id'] == 1]
    # print(d1, d2)
    # print(rve['pd'].unique())
    # ax.scatter(rve['x'], rve['y'], rve['z'])
