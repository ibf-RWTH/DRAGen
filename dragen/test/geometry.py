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


def one_side(p1: pd.DataFrame, p2: pd.DataFrame, rve: pd.DataFrame):
    pass


def dis_in_rve(p1: pd.DataFrame, p2: pd.DataFrame, x_moved: bool, y_moved: bool, z_moved: bool):
    x_dis = p1['x'] - p2['x'] if not x_moved else RveInfo.box_size - abs(p1['x'] - p2['x'])
    y_dis = p1['y'] - p2['y'] if not y_moved else RveInfo.box_size_y - abs(p1['y'] - p2['y'])
    z_dis = p1['z'] - p2['z'] if not z_moved else RveInfo.box_size_z - abs(p1['z'] - p2['z'])
    return np.linalg.norm([x_dis, y_dis, z_dis])


def get_pedal_point(p1: pd.DataFrame, n: np.ndarray, d: [pd.DataFrame]) -> pd.DataFrame:
    """
    get the pedal point of p1 on the plane with norm n and intercept d
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

    x = y = z = np.linspace(0, 3, num=100)
    grids = xs, ys, zs = np.meshgrid(x, y, z)
    ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter(xs, ys, zs)
    # plt.show()
    points = np.vstack(map(np.ravel, grids)).reshape(-1, 3)
    rve = pd.DataFrame(columns=['x', 'y', 'z'], data=points)
    p1 = pd.DataFrame(columns=['x', 'y', 'z'], data=np.array([[0, 0, 0]]))
    p2 = pd.DataFrame(columns=['x', 'y', 'z'], data=np.array([[1, 1, 1]]))

    cylinder = Cylinder(p1=p1.to_numpy(), p2=p2.to_numpy(), r=1.0)
    inside = rve.apply(lambda p: cylinder.is_inside(p.to_numpy()), axis=1)

    cl = rve[inside]
    print(len(cl))
    ax.scatter(cl['x'], cl['y'], cl['z'])
    plt.show()
