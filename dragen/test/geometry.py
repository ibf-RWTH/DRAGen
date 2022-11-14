from dragen.utilities.InputInfo import RveInfo
import pandas as pd
import numpy as np


class Cylinder:
    """
    a cylinder, the center point of 2 end sections are p1, p2 radius is r
    """

    def __init__(self, p1: pd.DataFrame, p2: pd.DataFrame, r: float):
        self.p1 = p1
        self.p2 = p2
        self.r = r

    def is_inside(self, p: pd.DataFrame) -> bool:
        """
        determine whether a point p is inside the cylinder
        """
        pass

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
    points = pd.DataFrame(columns=['x', 'y', 'z'])
    points['x'] = points['y'] = points['z'] = [1, 2, 3]
    p1 = points.iloc[0][['x', 'y', 'z']]
    p2 = points.iloc[1][['x', 'y', 'z']]
    p3 = points.iloc[2][['x', 'y', 'z']]
    d = [0.1] * 3
    points['d'] = d
    n = np.random.uniform(0, 1, (1, 3))
    # pedal_points = get_pedal_point(p1, n, points['d'])
    # print(pedal_points)
    # print(dis_in_rve(p1, p2, False, False, False))
    #
    # RveInfo.box_size = RveInfo.box_size_y = RveInfo.box_size_z = 5
    # print(dis_in_rve(p2, p3, True, True, True))
