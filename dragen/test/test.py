from dragen.utilities.InputInfo import RveInfo
import pandas as pd
import numpy as np
from dragen.substructure.Geometry import one_side, dis_in_rve


class SubsTester:
    def __init__(self):
        RveInfo.box_size = RveInfo.box_size_y = RveInfo.box_size_z = 3
        x = y = z = np.linspace(0, 3, num=30, endpoint=False)
        grids = xs, ys, zs = np.meshgrid(x, y, z)
        points = np.vstack(map(np.ravel, grids)).T
        self.rve = pd.DataFrame(columns=['x', 'y', 'z'], data=points)
        self.p1 = pd.DataFrame(columns=['x', 'y', 'z'], data=np.array([[0, 0, 0]]))
        self.p2 = pd.DataFrame(columns=['x', 'y', 'z'], data=np.array([[0.5, 0, 0]]))
        self.p3 = pd.DataFrame(columns=['x', 'y', 'z'], data=np.array([[2.5, 0, 0]]))
        self.rve['packet_id'] = 0
        self.rve.loc[self.rve['x'] <= 1.0, 'packet_id'] = 1
        self.rve.loc[(self.rve['x'] <= 2.0) & (self.rve['x'] > 1), 'packet_id'] = 2
        self.rve.loc[(self.rve['x'] <= 3.0) & (self.rve['x'] > 2), 'packet_id'] = 1

    def test_one_side(self):
        n = np.array([1, 0, 0]).reshape(1, 3)
        assert one_side(n, self.p1, self.p2, self.rve, packet_id=1)
        assert not one_side(n, self.p1, self.p3, self.rve, packet_id=1)

    def test_dis_in_rve(self):
        dis1 = dis_in_rve(same_side=False, p1=self.p1, p2=self.p2, x_moved=True, y_moved=False, z_moved=False)
        dis2 = dis_in_rve(same_side=False, p1=self.p2, p2=self.p3, x_moved=True, y_moved=False, z_moved=False)
        assert dis1 == 0.5 and dis2 == 1.0

if __name__ == "__main__":
    subs_tester = SubsTester()
    subs_tester.test_one_side()