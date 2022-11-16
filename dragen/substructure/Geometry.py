from dragen.utilities.InputInfo import RveInfo
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


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
    x_max = RveInfo.box_size
    y_max = RveInfo.box_size_y if RveInfo.box_size_y is not None else RveInfo.box_size
    z_max = RveInfo.box_size_y if RveInfo.box_size_z is not None else RveInfo.box_size
    if same_side:
        x_dis = p1['x'] - p2['x']
        y_dis = p1['y'] - p2['y']
        z_dis = p1['z'] - p2['z']
    else:
        x_dis = p1['x'] - p2['x'] if not x_moved else x_max - abs(p1['x'] - p2['x'])
        y_dis = p1['y'] - p2['y'] if not y_moved else y_max - abs(p1['y'] - p2['y'])
        z_dis = p1['z'] - p2['z'] if not z_moved else z_max - abs(p1['z'] - p2['z'])
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


def compute_num_clusters(packet: np.ndarray) -> int:
    """
    return the number of clusters of a packet
    :param packet: packet
    :return: num of clusters within the packet
    """
    x_max = RveInfo.box_size
    y_max = RveInfo.box_size_y if RveInfo.box_size_y is not None else RveInfo.box_size
    z_max = RveInfo.box_size_y if RveInfo.box_size_z is not None else RveInfo.box_size
    i = 0
    if np.isclose(packet[:, 0], 0, 1e-5).any() and np.isclose(packet[:, 0], x_max, 1e-5).any():
        i += 1
    if np.isclose(packet[:, 1], 0, 1e-5).any() and np.isclose(packet[:, 1], y_max, 1e-5).any():
        i += 1
    if np.isclose(packet[:, 2], 0, 1e-5).any() and np.isclose(packet[:, 2], z_max, 1e-5).any():
        i += 1
    return 2 ** i


def train_kmeans(num_clusters: int, packet: np.ndarray, random_state: int = 0) -> object:
    """
    train a k_means cluster within a packet
    :param num_clusters: number of clusters within the packet
    :param packet: packet
    :param random_state: default 0
    :return: KMeans object
    """
    return KMeans(n_clusters=num_clusters, random_state=random_state).fit(packet)


def issame_side(kmeans: object, p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    determine if 2 points on block boundaries are in the same cluster of a packet
    :param kmeans: kmeans cluster
    :param p1: point on 1st block boundaries
    :param p2: point on 2nd block boundaries
    :return: same cluster True else False
    """
    pred1 = kmeans.predict(p1.reshape(1, -1))[0]
    pred2 = kmeans.predict(p2.reshape(1, -1))[0]
    return pred1 == pred2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rve = pd.read_csv(r"/home/doelz-admin/DRAGen/dragen/test/results/substruct_data.csv")
    ax = plt.figure().add_subplot(projection='3d')
    p1 = pd.DataFrame(columns=['x', 'y', 'z'], data=np.array([[0, 0, 0]]))
    p2 = pd.DataFrame(columns=['x', 'y', 'z'], data=np.array([[0.5, 0, 0]]))
    p3 = pd.DataFrame(columns=['x', 'y', 'z'], data=np.array([[2.5, 0, 0]]))

    packet = rve.loc[rve['packet_id'] == 4, ['x', 'y', 'z']].to_numpy()
    # print(packet)
    ax.scatter(packet[:, 0], packet[:, 1], packet[:, 2])
    # plt.show()
    RveInfo.box_size = RveInfo.box_size_y = RveInfo.box_size_z = 30
    n = compute_num_clusters(packet=packet)
    print(n)
    block_norm = np.array([1, 0, 0]).reshape(1, 3)
    d_values = -(packet[:, 0] * block_norm[0, 0] +
                 packet[:, 1] * block_norm[0, 1] +
                 packet[:, 2] * block_norm[0, 2])
    print(d_values)
    p1 = packet[packet[:, 0] * block_norm[0, 0] +
                packet[:, 1] * block_norm[0, 1] +
                packet[:, 2] * block_norm[0, 2] + d_values[0] == 0][0, :]
    print(p1)
    p2 = packet[packet[:, 0] * block_norm[0, 0] +
                packet[:, 1] * block_norm[0, 1] +
                packet[:, 2] * block_norm[0, 2] + d_values[120] == 0][0, :]
    print(p2)
    kmeans = train_kmeans(num_clusters=n, packet=packet)
    issame_side(kmeans=kmeans, p1=p1, p2=p2)

    clusters = sorted(kmeans.labels_)
    print(clusters)
