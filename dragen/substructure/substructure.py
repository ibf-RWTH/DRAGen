# _*_ coding: utf-8 _*_
"""
Time:     2021/8/19 15:47
Author:   Linghao Kong
Version:  V 0.1
File:     new_substructure
Describe: Write during the internship at IEHK RWTH"""
from dragen.stats.preprocessing import *
from dragen.substructure.DataParser import subs_sampler, SubsDistribution
from typing import Tuple
from dragen.substructure.Geometry import dis_in_rve, get_pedal_point, issame_side, compute_num_clusters, train_kmeans
from dragen.substructure.Crystallograhy import CrystallInfo


def plot_rve_subs(rve_data, subs_name, store_path=None):
    if subs_name == 'Grain' or 'phase':
        ID = '%sID' % subs_name

    if subs_name == 'packet' or 'block':
        ID = '%s_id' % subs_name

    subs_list = list(set(rve_data[ID]))
    fig = plt.figure(dpi=200)
    ax = fig.gca(projection='3d')

    for subs in subs_list:
        points_data = rve_data[rve_data[ID] == subs]
        ax.plot(points_data['x'], points_data['y'], points_data['z'], 'x', markersize=2)

    if store_path:
        plt.savefig(store_path + '/%s.png' % subs_name)

    plt.show()


def choose_block_boundary_norm(packet: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    bad_bt_flag = False  # if the packet are subdivided into more than 4 parts the computation block thickness is wrong
    block_boundary_norm = np.random.uniform(0, 1, (1, 3))
    x_max = RveInfo.box_size
    y_max = RveInfo.box_size_y if RveInfo.box_size_y is not None else RveInfo.box_size
    z_max = RveInfo.box_size_y if RveInfo.box_size_z is not None else RveInfo.box_size
    if 0.0 in packet['x'].values and x_max in packet['x'].values:
        block_boundary_norm[0, 0] = 0.0

    if 0.0 in packet['y'].values and y_max in packet['y'].values:
        block_boundary_norm[0, 0] = 0.0

    if 0.0 in packet['z'].values and z_max in packet['z'].values:
        block_boundary_norm[0, 0] = 0.0

    if len(block_boundary_norm[block_boundary_norm == 0]) == 3:
        block_boundary_norm = np.array([[0, 0, 1]])
        bad_bt_flag = True
    return block_boundary_norm, bad_bt_flag


def choose_norm(GrainID: [int, float]):
    hp_normal_list = CrystallInfo.gid2habit_planes[GrainID]
    index = np.random.randint(4)
    chosen_norm = hp_normal_list[index, ...]
    return chosen_norm


def recursive_bisect(grain: pd.DataFrame, num_grid_points: list, chosen_norm: np.ndarray):
    points_data = grain.copy()
    i = 0
    for num in num_grid_points:
        d = -(points_data['x'] * chosen_norm[0, 0] + points_data['y'] * chosen_norm[0, 1] +
              points_data['z'] * chosen_norm[0, 2])
        d.sort_values(inplace=True)
        chosen_d = d.iloc[num - 1]

        points_idx = points_data[points_data['x'] * chosen_norm[0, 0] +
                                 points_data['y'] * chosen_norm[0, 1] +
                                 points_data['z'] * chosen_norm[0, 2] + chosen_d >= 0].index
        points_data.loc[points_idx, 'packet_id'] = str(
            int(points_data.loc[points_idx, 'GrainID'].iloc[0])) + 'p' + str(i)
        # PID2NORM[str(int(points_data.loc[points_idx, 'GrainID'].iloc[0])) + 'p' + str(i)] = chosen_norm
        CrystallInfo.pid2hp_norm[str(int(points_data.loc[points_idx, 'GrainID'].iloc[0])) + 'p' + str(i)] = chosen_norm
        CrystallInfo.pid2gid[str(int(points_data.loc[points_idx, 'GrainID'].iloc[0])) + 'p' + str(i)] = grain.iloc[0][
            'GrainID']
        packet_df = points_data.loc[points_idx]
        grain['packet_id'] = packet_df['packet_id']
        points_data.drop(packet_df.index, inplace=True)
        i += 1


def gen_packets(rve: pd.DataFrame, grains_df: pd.DataFrame, pv_distribution: SubsDistribution,
                num_packets_list: list = None) -> pd.DataFrame:
    """
    generate packets in rve
    :param num_packets_list: the number of packets in each grain
    :param grains_df: dataframe containing information of grains
    :param rve: the rve to be subdivided into packets
    :param pv_distribution: statistical distribution of packet volume
    :return: rve has packets
    """
    num_pags = int(rve['GrainID'].max())
    pid2gid = dict()
    pid2hp_norm = dict()

    for i in range(1, num_pags + 1):
        grain = rve[rve['GrainID'] == i]
        grain_data = grains_df.iloc[i - 1]
        orientation = (grain_data['phi1'], grain_data['PHI'], grain_data['phi2'])
        grain_volume = grain_data['final_discrete_volume']
        # habit plane norm
        chosen_norm = choose_norm(GrainID=i)
        # sample packet volumes
        if not RveInfo.subs_file_flag:
            pv_list = subs_sampler(subs_distribution=pv_distribution, y=grain_volume,
                                   interval=[RveInfo.pv_min, RveInfo.pv_max])
            # convert packet volumes to number of grid points in the grain
            num_grid_points = np.asarray(pv_list) / grain_volume * len(grain)
        else:
            num_packets = num_packets_list[i - 1]
            num_grid_points = [np.random.randint(20, len(grain)) for _ in range(num_packets - 1)]
            num_grid_point = len(grain) - sum(num_grid_points)
            num_grid_points.append(num_grid_point)

        if len(num_grid_points) > 1:
            recursive_bisect(grain=grain, num_grid_points=num_grid_points, chosen_norm=chosen_norm)
        else:
            rve.loc[rve['GrainID'] == i, 'packet_id'] = grain['GrainID'].astype(str)
            # PID2NORM[str(i)] = chosen_norm
            CrystallInfo.pid2gid[grain.iloc[0]['GrainID'].astype(str)] = i
            CrystallInfo.pid2hp_norm[grain.iloc[0]['GrainID'].astype(str)] = chosen_norm

    return rve


def dis_to_id(dis, bt_list):
    idx = np.where(bt_list > dis)
    if len(idx[0]) == 0:
        return str(len(bt_list))
    else:
        return str(int(idx[0][0] - 1))


def gen_blocks(rve: pd.DataFrame, bt_distribution: SubsDistribution) -> pd.DataFrame:
    # select a random norm direction for block boundary
    rve["p_dis"] = 0
    rve['pd'] = 0
    rve['block_id'] = str(0)
    num_packets = rve["packet_id"].max()
    rve['packet_id'] = rve['packet_id'].astype(str)
    for i in range(1, num_packets + 1):
        packet = rve[rve["packet_id"] == str(i)].copy()
        block_boundary_norm = np.random.uniform(0, 1, (1, 3))
        # compute d of block boundary
        pd = -(packet['x'] * block_boundary_norm[0, 0] +
               packet['y'] * block_boundary_norm[0, 1] +
               packet['z'] * block_boundary_norm[0, 2])
        # compute the distance to the block boundary with maximum d
        packet['pd'] = pd
        x_moved = y_moved = z_moved = False
        x_max = RveInfo.box_size
        y_max = RveInfo.box_size_y if RveInfo.box_size_y is not None else RveInfo.box_size
        z_max = RveInfo.box_size_y if RveInfo.box_size_z is not None else RveInfo.box_size
        if 0.0 in packet['x'].values and x_max in packet['x'].values:
            x_moved = True

        if 0.0 in packet['y'].values and y_max in packet['y'].values:
            y_moved = True

        if 0.0 in packet['z'].values and z_max in packet['z'].values:
            z_moved = True

        # print(packet['pd'])
        p1 = packet.loc[packet['pd'].idxmax(), ['x', 'y', 'z']]

        pedal_points = get_pedal_point(p1=p1, n=block_boundary_norm, d=packet['pd'])
        num_clusters = compute_num_clusters(packet=packet[['x', 'y', 'z']].to_numpy())

        kmeans = train_kmeans(num_clusters=num_clusters, packet=packet[['x', 'y', 'z']].to_numpy())
        same_side = pedal_points.apply(lambda p2: issame_side(kmeans=kmeans,
                                                              p1=p1.to_numpy(),
                                                              p2=p2.to_numpy()),
                                       axis=1)

        pedal_points['same_side'] = same_side
        p_dis = pedal_points.apply(lambda p2: dis_in_rve(same_side=p2['same_side'],
                                                         p1=p1,
                                                         p2=p2,
                                                         x_moved=x_moved,
                                                         y_moved=y_moved,
                                                         z_moved=z_moved),
                                   axis=1)

        packet['p_dis'] = p_dis

        total_bt = packet['p_dis'].max()
        bt_list = subs_sampler(subs_distribution=bt_distribution,
                               y=total_bt,
                               interval=[RveInfo.bt_min, RveInfo.bt_max])
        dis_list = np.cumsum(bt_list)
        dis_list = np.insert(dis_list, 0, 0)
        block_id = packet['p_dis'].map(lambda dis: dis_to_id(dis, dis_list))
        rve.loc[rve["packet_id"] == str(i), 'p_dis'] = p_dis
        rve.loc[rve["packet_id"] == str(i), 'block_id'] = packet['packet_id'] + 'b' + block_id

    rve['packet_id'] = rve['packet_id'].astype(int)
    return rve


def compute_bt(rve: pd.DataFrame) -> None:
    rve['block_thickness'] = np.nan
    bg = rve.groupby('block_id')
    bid_to_bt = bg['p_dis'].apply(max) - bg['p_dis'].apply(min)
    rve['block_thickness'] = rve.apply(lambda p: bid_to_bt[p['block_id']], axis=1)
    rve.drop('pd', axis=1, inplace=True)
    rve.drop('p_dis', axis=1, inplace=True)


if __name__ == '__main__':

    rve = pd.read_csv(r"/home/doelz-admin/DRAGen/dragen/test/results/substruct_data.csv")
    packet = rve[rve["packet_id"] == 1].copy()
    RveInfo.box_size_z = RveInfo.box_size_y = RveInfo.box_size = 30.0
    x_moved = y_moved = z_moved = False
    x_max = RveInfo.box_size
    y_max = RveInfo.box_size_y if RveInfo.box_size_y is not None else RveInfo.box_size
    z_max = RveInfo.box_size_y if RveInfo.box_size_z is not None else RveInfo.box_size
    if 0.0 in packet['x'].values and x_max in packet['x'].values:
        x_moved = True

    if 0.0 in packet['y'].values and y_max in packet['y'].values:
        y_moved = True

    if 0.0 in packet['z'].values and z_max in packet['z'].values:
        z_moved = True
    block_boundary_norm = np.random.uniform(0, 1, (1, 3))
    pd = -(packet['x'] * block_boundary_norm[0, 0] +
           packet['y'] * block_boundary_norm[0, 1] +
           packet['z'] * block_boundary_norm[0, 2])
    packet['pd'] = pd
    # print(packet['pd'])
    p1 = packet.loc[packet['pd'].idxmax(), ['x', 'y', 'z']]

    pedal_points = get_pedal_point(p1=p1, n=block_boundary_norm, d=packet['pd'])
    num_clusters = compute_num_clusters(packet=packet[['x', 'y', 'z']].to_numpy())

    kmeans = train_kmeans(num_clusters=num_clusters, packet=packet[['x', 'y', 'z']].to_numpy())
    same_side = pedal_points.apply(lambda p2: issame_side(kmeans=kmeans,
                                                          p1=p1.to_numpy(),
                                                          p2=p2.to_numpy()),
                                   axis=1)

    pedal_points['same_side'] = same_side
    p_dis = pedal_points.apply(lambda p2: dis_in_rve(same_side=p2['same_side'],
                                                     p1=p1,
                                                     p2=p2,
                                                     x_moved=x_moved,
                                                     y_moved=y_moved,
                                                     z_moved=z_moved),
                               axis=1)
