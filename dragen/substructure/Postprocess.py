import pandas as pd
from dragen.substructure.postprocessing import SubsNode, merge_tiny_subs
from dragen.substructure.Crystallograhy import CrystallInfo


def merge_tiny_packets(grain: pd.DataFrame, min_num_points: int) -> pd.DataFrame:
    """
    merge tiny grains
    :param grain: dataframe of grain
    :param min_num_points: minimum number of grid points in a substructure
    :return: None
    """
    grain['packet_id'] = grain['packet_id'].astype(int)
    while True:
        sub_nodes = []
        pids = grain['packet_id'].unique()
        num_points = grain.groupby("packet_id")['packet_id'].count()
        if (num_points >= min_num_points).all() or len(pids) == 1:
            break

        i = 0
        for pid in pids:
            if i == 0:
                sub_nodes.append(SubsNode(int(pid), float(len(grain[grain['packet_id'] == pid]))))
            else:
                sub_nodes.append(SubsNode(int(pid), sub_nodes[i - 1], float(len(grain[grain['packet_id'] == pid]))))
            i += 1

        pid2nodes = dict(zip(pids, sub_nodes))
        # print(pid2nodes)
        merge_tiny_subs(sub_nodes[-1], float(min_num_points))
        new_pids = grain.apply(lambda data: pid2nodes[data['packet_id']].id, axis=1)
        grain['packet_id'] = new_pids

        for pid in pids:
            new_pid = pid2nodes[pid].id
            if pid != new_pid:
                CrystallInfo.pid2variants[CrystallInfo.old_pid[pid - 1]] = \
                    CrystallInfo.pid2variants[CrystallInfo.old_pid[new_pid - 1]]

        if len(pids) == len(new_pids.unique()):
            break

    return grain


def merge_all_tiny_packets(rve: pd.DataFrame, min_num_points: int):
    num_grains = int(rve['GrainID'].max())
    for i in range(1, num_grains + 1):
        grain = rve[rve['GrainID'] == i].copy()
        grain = merge_tiny_packets(grain, min_num_points)
        rve.loc[rve['GrainID'] == i, 'packet_id'] = grain['packet_id']
        print(f"finish merging in grain{i}")


def merge_tiny_blocks(packet: pd.DataFrame, min_num_points: int) -> pd.DataFrame:
    packet['block_id'] = packet['block_id'].astype(int)
    while True:
        sub_nodes = []
        pids = packet['block_id'].unique()
        # print("old ids are", pids)
        num_points = packet.groupby("block_id")['block_id'].count()
        if (num_points >= min_num_points).all() or len(pids) == 1:
            break

        i = 0
        for pid in pids:
            if i == 0:
                sub_nodes.append(SubsNode(int(pid), float(packet.loc[packet['block_id'] == pid,
                                                                     'block_thickness'].unique()[0])))
            else:
                sub_nodes.append(SubsNode(int(pid), sub_nodes[i - 1], float(packet.loc[packet['block_id'] == pid,
                                                                                       'block_thickness'].unique()[0])))
            i += 1

        pid2nodes = dict(zip(pids, sub_nodes))
        # print(pid2nodes)
        merge_tiny_subs(sub_nodes[-1], float(min_num_points))
        new_pids = packet.apply(lambda data: pid2nodes[data['block_id']].id, axis=1)
        new_bts = packet.apply(lambda data: pid2nodes[data['block_id']].size, axis=1)
        # print("new ids are", new_pids.unique())
        packet['block_id'] = new_pids
        packet['block_thickness'] = new_bts
        if len(pids) == len(new_pids.unique()):
            break

    return packet
    # print("info are", packet.groupby("block_id")['block_id'].count())


def merge_all_tiny_blocks(rve: pd.DataFrame, min_num_points: int):
    num_packets = rve['packet_id'].max()
    for i in range(1, num_packets + 1):
        packet = rve[rve['packet_id'] == i].copy()
        packet = merge_tiny_blocks(packet, min_num_points)
        rve.loc[rve['packet_id'] == i, 'block_id'] = packet['block_id']
        rve.loc[rve['packet_id'] == i, 'block_thickness'] = packet['block_thickness']
        # print(rve[rve['packet_id'] == packet_id])
        print(f"finish merging in packet{i}")


if __name__ == "__main__":
    rve = pd.read_csv(r"F:\codes\DRAGen\OutputData\2022-11-16_000\substruct_data.csv")
    rve.sort_values(by=['x', 'y', 'z'], inplace=True)
    rve['packet_id'] = rve['packet_id'].astype(int)
    # merge_tiny_packets(rve=rve, min_num_points=10)
    # print(rve.groupby("packet_id")['packet_id'].count())
    print(rve.groupby("packet_id")['packet_id'].count())
    merge_all_tiny_packets(rve, 20)
    print(rve.groupby("packet_id")['packet_id'].count())
    print(rve.groupby("block_id")['block_id'].count())
    merge_all_tiny_blocks(rve, 20)
    print(rve.groupby("block_id")['block_id'].count())
