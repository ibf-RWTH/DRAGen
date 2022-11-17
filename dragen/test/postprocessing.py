import pandas as pd
from dragen.substructure.postprocessing import SubsNode, merge_tiny_subs
from dragen.substructure.run import subsid2num

def hasid(subs_id, id_list):
    return subs_id in id_list


def merge_tiny_packets(rve: pd.DataFrame, min_num_points: int) -> None:
    """
    merge tiny grains
    :param subs_key: GrainID, packet_id or block id
    :param rve: dataframe of rve
    :param min_num_points: minimum number of grid points in a substructure
    :return: None
    """
    while True:
        sub_nodes = []
        num_subs = rve['packet_id'].max()
        print(num_subs)
        num_points = rve.groupby("packet_id")['packet_id'].count()
        if (num_points > min_num_points).all():
            break
        for i in range(1, num_subs + 1):
            if i == 1:
                sub_nodes.append(SubsNode(i, float(len(rve[rve['packet_id'] == i]))))
            else:
                sub_nodes.append(SubsNode(i, sub_nodes[i - 2], float(len(rve[rve['packet_id'] == i]))))

        merge_tiny_subs(sub_nodes[-1], float(min_num_points))

        for sub_node in sub_nodes:
            mlen = sub_node.get_merge_list_len()
            if not sub_node.merged and mlen > 0:
                merge_list = [sub_node.getMergedNode(i).id for i in range(mlen)]
                print(sub_node.id, merge_list)
                truth_values = rve.apply(lambda data: hasid(data['packet_id'], merge_list), axis=1)
                rve.loc[truth_values, 'packet_id'] = sub_node.id

        subsid2num(_rve_data=rve,subs_key='packet_id')
        # new_num_subs = rve['packet_id'].max()
        # if new_num_subs == num_subs:
        #     break



if __name__ == "__main__":
    rve = pd.read_csv(r"F:\codes\DRAGen\OutputData\2022-11-16_000\substruct_data.csv")
    # merge_tiny_packets(rve=rve, min_num_points=10)
    # print(rve.groupby("packet_id")['packet_id'].count())

    nids = rve['packet_id'].unique()
    pnodes = []
    i = 0
    for nid in nids:
        if i == 0:
            pnodes.append(SubsNode(i,float(len(rve[rve['packet_id'] == nid]))))
        else:
            pnodes.append(SubsNode(i, pnodes[i-1], float(len(rve[rve['packet_id'] == nid]))))
        i += 1

    for pnode in pnodes:
        print(pnode.size)

    print(pnodes[-1].next.id)
    print(pnodes[-1].size)
    print(pnodes[-1].merged)
    merge_tiny_subs(pnodes[-1], 10)
    print(pnodes[-1].next.id)
    print(pnodes[-1].size)
    print(pnodes[-1].merged)

