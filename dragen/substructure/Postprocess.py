from typing import Union
import pandas as pd


def sub_min(item1: Union[int, float, None], item2: Union[int, float, None]) -> Union[int, float]:
    assert item1 is not None or item2 is not None
    if item1 is None:
        return item2
    elif item2 is None:
        return item1
    else:
        return min(item1, item2)


class SubsNode:
    def __init__(self, sub_id: int, sub_count: int, sub_value: Union[int, float] = 0) -> None:
        self.sub_id = sub_id
        self.sub_value = sub_value
        self.sub_count = sub_count
        self.father = None
        self.son = None
        self.prev = None
        self.next = None

    def set_prev(self, prev) -> None:
        self.prev = prev
        if prev is not None:
            prev.next = self


def getValue(sub_node: Union[SubsNode, None]) -> Union[SubsNode, None]:
    if sub_node is None:
        return None
    else:
        return sub_node.sub_value


def getID(sub_node: Union[SubsNode, None]) -> Union[int, None]:
    if sub_node is None:
        return None
    else:
        return sub_node.sub_id

def getCount(sub_node: Union[SubsNode, None]) -> Union[int, None]:
    if sub_node is None:
        return None
    else:
        return sub_node.sub_count


def minValueNode(item1: Union[SubsNode, None], item2: Union[SubsNode, None]) -> SubsNode:
    assert item1 is not None or item2 is not None
    if item1 is None:
        return item2
    elif item2 is None:
        return item1
    else:
        if item1.sub_value < item2.sub_value:
            return item1
        else:
            return item2


def minCountNode(item1: Union[SubsNode, None], item2: Union[SubsNode, None]) -> SubsNode:
    assert item1 is not None or item2 is not None
    if item1 is None:
        return item2
    elif item2 is None:
        return item1
    else:
        if item1.sub_count < item2.sub_count:
            return item1
        else:
            return item2


def trackFather(sub_node: Union[None, SubsNode]) -> Union[None, SubsNode]:
    if sub_node is None:
        return None
    elif sub_node.father is None:
        return sub_node
    else:
        return sub_node.father


def trackForefather(sub_node: SubsNode) -> SubsNode:
    if sub_node.father is not None:
        return trackForefather(sub_node=sub_node.father)
    else:
        return sub_node


def getNext(sub_node: Union[SubsNode, None]) -> Union[SubsNode, None]:
    if sub_node is None:
        return None
    else:
        return sub_node.next


def merge_tiny_subs(sub_nodes: list, min_sub_count: Union[int, float], sub_num: int) -> list:
    while True:
        if len(sub_nodes) == 1:
            return sub_nodes
        count = 0  # count number of tiny sub
        root = sub_nodes[0]
        while root is not None:
            # iterate over all sub nodes
            if root.sub_count < min_sub_count:
                count += 1
                prevFather = trackFather(root.prev)
                if sub_num == 2:
                    min_svalue = sub_min(item1=getValue(prevFather),
                                         item2=getValue(root.next))
                    new_svalue = root.sub_value + min_svalue
                    merge_node = minValueNode(item1=prevFather,
                                              item2=root.next)
                else:
                    min_svalue = sub_min(item1=getCount(prevFather),
                                         item2=getCount(root.next))
                    new_svalue = root.sub_value + min_svalue
                    merge_node = minCountNode(item1=prevFather,
                                              item2=root.next)
                # form new sub nodes tree
                new_sub_node = SubsNode(sub_id=merge_node.sub_id,
                                        sub_count=merge_node.sub_count + root.sub_count,
                                        sub_value=new_svalue)
                root.father = new_sub_node
                merge_node.father = new_sub_node

                if merge_node is root.next:
                    root = root.next

            root = getNext(sub_node=root)

        if count == 0:  # no tiny sub then break
            return sub_nodes

        # update sub nodes tree
        new_sub_nodes = []
        prev_forefather = None
        for sub_node in sub_nodes:
            forefather = trackForefather(sub_node)
            if prev_forefather is not forefather:
                new_sub_nodes.append(forefather)
                forefather.set_prev(prev=prev_forefather)
            prev_forefather = forefather

        if len(sub_nodes) == len(new_sub_nodes):
            return sub_nodes

        sub_nodes = new_sub_nodes


def create_block_nodes(packet: pd.DataFrame) -> list:
    blocks = packet.groupby("block_id").head(1)
    sub_nodes = []
    for i in range(len(blocks)):
        bid = blocks.iloc[i]['block_id']
        bt = blocks.iloc[i]['block_thickness']
        counts = packet['block_id'].value_counts()[bid]
        sub_node = SubsNode(sub_id=bid,
                            sub_count=counts,
                            sub_value=bt)
        if i > 0:
            sub_node.set_prev(prev=sub_nodes[-1])
        sub_nodes.append(sub_node)

    return sub_nodes


def create_packet_nodes(grain: pd.DataFrame) -> list:
    packets = grain.groupby("packet_id").head(1)
    sub_nodes = []
    for i in range(len(packets)):
        pid = packets.iloc[i]['packet_id']
        counts = grain['packet_id'].value_counts()[pid]
        sub_node = SubsNode(sub_id=pid,
                            sub_count=counts)
        if i > 0:
            sub_node.set_prev(prev=sub_nodes[-1])
        sub_nodes.append(sub_node)

    return sub_nodes


def merge_tiny_blocks(rve: pd.DataFrame, min_num_points) -> None:
    for i in range(1, int(rve['packet_id'].max()) + 1):
        packet = rve[rve['packet_id'] == i]
        sub_nodes = create_block_nodes(packet=packet)
        merge_tiny_subs(sub_nodes=sub_nodes, min_sub_count=min_num_points, sub_num=2)
        new_bts = []
        new_bids = []
        old_bids = []
        for sub_node in sub_nodes:
            forefather = trackForefather(sub_node=sub_node)
            old_bids.append(sub_node.sub_id)
            new_bids.append(forefather.sub_id)
            new_bts.append(forefather.sub_value)

        old_bid2new_bid = dict(zip(old_bids, new_bids))
        old_bid2new_bt = dict(zip(old_bids, new_bts))

        new_bids = packet.apply(lambda p: old_bid2new_bid[p['block_id']], axis=1)
        new_bts = packet['block_id'].apply(lambda bid: old_bid2new_bt[bid])
        rve.loc[packet.index, 'block_id'] = new_bids
        rve.loc[packet.index, 'block_thickness'] = new_bts


def merge_tiny_packets(rve: pd.DataFrame, min_num_points) -> None:
    for i in range(1, int(rve['GrainID'].max()) + 1):
        grain = rve[rve['GrainID'] == i]
        sub_nodes = create_packet_nodes(grain=grain)
        merge_tiny_subs(sub_nodes=sub_nodes, min_sub_count=min_num_points, sub_num=1)
        new_pids = []
        old_pids = []
        for sub_node in sub_nodes:
            forefather = trackForefather(sub_node=sub_node)
            old_pids.append(sub_node.sub_id)
            new_pids.append(forefather.sub_id)

        old_bid2new_bid = dict(zip(old_pids, new_pids))

        new_pids = grain.apply(lambda p: old_bid2new_bid[p['packet_id']], axis=1)
        rve.loc[grain.index, 'packet_id'] = new_pids

if __name__ == "__main__":
    # sub_values = np.random.randint(1,15,(1,10))
    # sub_values = np.array([[4, 8, 8, 11, 29]])
    #
    # print(sub_values)
    #
    # sub_nodes = []
    # for i in range(1, 6):
    #     sub_node = SubsNode(sub_id=i, sub_value=sub_values[0, i - 1], sub_count=sub_values[0, i - 1])
    #     if i > 1:
    #         sub_node.set_prev(sub_nodes[-1])
    #     sub_nodes.append(sub_node)
    # #
    # merged_sub_nodes = merge_tiny_subs(sub_nodes=sub_nodes, min_sub_count=10)
    # print(len(merged_sub_nodes))
    # root = merged_sub_nodes[0]
    # while root is not None:
    #     print("value is", root.sub_value)
    #     print("id is", root.sub_id)
    #     root = root.next
    # #
    # for sub_node in sub_nodes:
    #     forefather = trackForefather(sub_node=sub_node)
    #     print(forefather.sub_value)
    #     print(sub_node.sub_id, forefather.sub_id)

    data = pd.read_csv(r"P:\DRAGen\DRAGen\OutputData\2023-01-11_000\substruct_data.csv")
    merge_tiny_packets(rve=data, min_packet_counts=20)
    data.to_csv(r"P:\DRAGen\DRAGen\OutputData\2023-01-11_000\substruct_data3.csv")
    print(data['GrainID'].value_counts())
