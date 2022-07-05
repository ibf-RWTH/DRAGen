# _*_ coding: utf-8 _*_
"""
Time:     2021/11/9 22:45
Author:   Linghao Kong
Version:  V 0.1
File:     KDTree
Describe: Write during the internship at IEHK RWTH"""

#find all adjacent blocks for each block
from dragen.utilities.InputInfo import RveInfo
import pandas as pd
import warnings
import numpy as np

class Node:
    def __init__(self):
        self.level = None # level of substructure
        self.subID = None # id of substructure
        self.father = None # father node
        self.children = [] # children nodes
        self.points = None # coordinates
        self.bt = None # block thickness if level is block


    def get_hull_df(self,rve_df: pd.DataFrame):
        subID, level = self.subID, self.level
        if level == "grain":
            sub_df = rve_df[rve_df["GrainID"] == subID].copy()

        elif level == "packet":
            sub_df = rve_df[rve_df["packet_id"] == subID].copy()

        else:
            sub_df = rve_df[rve_df["block_id"] == subID].copy()

        x_min, y_min, z_min = min(sub_df['x']), min(sub_df['y']), min(sub_df['z'])
        x_max, y_max, z_max = max(sub_df['x']), max(sub_df['y']), max(sub_df['z'])


        self.points = sub_df.loc[(sub_df['x'] == x_min) | (sub_df['x'] == x_max)|
                                 (sub_df['y'] == y_min) | (sub_df['y'] == y_max)|
                                 (sub_df['z'] == z_min) | (sub_df['z'] == z_max)]

class NodeArr:
    def __init__(self,nodes):
        self._nodes = nodes

    def __array__(self):
        ids = [node.subID for node in self._nodes]
        return np.array(ids)

def build_IDtree(rve_df):
    gids = list(set(rve_df["GrainID"]))
    grain_nodes,packet_nodes,block_nodes = [],[],[]

    rve_node = Node()
    for gid in gids:
        grain_node = Node()
        grain_node.level = "grain"
        grain_node.subID = gid
        grain_node.father = rve_node
        grain_node.get_hull_df(rve_df)
        rve_node.children.append(grain_node)
        grain_nodes.append(grain_node)

        pids = list(set(rve_df.loc[rve_df["GrainID"]==gid,"packet_id"]))
        for pid in pids:
            packet_node = Node()
            packet_node.level = "packet"
            packet_node.subID = pid
            packet_node.father = grain_node
            packet_node.get_hull_df(rve_df)
            packet_nodes.append(packet_node)
            bids = list(set(rve_df.loc[rve_df["packet_id"]==pid,"block_id"]))
            for bid in bids:
                block_node = Node()
                block_node.level = "block"
                block_node.subID = bid
                block_node.father = packet_node
                block_node.bt = rve_df.loc[rve_df["block_id"]==bid,"block_thickness"].values[0]
                block_node.get_hull_df(rve_df)
                packet_node.children.append(block_node)
                block_nodes.append(block_node)

            grain_node.children.append(packet_node)
    return grain_nodes,packet_nodes,block_nodes

def find_branchend(leave:Node):
    if len(leave.father.children) > 1:
        if isinstance(leave,np.ndarray):
            leave = leave.item(0)
        return leave
    else:
        return find_branchend(leave.father)

def istiny(node:Node,lower_bt:float):
    if node.bt < lower_bt:
        return True
    else:
        return False

def isblock(node):
    if isinstance(node,np.ndarray):
        node = node.item(0)
    if node.level == "block":
        return True
    else:
        return False

def ispacket(node):
    if isinstance(node,np.ndarray):
        node = node.item(0)
    if node.level == "packet":
        return True
    else:
        return False

def isgrain(node):
    if isinstance(node,np.ndarray):
        node = node.item(0)
    if node.level == "grain":
        return True
    else:
        return False

def relink(neighbors):
    for sub, adjacent in neighbors.items():
        if adjacent.father != sub:
            sub.father = adjacent
            adjacent.children.append(sub)
            if sub.level == "block":
                adjacent.bt = adjacent.bt + sub.bt

def get_adjacent_node(node:Node):
    if isinstance(node,np.ndarray):# the type of data changes so magic....
        node = node.item(0)
    sub_to_id = {"grain": "GrainID", "packet": "packet_id", "block": "block_id"}
    idname = sub_to_id[node.level]
    children = node.father.children
    ids = [child.subID for child in node.father.children]
    ids_to_children = dict(zip(ids,children))
    trial_space = pd.DataFrame()
    for child in node.father.children:
        trial_space = pd.concat([trial_space,child.points])

    trial_space.sort_values(by=['x','y','z'],inplace=True)
    n = 0
    while True:

        id1 = trial_space.iloc[n][idname]
        id2 = trial_space.iloc[n+1][idname]
        if id1 == node.subID and id1 != id2:
            adjacent_id = id2
            break
        elif id2 == node.subID and id1 != id2:
            adjacent_id = id1
            break
        else:
             n += 1
    neighbor = ids_to_children[adjacent_id]
    if isinstance(neighbor,np.ndarray):
        neighbor = neighbor.item(0)
    return neighbor
get_adjacent_node = np.vectorize(get_adjacent_node)
def get_root(node:Node):
    # get id
    while node.father.level == node.level:
        node = node.father

    return node

def reset_father(node:Node,father:Node):
    node.father = father
reset_father = np.vectorize(reset_father)
def rebuild_tree(nodes):

    for node in nodes:
        if node.father.level == node.level:
            node.father.children.remove(node)
            if len(node.children) > 0:
                node.father.children.extend(node.children)
                reset_father(node.children, node.father)

def clean_children(node:Node):
    node.children = list(filter(lambda child: child.father == node, node.children))
clean_children = np.vectorize(clean_children)

istiny = np.vectorize(istiny)
find_branchend = np.vectorize(find_branchend)
isblock,ispacket,isgrain = np.vectorize(isblock),np.vectorize(ispacket),np.vectorize(isgrain)
'''
1.build substructure tree
2.find all tiny blocks
3.get branchends for tiny blocks
4. find neighbors for all branchends
5. relink branchends
6. old_id->new_id old_bt->new_bt (if block ends)
7. modify rve_df
8. repeat 1-7 until no tiny blocks
'''
def merge_tiny_blocks(rve_df,lower_bt):
# step 1
    grain_nodes,packet_nodes,block_nodes = build_IDtree(rve_df)
    while True:

        if len(grain_nodes) == 1 and len(packet_nodes) == 1 and len(block_nodes) == 1:
            print("the blocks in grain {} can't be merged anymore".format(grain_nodes[0].subID))
            break
        old_bid = [block.subID for block in block_nodes]
        bid_to_new = dict(zip(old_bid,old_bid))
        old_pid = [packet.subID for packet in packet_nodes]
        pid_to_new = dict(zip(old_pid,old_pid))
        old_gid = [grain.subID for grain in grain_nodes]
        gid_to_new = dict(zip(old_gid,old_gid))
        old_bt = [block.bt for block in block_nodes]
        bt_to_new = dict(zip(old_bid,old_bt))

        #step 2
        barr = np.asarray(block_nodes)
        tiny_blocks = list(barr[istiny(block_nodes,lower_bt)])
        #print(len(tiny_blocks))
        if len(tiny_blocks) == 0:
            break

        #step 3
        branchends = find_branchend(tiny_blocks)
        block_ends = branchends[isblock(branchends)]
        packet_ends = branchends[ispacket(branchends)]
        grain_ends = branchends[isgrain(branchends)]

        #step 4
        block_neighbors, packet_neighbors, grain_neighbors = None, None, None
        if len(block_ends)>0:
            bneighbors = get_adjacent_node(block_ends)
            try:
                block_ends = [block.item(0) for block in block_ends]
            except:
                pass
            block_neighbors = dict(zip(block_ends,bneighbors))
        if len(packet_ends) > 0:
            pneighbors = get_adjacent_node(packet_ends)
            packet_ends = [packet.item(0) for packet in packet_ends]
            packet_neighbors = dict(zip(packet_ends, pneighbors))

        if len(grain_ends) > 0:
            gneighbors = get_adjacent_node(grain_ends)
            grain_ends = [grain.item(0) for grain in grain_ends]
            grain_neighbors = dict(zip(grain_ends,gneighbors))

        #step 5
        if block_neighbors is not None:
            relink(block_neighbors)
        if packet_neighbors is not None:
            relink(packet_neighbors)
        if grain_neighbors is not None:
            relink(grain_neighbors)

        #step 6

        for block in block_nodes:
            root = get_root(block)
            bid_to_new[block.subID] = root.subID
            bt_to_new[block.subID] = root.bt
            bt_to_new[root.subID] = root.bt

        for packet in packet_nodes:
            root = get_root(packet)
            pid_to_new[packet.subID] = root.subID

        for grain in grain_ends:
            root = get_root(grain)
            gid_to_new[grain.subID] = root.subID

        #step7
        new_bid = rve_df["block_id"].apply(lambda bid:bid_to_new[bid])
        new_bt = rve_df["block_id"].apply(lambda bid:bt_to_new[bid])
        new_pid = rve_df["packet_id"].apply(lambda pid:pid_to_new[pid])
        new_gid = rve_df["GrainID"].apply(lambda gid:gid_to_new[gid])

        rve_df["block_id"] = new_bid
        rve_df["block_thickness"] = new_bt
        rve_df["packet_id"] = new_pid
        rve_df["GrainID"] = new_gid

        #step 8
        rebuild_tree(block_nodes)
        rebuild_tree(packet_nodes)
        rebuild_tree(grain_nodes)

        block_nodes = list(filter(lambda node:node.level != node.father.level,block_nodes))
        packet_nodes = list(filter(lambda node:node.level != node.father.level,packet_nodes))
        clean_children(packet_nodes)
        grain_nodes = list(filter(lambda node:node.level != node.father.level,grain_nodes))

    return rve_df

def mod_bt(rve_df):
    grain_nodes, packet_nodes, block_nodes = build_IDtree(rve_df)
    print("modifying block thickness...")
    # compute the needed number of blocks
    needed_nb = int(round(sum([block.bt for block in block_nodes]) / RveInfo.t_mu))
    while True:
        # get the number of generated blocks
        gen_nb = len(block_nodes)
        # print(gen_nb)
        # how many blocks are needed to be merged
        n = gen_nb - needed_nb
        if n == 0:
            break

        elif n > 0:
            selected_nodes = np.random.choice(block_nodes, n)
            old_bid = [block.subID for block in block_nodes]
            bid_to_new = dict(zip(old_bid,old_bid))
            old_pid = [packet.subID for packet in packet_nodes]
            pid_to_new = dict(zip(old_pid,old_pid))
            old_gid = [grain.subID for grain in grain_nodes]
            gid_to_new = dict(zip(old_gid,old_gid))
            old_bt = [block.bt for block in block_nodes]
            bt_to_new = dict(zip(old_bid,old_bt))

            #step 3
            branchends = find_branchend(selected_nodes)
            block_ends = branchends[isblock(branchends)]
            packet_ends = branchends[ispacket(branchends)]
            grain_ends = branchends[isgrain(branchends)]

            #step 4
            block_neighbors, packet_neighbors, grain_neighbors = None, None, None
            if len(block_ends)>0:
                bneighbors = get_adjacent_node(block_ends)
                try:
                    block_ends = [block.item(0) for block in block_ends]
                except:
                    pass
                block_neighbors = dict(zip(block_ends,bneighbors))
            if len(packet_ends) > 0:
                pneighbors = get_adjacent_node(packet_ends)
                packet_ends = [packet.item(0) for packet in packet_ends]
                packet_neighbors = dict(zip(packet_ends, pneighbors))

            if len(grain_ends) > 0:
                gneighbors = get_adjacent_node(grain_ends)
                grain_ends = [grain.item(0) for grain in grain_ends]
                grain_neighbors = dict(zip(grain_ends,gneighbors))

            #step 5
            if block_neighbors is not None:
                relink(block_neighbors)
            if packet_neighbors is not None:
                relink(packet_neighbors)
            if grain_neighbors is not None:
                relink(grain_neighbors)

            #step 6

            for block in block_nodes:
                root = get_root(block)
                bid_to_new[block.subID] = root.subID
                bt_to_new[block.subID] = root.bt
                bt_to_new[root.subID] = root.bt

            for packet in packet_nodes:
                root = get_root(packet)
                pid_to_new[packet.subID] = root.subID

            for grain in grain_ends:
                root = get_root(grain)
                gid_to_new[grain.subID] = root.subID

            #step7
            new_bid = rve_df["block_id"].apply(lambda bid:bid_to_new[bid])
            new_bt = rve_df["block_id"].apply(lambda bid:bt_to_new[bid])
            new_pid = rve_df["packet_id"].apply(lambda pid:pid_to_new[pid])
            new_gid = rve_df["GrainID"].apply(lambda gid:gid_to_new[gid])

            rve_df["block_id"] = new_bid
            rve_df["block_thickness"] = new_bt
            rve_df["packet_id"] = new_pid
            rve_df["GrainID"] = new_gid

            #step 8
            rebuild_tree(block_nodes)
            rebuild_tree(packet_nodes)
            rebuild_tree(grain_nodes)

            block_nodes = list(filter(lambda node:node.level != node.father.level,block_nodes))
            packet_nodes = list(filter(lambda node:node.level != node.father.level,packet_nodes))
            grain_nodes = list(filter(lambda node:node.level != node.father.level,grain_nodes))

        else:
            pass

    print("modify block thickness successfully")
    return rve_df

if __name__ == "__main__":
    rve_df = pd.read_csv("F:/git/merged_substructure/OutputData/2021-10-29_0/substruct_data_abq.csv")
    grain_nodes, packet_nodes, block_nodes = build_IDtree(rve_df)
    new_df = mod_bt(rve_df,1.5)
    blocks = new_df.groupby("block_id").first()
    print(blocks["block_thickness"].mean())




























