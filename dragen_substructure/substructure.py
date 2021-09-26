# _*_ coding: utf-8 _*_
"""
Time:     2021/8/19 15:47
Author:   Linghao Kong
Version:  V 0.1
File:     new_substructure
Describe: Write during the internship at IEHK RWTH"""
from dragen.utilities.RVE_Utils import RVEUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import math

class Grain(RVEUtils):

    def __init__(self,v,grainID = 1,phaseID = 2,orientation = None,alpha = 0,
                 points = None,OR = 'KS'):

        self.volume = v
        self.alpha = alpha
        self.points = points
        self.orientation = orientation
        self.grainID = int(grainID)
        self.phaseID = int(phaseID)
        self.packets_list = []
        self.pid_to_packet = dict()

        if isinstance(points,np.ndarray):

            self.points_data = pd.DataFrame(points,columns=['x','y','z'])
            self.points_data['GrainID'] = grainID
            self.points_data['phaseID'] = phaseID

        variants_list = []
        for j in range(1, 5):
            packet_variants_list = []
            for i in range(1, 7):
                packet_variants_list.append('V' + str(i + 6 * (j - 1)))

            variants_list.append(packet_variants_list)

        self.variants = np.array(variants_list)

        if OR == 'KS':
            self.hp_normal_list = self.lc_to_gc([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]])

        self.second_cut_list = []

    def lc_to_gc(self,habit_plane_list):

        orientation = self.orientation

        R1 = np.array([[np.cos(np.deg2rad(orientation[0])), -np.sin(np.deg2rad(orientation[0])), 0],
                       [np.sin(np.deg2rad(orientation[0])), np.cos(np.deg2rad(orientation[0])), 0],
                       [0, 0, 1]])

        R2 = np.array([[1, 0, 0],
                       [0, np.cos(np.deg2rad(orientation[1])), -np.sin(np.deg2rad(orientation[1]))],
                       [0, np.sin(np.deg2rad(orientation[1])), np.cos(np.deg2rad(orientation[1]))]])

        R3 = np.array([[np.cos(np.deg2rad(orientation[2])), -np.sin(np.deg2rad(orientation[2])), 0],
                       [np.sin(np.deg2rad(orientation[2])), np.cos(np.deg2rad(orientation[2])), 0],
                       [0, 0, 1]])

        result = np.dot(R3, R2)
        R = np.matrix(np.dot(result, R1))
        R_I = R.I

        habit_plane_list = np.array(habit_plane_list)
        transferred_hp_list = (R_I.dot(habit_plane_list.T)).T

        return transferred_hp_list

    def gen_subs(self, equiv_d=None, sigma=0.1, block_thickness=0.5, b_sigma=0.1, lower_t=None, upper_t=None, circularity=1,n_pack=None,orientations=None):

        if n_pack is None:
            r = equiv_d / 2
            average_pv = 4 / 3 * np.pi * r ** 3 * circularity ** (1.5)
            av_pn = round(self.volume / average_pv)
            self.points_data['block_variant'] = None

        else:
            av_pn = n_pack
            self.points_data['phi1'] = None
            self.points_data['PHI'] = None
            self.points_data['phi2'] = None

        if av_pn < 1:
            av_pn = 1
        # average num of packet
        av_points_num = int(len(self.points_data) / av_pn)
        # average num of grid points contained in packets
        LN = lognorm(s=sigma, scale=av_points_num)
        num_grid_ps = LN.rvs(av_pn)
        out_lrange_num = num_grid_ps[num_grid_ps <= 40]
        out_hrange_num = num_grid_ps[num_grid_ps >= len(self.points_data)]
        # the num of points in packet must be larger than 40 smaller than half of the grain
        num_grid_ps = np.delete(num_grid_ps, num_grid_ps <= 40)
        num_grid_ps = np.delete(num_grid_ps, num_grid_ps >= len(self.points_data))

        if len(num_grid_ps) > 0:
            increasing = (sum(out_lrange_num) + sum(out_hrange_num)) / len(num_grid_ps)
            num_grid_ps = num_grid_ps + increasing  # add the out range num to rest points

        else:
            num_grid_ps = np.array([len(self.points_data)])  # if the points num all smaller than 20, this is a small grain,only 1 packet is generated

        factor = len(self.points_data) / sum(num_grid_ps)
        num_grid_ps = num_grid_ps * factor
        num_grid_ps = num_grid_ps.astype(int)

        if sum(num_grid_ps) - len(self.points_data) > 0:

            for i in range(sum(num_grid_ps) - len(self.points_data)):
                num_grid_ps[i] -= 1

        if len(self.points_data) - sum(num_grid_ps) > 0:

            for i in range(len(self.points_data) - sum(num_grid_ps)):
                num_grid_ps[i] += 1

        points_data = self.points_data.copy()
        points_data['packet_id'] = '0'
        self.points_data['packet_id'] = '0'
        self.points_data['block_id'] = '0'
        self.points_data['block_thickness'] = 0

        i = 0
        for num in num_grid_ps:

            trial_index = [0, 1, 2, 3]
            if i > 0:
                trial_index.remove(self.packets_list[i-1].chosen_nidx)
                index = np.random.choice(trial_index,1)[0]

            else:
                index = np.random.randint(4)

            chosen_norm = self.hp_normal_list[index, ...]
            d = -(points_data['x'] * chosen_norm[0, 0] + points_data['y']* chosen_norm[0, 1] +
                  points_data['z'] * chosen_norm[0, 2])
            d.sort_values(inplace=True)
            chosen_d = d.iloc[num-1]

            # self.points_data.loc[chosen_index,'packet_id'] = str(int(self.points_data.loc[chosen_index,'GrainID'].iloc[0])) + 'p' + str(i)
            points_idx = points_data[points_data['x'] * chosen_norm[0, 0] +
                                     points_data['y'] * chosen_norm[0, 1] +
                                     points_data['z'] * chosen_norm[0, 2] + chosen_d >= 0].index
            points_data.loc[points_idx, 'packet_id'] = str(int(points_data.loc[points_idx, 'GrainID'].iloc[0])) + 'p' + str(i)

            packet_df = points_data.loc[points_idx]
            packet = Packet(packet_df)
            packet.chosen_nidx = index
            packet.boundary = chosen_norm
            packet.variants = self.trial_variants_select(packet)
            packet.pag_ori = self.orientation
            packet.gen_blocks(t_mu=block_thickness, sigma=b_sigma, lower=lower_t,upper=upper_t)  # complete df
            comp_df = packet.get_bt()

            self.points_data.loc[comp_df.index, 'packet_id'] = packet['id']
            self.points_data.loc[comp_df.index, 'block_id'] = comp_df['block_id']
            self.points_data.loc[comp_df.index, 'block_thickness'] = comp_df['block_thickness']

            self.packets_list.append(packet)
            self.pid_to_packet[packet['id']] = packet
            points_data.drop(packet_df.index, inplace=True)

            i += 1

        if orientations is None:
            for packet in self.packets_list:
                comp_df =  packet.assign_bv()
                blocks = packet.points_data.groupby('block_id').head(1)
                bid = blocks.apply(lambda block: block['block_id'], axis=1)
                angles = blocks.apply(lambda block: packet.comp_angle(self.orientation, block), axis=1)
                bid_to_angles = dict(zip(bid, angles))
                phi1 = packet.points_data['block_id'].map(lambda bid: bid_to_angles[bid][0])
                PHI = packet.points_data['block_id'].map(lambda bid: bid_to_angles[bid][1])
                phi2 = packet.points_data['block_id'].map(lambda bid: bid_to_angles[bid][2])
                self.points_data.loc[comp_df.index, 'block_variant'] = comp_df['block_variant']
                self.points_data.loc[comp_df.index,'phi1'] = phi1
                self.points_data.loc[comp_df.index,'PHI'] = PHI
                self.points_data.loc[comp_df.index,'phi2'] = phi2

        else:
            for i in range(len(self.packets_list)):
                self.packets_list[i].orientations = orientations[i]
                comp_df = self.packets_list[i].assign_block_ori()
                self.points_data.loc[comp_df.index, 'phi1'] = comp_df['phi1']
                self.points_data.loc[comp_df.index, 'PHI'] = comp_df['PHI']
                self.points_data.loc[comp_df.index, 'phi2'] = comp_df['phi2']

        return self.points_data

    def trial_variants_select(self, packet):

        normal = packet.boundary
        variants = self.variants
        h_list = self.hp_normal_list
        hp_list = []
        for h in h_list:
            hp_list.append(list(np.array(h).squeeze()))
        hp_list = np.array(hp_list)

        possible_v = variants[np.where((hp_list == normal).all(axis = 1))[0],:]

        return possible_v

class Plane:

    def __init__(self, normal, d):
        self.normal = normal
        self.d = d

        if isinstance(normal,np.ndarray):

            self.__dict = {'a':normal[...,0],'b':normal[...,1],'c':normal[...,2],'d':d}

        else:

            self.__dict = {'a': normal[0], 'b': normal[1], 'c': normal[2], 'd': d}

    def __getitem__(self, item):

        if isinstance(item,str):

            return self.__dict[item]

        if isinstance(item,np.ndarray):

            return self.normal[...,item]

        else:
            return self.normal[item]

def plot_rve_subs(rve_data,subs_name,store_path = None):

    if subs_name == 'Grain' or 'phase':
        ID = '%sID' % subs_name

    if subs_name == 'packet' or 'block':
        ID = '%s_id' % subs_name


    subs_list = list(set(rve_data[ID]))
    fig = plt.figure(dpi=200)
    ax = fig.gca(projection='3d')

    for subs in subs_list:
        points_data = rve_data[rve_data[ID] == subs]
        ax.plot(points_data['x'],points_data['y'],points_data['z'],'x',markersize=2)

    if store_path:

        plt.savefig(store_path + '/%s.png'%subs_name)

    plt.show()

T_list = [np.array(0) for i in range(24)]
T_list[0] = np.array([[0.742, 0.667, 0.075],
                      [0.650, 0.742, 0.167],
                      [0.167, 0.075, 0.983]])

T_list[1] = np.array([[0.075, 0.667, -0.742],
                      [-0.167, 0.742, 0.650],
                      [0.983, 0.075, 0.167]])

T_list[2] = np.array([[-0.667, -0.075, 0.742, ],
                      [0.742, -0.167, 0.650],
                      [0.075, 0.983, 0.167]])

T_list[3] = np.array([[0.667, -0.742, 0.075],
                      [0.742, 0.650, -0.167],
                      [0.075, 0.167, 0.983]])

T_list[4] = np.array([[-0.075, 0.742, -0.667],
                      [-0.167, 0.650, 0.742],
                      [0.983, 0.167, 0.075]])

T_list[5] = np.array([[-0.742, 0.075, 0.667],
                      [0.650, -0.167, 0.742],
                      [0.167, 0.983, 0.075]])

T_list[6] = np.array([[-0.075, 0.667, 0.742],
                      [-0.167, -0.742, 0.650],
                      [0.983, -0.075, 0.167]])

T_list[7] = np.array([[-0.742, -0.667, 0.075],
                      [0.650, -0.742, -0.167],
                      [0.167, -0.075, 0.983]])

T_list[8] = np.array([[0.742, 0.075, -0.667],
                      [0.650, 0.167, 0.742],
                      [0.167, -0.983, 0.075]])

T_list[9] = np.array([[0.075, 0.742, 0.667],
                      [-0.167, -0.650, 0.742],
                      [0.983, -0.167, 0.075]])

T_list[10] = np.array([[-0.667, -0.742, -0.075],
                       [0.742, -0.650, -0.167],
                       [0.075, -0.167, 0.983]])

T_list[11] = np.array([[0.667, -0.075, -0.742],
                       [0.742, 0.167, 0.650],
                       [0.075, -0.983, 0.167]])

T_list[12] = np.array([[0.667, 0.742, -0.075],
                       [-0.742, 0.650, -0.167],
                       [-0.075, 0.167, 0.983]])

T_list[13] = np.array([[-0.667, 0.075, -0.742],
                       [-0.742, -0.167, 0.650],
                       [-0.075, 0.983, 0.167]])

T_list[14] = np.array([[0.075, -0.667, 0.742],
                       [0.167, 0.742, 0.650],
                       [-0.983, 0.075, 0.167]])

T_list[15] = np.array([[0.742, 0.667, 0.075],
                       [-0.650, 0.742, -0.167],
                       [-0.167, 0.075, 0.983]])

T_list[16] = np.array([[-0.742, 0.075, -0.667],
                       [-0.650, -0.167, 0.742],
                       [-0.167, 0.983, 0.075]])

T_list[17] = np.array([[-0.075, -0.742, 0.667],
                       [0.167, 0.650, 0.742],
                       [-0.983, 0.167, 0.075]])

T_list[18] = np.array([[0.742, -0.075, 0.667],
                       [0.650, -0.167, -0.742],
                       [0.167, 0.983, -0.075]])

T_list[19] = np.array([[0.075, -0.742, -0.667],
                       [-0.167, 0.650, -0.742],
                       [0.983, 0.167, -0.075]])

T_list[20] = np.array([[-0.667, 0.742, 0.075],
                       [0.742, 0.650, 0.167],
                       [0.075, 0.167, -0.983]])

T_list[21] = np.array([[0.667, 0.075, 0.742],
                       [0.742, -0.167, -0.650],
                       [0.075, 0.983, -0.167]])

T_list[22] = np.array([[-0.075, -0.667, -0.742],
                       [-0.167, 0.742, -0.650],
                       [0.983, 0.075, -0.167]])

T_list[23] = np.array([[-0.742, 0.667, -0.075],
                       [0.650, 0.742, 0.167],
                       [0.167, 0.075, -0.983]])
#global variables

class Packet():

    def __init__(self,points_data):

        self.points_data = points_data
        self.boundary = None
        self.chosen_nidx = 0
        self.variants = None
        self.orientations = None
        self.pag_ori = None

    def __len__(self):

        return self.points_data.shape[0]

    def __getitem__(self, item):

        if item == 'id':

            return list(set(self.points_data['packet_id']))[0]

    def dis_to_id(self,dis, t_list):

        idx = np.where(t_list > dis)
        if len(idx[0]) == 0:
            return str(len(t_list))
        else:
            return str(int(idx[0][0] - 1))

    def gen_blocks(self, t_mu, sigma, lower=None, upper=None):

        points_data = self.points_data.copy()

        block_plane = np.random.random((1, 3))
        # boundary plane of block within packet

        pd = -(points_data['x'] * block_plane[..., 0] + points_data['y'] * block_plane[..., 1] + points_data['z'] *
               block_plane[..., 2])
        # compute d of block boundary

        points_data.insert(6, 'pd', value=pd)
        sq = np.sqrt(block_plane[..., 0] ** 2 + block_plane[..., 1] ** 2 + block_plane[..., 2] ** 2)
        p_dis = (points_data['pd'].max() - points_data['pd']) / sq
        points_data.insert(7, 'p_dis', value=p_dis)

        n = points_data['p_dis'].max() / t_mu
        n = math.ceil(n)

        if lower == None:
            lower = - np.inf

        if upper == None:
            upper = points_data['p_dis'].max()

        if sigma == 0:
            sigma = 0.1

        LN = lognorm(s=sigma,scale=t_mu)
        bt_list = LN.rvs(n)
        out_lrange = bt_list[bt_list <= lower]
        out_hrange = bt_list[bt_list >= upper]
        bt_list = np.delete(bt_list,bt_list <= lower)
        bt_list = np.delete(bt_list,bt_list >= upper)

        if len(bt_list) > 0:
            increasing = (np.sum(out_hrange) + np.sum(out_lrange))/len(bt_list)
            bt_list = bt_list + increasing

        else:
            bt_list = [t_mu]

        dis_list = [0 for i in range(len(bt_list)+1)]
        for i in range(len(bt_list)):
            dis_list[i+1] = np.sum(bt_list[:i+1])

        dis_list = np.array(dis_list)
        block_id = points_data['p_dis'].map(lambda dis: self.dis_to_id(dis, dis_list))
        points_data['block_id'] = points_data['packet_id'] + block_id
        points_data = points_data.drop('pd', axis=1)

        self.points_data = points_data
        if len(self.points_data) > 10:
            self.merge_small_block()

    def strip_pid(self,bid):

        return(int(bid[len(self['id']):]))

    def merge_small_block(self):

        while (self.points_data.groupby('block_id').apply(len) < 10).any():
            old_ids = list(set(self.points_data['block_id']))
            bid = self.points_data['block_id'].map(lambda bid: self.strip_pid(bid))
            self.points_data['strip_block_id'] = bid
            self.points_data.sort_values('strip_block_id',inplace=True)#sort by bid, so that the order of blocks is arranged

            block_groups = self.points_data.groupby('strip_block_id')
            big_blocks = []
            small_blocks = []
            bb_list = []
            for k in block_groups.groups.keys():
                group = block_groups.get_group(k)

                if len(group) < 10:
                    small_blocks.append(k) #get the id of small blocks
                else:
                    big_blocks.append(k)
                    bb_list.append(group['block_id'].iloc[0])

            if len(small_blocks) == 1:
                dis = np.array([abs(small_blocks[0] - big_block) for big_block in big_blocks])
                new_id = self.points_data.loc[self.points_data['strip_block_id'] == big_blocks[np.argmin(dis)], 'block_id'].iloc[0]
                self.points_data.loc[self.points_data['strip_block_id'] == small_blocks[0],'block_id'] = new_id
                break

            new_id_list = []
            old_id_list = []
            for j in range(len(small_blocks)):
                i = 0
                old_id = self.points_data.loc[self.points_data['strip_block_id'] == small_blocks[j],'block_id'].iloc[0]
                temp = small_blocks[j]
                while small_blocks[j]+i in small_blocks:

                    temp = small_blocks[j] + i
                    i += 1
                small_blocks[j] = temp

                if i == 1: # no adjacent block is small one

                    if small_blocks[j] == small_blocks[j-1]:
                        new_id = old_id
                    else:
                        if len(big_blocks) == 0:
                            dis = np.array([abs(small_blocks[j] - small_block) for small_block in small_blocks if small_block != small_blocks[j]])
                            new_id = self.points_data.loc[self.points_data['strip_block_id'] == small_blocks[np.argmin(dis)], 'block_id'].iloc[0]
                        else:
                            dis = np.array([abs(small_blocks[j] - big_block) for big_block in big_blocks])
                            new_id = self.points_data.loc[self.points_data['strip_block_id'] == big_blocks[np.argmin(dis)],'block_id'].iloc[0]

                else: #adjacent block is small one
                    new_id = self.points_data.loc[self.points_data['strip_block_id'] == temp, 'block_id'].iloc[0]  # merge adjacent small blocks into one block

                old_id_list.append(old_id)
                new_id_list.append(new_id)


            old_id_list.extend(bb_list)
            new_id_list.extend(bb_list)

            old_to_new = dict(zip(old_id_list,new_id_list))

            new_bid = self.points_data['block_id'].map(lambda bid:old_to_new[bid])
            self.points_data['block_id'] = new_bid
            self.points_data.drop('strip_block_id',axis=1)

            new_ids = list(set(self.points_data['block_id']))
            if new_ids == old_ids:
                print('cannot merge blocks anymore')
                break

        print('all small blocks in packet {} are merged'.format(self['id']))

    def get_bt(self):

        self.points_data['block_thickness'] = np.nan
        bg = self.points_data.groupby('block_id')
        bid_to_bt = bg['p_dis'].apply(max)-bg['p_dis'].apply(min)
        self.points_data['block_thickness'] = self.points_data.apply(lambda p:bid_to_bt[p['block_id']],axis=1)

        return self.points_data

    def assign_bv(self):

        points_data = self.points_data.copy()
        points_data.sort_values(by='p_dis',inplace=True)
        bid = points_data['block_id'].map(lambda bid: self.strip_pid(bid))
        points_data['strip_bid'] = bid
        bid = list(set(bid))

        vidx = [1 for i in range(len(bid))]

        for i in range(len(bid)):
            pv = [0, 1, 2, 3, 4, 5]
            if i >= 1:
                pv.remove(vidx[i-1])
            vidx[i] = np.random.choice(pv,1)[0]

        variant_trial_list = self.variants
        bid_to_vidx = dict(zip(bid,vidx))

        points_data['block_variant'] = points_data.apply(lambda p:variant_trial_list[...,bid_to_vidx[p['strip_bid']]][0],axis=1)
        self.points_data = points_data
        return self.points_data

    def assign_block_ori(self):

        points_data = self.points_data.copy()
        points_data[['phi1','PHI','phi2']] = None
        block_ids = list(set(points_data['block_id']))

        chosen_ori = [i for i in range(len(block_ids))]

        if len(self.orientations) > 1:
            for i in range(len(block_ids)):
                bori = self.orientations.copy()
                if i >= 1:
                    try:
                        bori.drop(chosen_ori[i-1].index,inplace=True)
                    except:
                        bori.drop(bori.head(1).index,inplace=True)

                chosen_ori[i] = bori.sample(1,axis=0)

            bid_to_ori = dict(zip(block_ids, chosen_ori))
            points_data['phi1'] = points_data['block_id'].map(lambda bid: bid_to_ori[bid]['phi1'].values[0])
            points_data['PHI'] = points_data['block_id'].map(lambda bid: bid_to_ori[bid]['PHI'].values[0])
            points_data['phi2'] = points_data['block_id'].map(lambda bid: bid_to_ori[bid]['phi2'].values[0])

        else:
            chosen_idx = np.random.randint(6)
            assigned_idx_list = []
            assigned_ori = []
            for i in range(len(block_ids)):
                trial_idx = [0,1,2,3,4,5]
                if i >= 1:
                    trial_idx.remove(assigned_idx_list[i-1])
                assigned_idx = np.random.choice(trial_idx,1)[0]
                assigned_idx_list.append(assigned_idx)
                phi1,PHI,phi2 = self.comp_ori(chosen_idx,assigned_idx)
                ori = [phi1,PHI,phi2]
                assigned_ori.append(ori)

            bid_to_ori = dict(zip(block_ids, assigned_ori))
            points_data['phi1'] = points_data['block_id'].map(lambda bid: bid_to_ori[bid][0])
            points_data['PHI'] = points_data['block_id'].map(lambda bid: bid_to_ori[bid][1])
            points_data['phi2'] = points_data['block_id'].map(lambda bid: bid_to_ori[bid][2])

        self.points_data = points_data
        return self.points_data

    def comp_ori(self,chosen_idx,assigned_idx):

        phi1 = self.pag_ori[0]
        PHI = self.pag_ori[1]
        phi2 = self.pag_ori[2]

        R1 = np.array([[np.cos(np.deg2rad(phi1)), -np.sin(np.deg2rad(phi1)), 0],
                       [np.sin(np.deg2rad(phi1)), np.cos(np.deg2rad(phi1)), 0],
                       [0, 0, 1]])

        R2 = np.array([[1, 0, 0],
                       [0, np.cos(np.deg2rad(PHI)), -np.sin(np.deg2rad(PHI))],
                       [0, np.sin(np.deg2rad(PHI)), np.cos(np.deg2rad(PHI))]])

        R3 = np.array([[np.cos(np.deg2rad(phi2)), -np.sin(np.deg2rad(phi2)), 0],
                       [np.sin(np.deg2rad(phi2)), np.cos(np.deg2rad(phi2)), 0],
                       [0, 0, 1]])

        result = np.dot(R3, R2)
        R = np.matrix(np.dot(result, R1))
        trial_variants = self.variants

        t1_idx = int(trial_variants[0, chosen_idx][-1])
        T1 = np.matrix(T_list[t1_idx])
        g = T1.I * R

        assigned_v = trial_variants[0, assigned_idx]

        t_idx = int(assigned_v[-1])
        T = T_list[t_idx]
        R = T * g

        N, n, n1, n2 = 1, 1, 1, 1

        if R[2, 2] > 1:
            N = 1 / R[2, 2]

        if R[2, 2] < -1:
            N = -1 / R[2, 2]

        R[2, 2] = N * R[2, 2]
        PHIB = np.degrees(np.arccos(R[2, 2]))
        sin_PHIB = np.sin(np.deg2rad(PHIB))

        if R[2, 0] / sin_PHIB > 1 or R[2, 0] / sin_PHIB < -1:
            n1 = sin_PHIB / R[2, 0]

        if R[0, 2] / sin_PHIB > 1 or R[0, 2] / sin_PHIB < -1:
            n2 = sin_PHIB / R[0, 2]

        if abs(n1) > abs(n2):

            n = n2

        else:

            n = n1

        # recalculate after scaling
        RB = N * n * R
        PHIB = np.degrees(np.arccos(RB[2, 2]))
        if PHIB < 0:
            PHIB = PHIB + 360
        sin_PHIB = np.sin(np.deg2rad(PHIB))
        phi1B = np.degrees(np.arcsin(RB[2, 0] / sin_PHIB))
        if phi1B < 0:
            phi1B = phi1B + 360
        phi2B = np.degrees(np.arcsin(RB[0, 2] / sin_PHIB))
        if phi2B < 0:
            phi2B = phi2B + 360

        return phi1B, PHIB, phi2B

    def comp_angle(self, pag_ori, point_data):

        i = int(str(point_data['block_variant']).lstrip('V')) - 1
        T = T_list[i]
        phi1 = pag_ori[0]
        PHI = pag_ori[1]
        phi2 = pag_ori[2]

        R1 = np.array([[np.cos(np.deg2rad(phi1)), -np.sin(np.deg2rad(phi1)), 0],
                       [np.sin(np.deg2rad(phi1)), np.cos(np.deg2rad(phi1)), 0],
                       [0, 0, 1]])

        R2 = np.array([[1, 0, 0],
                       [0, np.cos(np.deg2rad(PHI)), -np.sin(np.deg2rad(PHI))],
                       [0, np.sin(np.deg2rad(PHI)), np.cos(np.deg2rad(PHI))]])

        R3 = np.array([[np.cos(np.deg2rad(phi2)), -np.sin(np.deg2rad(phi2)), 0],
                       [np.sin(np.deg2rad(phi2)), np.cos(np.deg2rad(phi2)), 0],
                       [0, 0, 1]])

        result = np.dot(R3, R2)
        R = np.matrix(np.dot(result, R1))

        RB = T * R
        N, n, n1, n2 = 1, 1, 1, 1

        if RB[2, 2] > 1:
            N = 1 / RB[2, 2]

        if RB[2, 2] < -1:
            N = -1 / RB[2, 2]

        RB[2, 2] = N * RB[2, 2]
        PHIB = np.degrees(np.arccos(RB[2, 2]))
        sin_PHIB = np.sin(np.deg2rad(PHIB))

        if RB[2, 0] / sin_PHIB > 1 or RB[2, 0] / sin_PHIB < -1:
            n1 = sin_PHIB / RB[2, 0]

        if RB[0, 2] / sin_PHIB > 1 or RB[0, 2] / sin_PHIB < -1:
            n2 = sin_PHIB / RB[0, 2]

        if abs(n1) > abs(n2):

            n = n2

        else:

            n = n1

        # recalculate after scaling
        RB = N * n * RB
        PHIB = np.degrees(np.arccos(RB[2, 2]))
        if PHIB < 0:
            PHIB = PHIB + 360
        sin_PHIB = np.sin(np.deg2rad(PHIB))
        phi1B = np.degrees(np.arcsin(RB[2, 0] / sin_PHIB))
        if phi1B < 0:
            phi1B = phi1B + 360
        phi2B = np.degrees(np.arcsin(RB[0, 2] / sin_PHIB))
        if phi2B < 0:
            phi2B = phi2B + 360

        return phi1B, PHIB, phi2B

if __name__ == '__main__':

    grains_data = pd.read_csv('F:/pycharm/2nd_mini_thesis/dragen-master/dragen/OutputData/2021-06-10_0/Generation_Data/grain_data_output_discrete.csv')

    grain_data = grains_data.iloc[1]
    orientation = (grain_data['phi1'],grain_data['PHI'],grain_data['phi2'])
    grain = Grain(grain_data['final_conti_volume'],grainID=grain_data['GrainID'],phaseID=grain_data['phaseID'],
                          alpha=grain_data['alpha'],orientation=orientation)

    grain.gen_subs(6,0.1,0.5,0.1)
    print(grain.points_data)






























