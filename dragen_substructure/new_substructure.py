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
from scipy.stats import lognorm,truncnorm
from substructure.data import save_data
import math

#block_id needs modification
#using magic method
class Grain(RVEUtils):

    def __init__(self,box_size,n_pts,v,a = 2,b = 2,c = 3,x_0 = 0,y_0 = 0,z_0 = 0,grainID = 1,phaseID = 2,orientation = None,alpha = 0,
                 points = None,OR = 'KS'):

        self.box_size = box_size
        self.n_pts = n_pts
        self.volume = v
        self.a = a
        self.b = b
        self.c = c
        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0
        self.alpha = alpha
        self.points = points
        self.orientation = orientation
        self.grainID = int(grainID)
        self.phaseID = int(phaseID)
        self.packets_list = []

        if isinstance(points,np.ndarray):

            self.points_data = pd.DataFrame(points,columns=['x','y','z'])
            self.points_data['GrainID'] = grainID
            self.points_data['phaseID'] = phaseID

        else:

            points = self.points_gen()
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

        super().__init__(box_size,n_pts,debug=False)

    def points_gen(self):

        x,y,z = super().gen_grid()
        ellip = super().ellipsoid(self.a,self.b,self.c,self.x_0,self.y_0,self.z_0,self.alpha)
        t = (ellip <= 1)

        X = []
        Y = []
        Z = []

        for px in np.nditer(t * x):
            X.append(float(px))
        for py in np.nditer(t * y):
            Y.append(float(py))
        for pz in np.nditer(t * z):
            Z.append(float(pz))

        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        Z = np.array(Z).reshape(-1, 1)

        points_list = np.concatenate([X, Y, Z], axis=1)
        points_list = np.unique(points_list, axis=0)

        self.points = points_list
        self.points_data = pd.DataFrame(points_list,columns=['x','y','z'])

        return points_list

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

    def gen_subs(self, equiv_d, sigma, block_thickness, b_sigma, lower_t=None, upper_t=None, circularity=1):

        r = equiv_d / 2
        average_pv = 4 / 3 * np.pi * r ** 3 * circularity ** (1.5)
        av_pn = round(self.volume / average_pv)
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
        self.points_data['block_orientation'] = None

        i = 1
        for num in num_grid_ps:

            trial_index = [0, 1, 2, 3]
            if i > 1:
                trial_index.remove(self.packets_list[i-2].chosen_nidx)
                index = np.random.choice(trial_index,1)[0]

            else:
                index = np.random.randint(4)

            chosen_norm = self.hp_normal_list[index, ...]

            passing_point = points_data.sample(1, axis=0)
            d = -(passing_point['x'].values * chosen_norm[..., 0] + passing_point['y'].values * chosen_norm[..., 0] +
                  passing_point['z'].values * chosen_norm[..., 0])
            values = points_data['x'] * chosen_norm[..., 0][0, 0] + points_data['y'] * chosen_norm[..., 0][0, 0] + \
                     points_data['z'] * chosen_norm[..., 0][0, 0] + d[0, 0]

            values.sort_values()
            chosen_index = values.iloc[0:num].index

            # self.points_data.loc[chosen_index,'packet_id'] = str(int(self.points_data.loc[chosen_index,'GrainID'].iloc[0])) + 'p' + str(i)
            points_data.loc[chosen_index, 'packet_id'] = str(
                int(points_data.loc[chosen_index, 'GrainID'].iloc[0])) + 'p' + str(i)

            packet_df = points_data.loc[chosen_index]
            packet = Packet(packet_df)
            packet.chosen_nidx = index
            packet.boundary = chosen_norm
            packet.gen_blocks(t_mu=block_thickness, sigma=b_sigma, lower=lower_t,upper=upper_t)  # complete df
            packet.get_bt()
            comp_df = packet.assign_bv(self)

            self.points_data.loc[comp_df.index, 'packet_id'] = packet['id']
            self.points_data.loc[comp_df.index, 'block_id'] = comp_df['block_id']
            self.points_data.loc[comp_df.index, 'block_thickness'] = comp_df['block_thickness']
            self.points_data.loc[comp_df.index, 'block_orientation'] = comp_df['block_orientation']
            self.packets_list.append(packet)

            points_data.drop(packet_df.index, inplace=True)

            i += 1

        #self.points_data.drop(self.points_data[self.points_data['block_thickness'] == 0].index,inplace=True)
        return self.points_data

    def trial_variants_select(self, normal):

        variants = self.variants
        h_list = self.hp_normal_list
        hp_list = []
        for h in h_list:
            hp_list.append(list(np.array(h).squeeze()))
        hp_list = np.array(hp_list)

        possible_v = variants[np.where((hp_list == normal).all(axis = 1))[0],:]
        trial_iter1 = np.random.randint(3)
        trial_iter2 = trial_iter1 + 3
        variant_trial_list = np.array([possible_v[..., trial_iter1], possible_v[..., trial_iter2]])

        return variant_trial_list

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

def rve_substruct_plotter(rve_data,subs_name,store_path = None):

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

class Packet():

    def __init__(self,points_data):

        self.points_data = points_data
        self.neighbors_list = []
        self.boundary = None
        self.chosen_nidx = 0

    def __len__(self):

        return self.points_data.shape[0]

    def __getitem__(self, item):

        if item == 'id':

            return list(set(self.points_data['packet_id']))[0]

    def dis_to_id(self,dis, t_list):

        t_1 = []
        t_list = list(t_list)
        if len(t_list) == 1:
            t_list.insert(0, 0)
            t_1 = t_list

            if dis >= max(t_1):

                return '1'

            else:

                return '0'

        else:
            for i in range(len(t_list) - 1):
                s = sum(t_list[0:i + 1])
                t_1.append(s)

            t_1.insert(0, 0)

        for i in range(len(t_1) - 1):
            if dis >= t_1[i] and dis < t_1[i + 1]:
                return str(i)

            if dis > max(t_1):
                return str(len(t_1))

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

        bt_list = []
        if n == 1:
            bt_list = [t_mu]

        if lower == None:
            lower = - np.inf

        if upper == None:
            intervals = lognorm.interval(0.9, s=sigma, scale=t_mu)
            upper = np.log(intervals[1])

        if sigma > 1:
            sigma = np.log(sigma)

        else:
            sigma = 0.01

        N = truncnorm((lower - t_mu) / sigma, (upper - np.log(t_mu)) / sigma, loc=np.log(t_mu), scale=sigma)
        bt_list = np.exp(N.rvs(n))

        block_id = points_data['p_dis'].map(lambda dis: self.dis_to_id(dis, bt_list))
        points_data['block_id'] = points_data['packet_id'] + block_id
        points_data = points_data.drop('pd', axis=1)

        points_data.dropna(axis=0, inplace=True, how='any')

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

    def assign_bv(self,grain):

        p_iter1 = np.random.randint(2)
        p_iter2 = p_iter1 - 1

        variant_trial_list = grain.trial_variants_select(self.boundary)

        pb = self.points_data.copy()

        bid = str(pb.iloc[0]['block_id'])
        vi = int(bid.replace('p', '')) % 2
        V1 = variant_trial_list[p_iter1]
        V2 = variant_trial_list[p_iter2]
        p = pb['block_id'].map(lambda bi: float(str(bi).replace('p', '')))
        pb['block_orientation'] = np.nan
        pb.loc[p % 2 == vi, 'block_orientation'] = V1[0]
        pb.loc[p % 2 != vi, 'block_orientation'] = V2[0]
        self.points_data = pb

        return self.points_data

if __name__ == '__main__':

    grains_data = pd.read_csv('F:/pycharm/2nd_mini_thesis/dragen-master/dragen/OutputData/2021-06-10_0/Generation_Data/grain_data_output_discrete.csv')

    for i in range(50):
        grain_data = grains_data.iloc[i]

        orientation = (grain_data['phi1'],grain_data['PHI'],grain_data['phi2'])
        grain = Grain(20,40,grain_data['final_conti_volume'],grain_data['a'],grain_data['b'],grain_data['c'],grainID=grain_data['GrainID'],phaseID=grain_data['phaseID'],
                          alpha=grain_data['alpha'],orientation=orientation)

        grain.gen_subs(6,0.1,1,0.1)

        print(i)
    # grain.block_orientation_assignment()


    # grain.substruct_plotter('packet')
    # grain.substruct_plotter('block')























