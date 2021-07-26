# _*_ coding: utf-8 _*_
"""
Time:     2021/6/17 22:55
Author:   Linghao Kong
Version:  V 0.1
File:     rve_substructure_old.py
Describe: Write during the internship at IEHK RWTH"""

from dragen.utilities.RVE_Utils import RVEUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from dragen_substructure.data import save_data
import math
import scipy.stats as stats
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
        self.packets_list = None
        self.id_to_cut = {}

        if isinstance(points,np.ndarray):

            self.points_data = pd.DataFrame(points,columns=['x','y','z'])
            self.points_data['GrainID'] = grainID
            self.points_data['phaseID'] = phaseID

        else:

            points = self.points_gen()
            self.points_data = pd.DataFrame(points,columns=['x','y','z'])
            self.points_data['GrainID'] = grainID
            self.points_data['phaseID'] = phaseID

        self.plane_list = None

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

    # points_gen is only for testing within 1 grain
    # def points_gen(self):
    #
    #     x,y,z = super().gen_grid()
    #     ellip = super().ellipsoid(self.a,self.b,self.c,self.x_0,self.y_0,self.z_0,self.alpha)
    #     t = (ellip <= 1)
    #
    #     X = []
    #     Y = []
    #     Z = []
    #
    #     for px in np.nditer(t * x):
    #         X.append(float(px))
    #     for py in np.nditer(t * y):
    #         Y.append(float(py))
    #     for pz in np.nditer(t * z):
    #         Z.append(float(pz))
    #
    #     X = np.array(X).reshape(-1, 1)
    #     Y = np.array(Y).reshape(-1, 1)
    #     Z = np.array(Z).reshape(-1, 1)
    #
    #     points_list = np.concatenate([X, Y, Z], axis=1)
    #     points_list = np.unique(points_list, axis=0)
    #
    #     self.points = points_list
    #     self.points_data = pd.DataFrame(points_list,columns=['x','y','z'])
    #
    #     return points_list

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

    def insert_plane(self, num_habit_plane):

        points = self.points

        # if num_habit_plane == 0:
        #
        #     packet_id = np.array([str(self.grainID)]).repeat(len(points))
        #     self.points_data['packet_id'] = packet_id
        #     plane_list = self.plane_list
        #
        #     return packet_id,plane_list

        selected_iter = np.random.choice(len(points),size=num_habit_plane,replace=False)
        selected_points_list = np.array([points[i] for i in selected_iter])

        habit_plane_list = self.hp_normal_list
        h_iter = np.random.randint(4, size=num_habit_plane)
        selected_habit_plane = np.array([habit_plane_list[i] for i in h_iter]).astype(float)

        if selected_habit_plane.ndim != 2:
            selected_habit_plane = selected_habit_plane.squeeze()

        if num_habit_plane == 1:
            selected_habit_plane  = selected_habit_plane.reshape((-1,3))

        d_list = np.sum(-selected_points_list * selected_habit_plane, axis=1)

        plane_list = np.insert(selected_habit_plane, 3, values=d_list, axis=1)

        ones = np.ones(len(points))
        biased_points = np.insert(points, 3, values=ones, axis=1)

        result = biased_points @ plane_list.T
        result[result >= 0] = 1
        result[result < 0] = 0
        result = (result.astype(int)).astype(str)

        c = result.shape[1]

        packet_id = result[..., 0]
        for i in range(1, c):
            packet_id = np.char.add(packet_id, result[..., i])

        self.points_data['packet_id'] = packet_id
        self.points_data['packet_id'] = str(self.phaseID) + str(self.grainID) + 'p' + self.points_data['packet_id']
        self.plane_list = plane_list

        points_data = self.points_data.copy()
        groups = points_data.groupby('packet_id')
        kl = list(groups.groups.keys())
        self.packets_list = [Packet(groups.get_group(pid)) for pid in kl]

        return packet_id,plane_list

    def int_solution(self,k):

        n = 1
        while (n ** 3 + 5 * n + 6) / 6 < k - 2:
            n += 1

        return n

    def gen_packets(self, equiv_d):

        r = equiv_d / 2
        Vp = 4 / 3 * np.pi * r ** 3
        k = int(self.volume / Vp)

        if k <= 1:

            m = np.random.randint(1,4)
            return self.insert_plane(m)  # modified needed

        n = self.int_solution(k)

        self.insert_plane(n)  # these are the packet_id and plane_list at first round, in other words, before the second cut process

        packets_list = self.packets_list.copy()

        n_packet = len(packets_list)

        while n_packet < k:

            sorted_packets_list = sorted(packets_list, key=lambda i: len(i), reverse=True)
            self.packets_list = sorted_packets_list  # change together
            diff_n = k - n_packet

            if diff_n <= n_packet:

                count = diff_n

            else:

                count = n_packet

            for i in range(count):
                packet1, packet2, cut_plane = sorted_packets_list[0].cut(
                    self.hp_normal_list)  # after deleting i_th packet always becomes the first one ......
                self.points_data.loc[sorted_packets_list[0].points_data.index, 'packet_id'] = \
                sorted_packets_list[0].points_data['packet_id']
                self.id_to_cut[packet1['id']] = cut_plane
                self.id_to_cut[packet2['id']] = cut_plane
                del (self.packets_list[0])
                self.packets_list.extend([packet1, packet2])
                self.second_cut_list.append(cut_plane)

            n_packet = len(self.packets_list)
            packets_list = self.packets_list

        self.second_cut_list = np.array(self.second_cut_list).squeeze()

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

        self.points_data['block_id'] = np.nan
        points_data = self.points_data.copy()

        for packet in self.packets_list:

            packet.gen_blocks(self,t_mu,sigma,lower=lower,upper=upper)
            points_data.loc[packet.points_data.index,'block_id'] = packet.points_data['block_id']

        self.points_data = points_data

    def closet_plane(self, packet):

        p1 = packet.points_data.iloc[0]
        planes_list = []
        plane_list = self.plane_list.copy()

        pid = packet['id'].lstrip(str(self.phaseID))
        pid = pid.lstrip(str(self.grainID))

        if pid.count('2') > 0:

            plane_list = np.vstack((plane_list,self.id_to_cut[packet['id']]))

        for plane in plane_list:
            pl = Plane(plane[..., 0:3], plane[..., 3])
            planes_list.append(pl) #convert into plane object

        sorted_pl = sorted(planes_list,key=lambda pl: abs(p1['x'] + (pl['b'] * p1['y'] + pl['c'] * p1['z'] + pl['d']) / pl['a'])
                           if pl['a'] != 0 else np.inf) #sorted according to distance

        return sorted_pl[0] #return the closet plane

    def trial_variants_select(self, normal):

        variants = self.variants
        h_list = self.hp_normal_list
        hp_list = []
        for h in h_list:
            hp_list.append(list(np.array(h).squeeze()))
        hp_list = np.array(hp_list)

        possible_v = variants[np.where((hp_list == normal).all(axis = 1))]

        trial_iter1 = np.random.randint(3)
        trial_iter2 = trial_iter1 + 3
        variant_trial_list = np.array([possible_v[..., trial_iter1], possible_v[..., trial_iter2]])

        return variant_trial_list

    def block_orientation_assignment(self):

        points_data = self.points_data.copy()
        points_data['block_orientation'] = np.nan

        for packet in self.packets_list:

            if len(packet.points_data) != 0:
                packet.assign_bv(self)
                points_data.loc[packet.points_data.index, 'block_orientation'] = packet.points_data['block_orientation']

        self.points_data = points_data

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

    def __len__(self):

        return self.points_data.shape[0]

    def __getitem__(self, item):

        if item == 'id':

            return list(set(self.points_data['packet_id']))[0]

    def id_assign(self,point_data,cut_plane,d):

        if point_data['x']*cut_plane[...,0] + point_data['y']*cut_plane[...,1] + point_data['z']*cut_plane[...,2] + d >= 0:

            return '1'

        else:

            return '0'

    def cut(self,hp_normal_list):

        warnings.filterwarnings('ignore')
        points_data = self.points_data

        old_id = points_data['packet_id'].to_numpy().astype(str)
        mark_s = np.array(list('2' * old_id.shape[0]))
        old_id = np.char.add(old_id, mark_s)

        h_iter = np.random.randint(4, size=1)[0]
        cut_plane = np.array(hp_normal_list[h_iter]).astype(float)
        cut_point = points_data.iloc[np.random.randint(len(points_data),size=1)]

        d = -(cut_point['x']*cut_plane[...,0] + cut_point['y']*cut_plane[...,1] + cut_point['z']*cut_plane[...,2]).values

        new_id = points_data.apply(lambda p:self.id_assign(p,cut_plane,d),axis=1)
        points_data['packet_id'] = np.char.add(old_id,new_id)

        #with new_id, 2 new packets generated within original one
        groups = points_data.groupby('packet_id')

        try:
            p_id1 = list(groups.groups.keys())[0]
            p_id2 = list(groups.groups.keys())[1]

            points1 = groups.get_group(p_id1)
            points2 = groups.get_group(p_id2)

        except:

            print('the size of packet is too small')

            return

        cut_plane = np.insert(cut_plane,3,d,axis=1)

        return Packet(points1),Packet(points2),cut_plane #return error sometimes

    def gen_blocks(self,grain,t_mu,sigma,lower=None,upper=None):

        points_data = self.points_data.copy()

        block_plane = np.random.random((1, 3))
        # boundary plane of block within packet

        pd = -(points_data['x'] * block_plane[...,0] + points_data['y'] * block_plane[...,1] + points_data['z'] * block_plane[...,2])
        # compute d of block boundary

        points_data.insert(6, 'pd', value=pd)
        sq = np.sqrt(block_plane[...,0] ** 2 + block_plane[...,1] ** 2 + block_plane[...,2] ** 2)
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
            intervals = stats.lognorm.interval(0.9, s=sigma, scale=t_mu)
            upper = np.log(intervals[1])

        if sigma > 1:
            sigma = np.log(sigma)

        else:
            sigma = 0.01

        N = stats.truncnorm((lower - t_mu) / sigma, (upper - np.log(t_mu)) / sigma, loc=np.log(t_mu), scale=sigma)
        bt_list = np.exp(N.rvs(n))

        block_id = points_data['p_dis'].map(lambda dis: grain.dis_to_id(dis, bt_list))

        points_data['block_id'] = points_data['packet_id'] + block_id
        points_data = points_data.drop(['pd', 'p_dis'], axis=1)

        points_data.dropna(axis=0,inplace=True,how='any')
        self.points_data = points_data

    def assign_bv(self,grain):

        p_iter1 = np.random.randint(2)
        p_iter2 = p_iter1 - 1

        pl = grain.closet_plane(self)
        variant_trial_list = grain.trial_variants_select(pl.normal)

        pb = self.points_data.copy()

        bid = str(pb.iloc[0]['block_id'])
        vi = int(bid.replace('p','')) % 2
        V1 = variant_trial_list[p_iter1]
        V2 = variant_trial_list[p_iter2]

        p = pb['block_id'].map(lambda bi: float(str(bi).replace('p','')))
        pb['block_orientation'] = np.nan
        pb.loc[p % 2 == vi, 'block_orientation'] = V1[0]
        pb.loc[p % 2 != vi, 'block_orientation'] = V2[0]

        self.points_data = pb

if __name__ == '__main__':

    grains_data = pd.read_csv('F:/pycharm/2nd_mini_thesis/dragen-master/dragen/OutputData/2021-06-10_0/Generation_Data/grain_data_output_discrete.csv')
    grain_data = grains_data.iloc[5]

    orientation = (grain_data['phi1'],grain_data['PHI'],grain_data['phi2'])
    grain = Grain(20,40,grain_data['final_conti_volume'],grain_data['a'],grain_data['b'],grain_data['c'],grainID=grain_data['GrainID'],phaseID=grain_data['phaseID'],
                  alpha=grain_data['alpha'],orientation=orientation)

    grain.gen_packets(5) #give the equivd of packets

    grain.gen_blocks(0.5, 0.61)
    grain.block_orientation_assignment()

    # grain.substruct_plotter('packet')
    # grain.substruct_plotter('block')


    # grains_x, grains_y, grains_z, ic = rve_load(
    #     'F:/pycharm/2nd_mini_thesis/dragen-master/dragen/OutputData/2021-06-10_0/RVE_Numpy.npy', box_size=20, n_pts=40)
    # grains_list = grains_in_rve(grains_x, grains_y, grains_z, ic,
    #                             'F:/pycharm/2nd_mini_thesis/dragen-master/dragen/OutputData/2021-06-10_0/Generation_Data/grain_data_output_conti.csv',equiv_d=6.8)
    #
    # data = grains_list
    # save_data(data,'data.pkl')
    #
    # print('plotting begins')
    # rve_substruct_plotter(grains_list, 'grain','rve_grain.png')
    # rve_substruct_plotter(grains_list, 'packet','rve_packet.png')
    # rve_substruct_plotter(grains_list, 'block','rve_block.png')






















