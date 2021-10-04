# _*_ coding: utf-8 _*_
"""
Time:     2021/6/18 0:29
Author:   Linghao Kong
Version:  V 0.1
File:     run
Describe: Write during the internship at IEHK RWTH"""
import pandas as pd
from substructure.substructure import plot_rve_subs
import numpy as np
from substructure.data import save_data
import logging
import logging.handlers
import os
from substructure.substructure import Grain

class Run():

    def __init__(self,box_size, n_pts,equiv_d=None, p_sigma=None, t_mu=None, b_sigma=0.001, decreasing_factor=0.95,
                 lower=None, upper=None, circularity=1, plt_name=None, save=True, plot=False,
                 filename=None, fig_path=None, gen_flag='user_define',block_file=None, OR='KS'):

        self.box_size = box_size
        self.n_pts = n_pts
        self.equiv_d = equiv_d
        self.p_sigma = p_sigma
        self.t_mu = t_mu
        self.b_sigma = b_sigma
        self.decreasing_factor = decreasing_factor
        self.lower = lower
        self.upper = upper
        self.circularity = circularity
        self.plt_name = plt_name
        self.save = save
        self.plot = plot
        self.filename = filename
        self.fig_path = fig_path
        self.OR = OR
        self.block_file = block_file
        self.gen_flag = gen_flag

    def get_orientations(self,block_df, grainID):
        blocks = block_df[block_df['grain_id'] == grainID+1] # +1...
        n_pack = len(list(set(blocks['packet_id'])))
        groups = blocks.groupby('packet_id')

        ori_dflist = []
        for name, group in groups:
            ori_df = group[['phi1', 'PHI', 'phi2']]
            ori_dflist.append(ori_df)

        n_pack = np.arange(n_pack)

        n_pack_to_ori = dict(zip(n_pack, ori_dflist))

        return n_pack_to_ori

    def get_bt_distribution(self,block_df):

        average_bt = block_df['block_thickness'].mean()
        return average_bt

    def run(self,rve_df,grains_df,store_path,logger):

        logger.info('------------------------------------------------------------------------------')
        logger.info('substructure generation begins')
        logger.info('------------------------------------------------------------------------------')
        rve_data = pd.DataFrame()
        if self.gen_flag == 'from_file':
            assert self.block_file is not None
            block_df = pd.read_csv(self.block_file)
            average_bt = self.get_bt_distribution(block_df)
            average_bt = self.decreasing_factor * average_bt

        for i in range(len(grains_df)):

            grain_data = grains_df.iloc[i]
            phaseID = int(grain_data['phaseID'])
            grain_id = grain_data['GrainID']
            x = rve_df[rve_df['GrainID'] == grain_id+1]['x'].to_numpy().reshape((-1, 1)) #needs + 1 finally
            y = rve_df[rve_df['GrainID'] == grain_id+1]['y'].to_numpy().reshape((-1, 1))
            z = rve_df[rve_df['GrainID'] == grain_id+1]['z'].to_numpy().reshape((-1, 1))

            points = np.concatenate((x, y, z), axis=1)

            if phaseID == 2:

                orientation = (grain_data['phi1'], grain_data['PHI'], grain_data['phi2'])
                grain = Grain(v=grain_data['final_conti_volume'], points=points,
                              phaseID=phaseID,grainID=grain_id,orientation=orientation)

                if self.gen_flag == 'from_file':
                    old_gid = grain_data['old_gid']
                    blocks = block_df[block_df['grain_id'] == old_gid + 1]
                    n_pack = len(list(set(blocks['packet_id'])))
                    orientations = self.get_orientations(block_df, old_gid)
                    grain.gen_subs(block_thickness=average_bt, b_sigma=self.b_sigma, lower_t=self.lower, upper_t=self.upper,
                                   circularity=self.circularity,
                                   n_pack=n_pack, orientations=orientations)

                if self.gen_flag == 'user_define':
                    assert self.equiv_d is not None
                    assert self.p_sigma is not None
                    assert self.t_mu is not None
                    grain.gen_subs(self.equiv_d,sigma=self.p_sigma,block_thickness=self.t_mu,b_sigma=self.b_sigma,lower_t=self.lower,
                                   upper_t=self.upper,circularity=self.circularity)
                rve_data = pd.concat([rve_data,grain.points_data])

            else:

                grain_data = pd.DataFrame(points,columns=['x','y','z'])
                grain_data['GrainID'] = grain_id
                grain_data['phaseID'] = phaseID
                grain_data['packet_id'] = grain_id
                grain_data['block_id'] = grain_id
                grain_data['block_orientation'] = np.NaN
                rve_data = pd.concat([rve_data, grain_data])

        #transfer id to number
        rve_data.loc[rve_data['block_id'].isnull(), 'block_id'] = rve_data[rve_data['block_id'].isnull()]['packet_id'] + '0'
        packet_id = list(set(rve_data['packet_id']))
        n_id = np.arange(1,len(packet_id) + 1)
        pid_to_nid = dict(zip(packet_id, n_id))
        pid_in_rve = rve_data['packet_id'].map(lambda pid: pid_to_nid[pid])

        block_id = list(set(rve_data['block_id']))
        n2_id = np.arange(1,len(block_id) + 1)
        bid_to_nid = dict(zip(block_id, n2_id))
        bid_in_rve = rve_data['block_id'].map(lambda bid: bid_to_nid[bid])

        rve_data['packet_id'] = pid_in_rve
        rve_data['block_id'] = bid_in_rve
        rve_data['n_pts'] = self.n_pts
        rve_data['box_size'] = self.box_size

        if self.save:

            if self.filename:

                save_data(rve_data,store_path,self.filename)

            else:
                save_data(rve_data,store_path)

        if self.plot:

            for name in self.plt_name:

                plot_rve_subs(rve_data,name,self.fig_path)

        logger.info('substructure generation successful')
        logger.info('------------------------------------------------------------------------------')

        return rve_data

if __name__ == '__main__':

    import datetime
    start = datetime.datetime.now()
    test_run = Run(20,40,'F:/pycharm/2nd_mini_thesis/substructure')
    rve_data = pd.read_csv('F:/pycharm/2nd_mini_thesis/dragen-master/OutputData/2021-07-23_0/substruct_data_abq.csv') #changes into rve data
    grains_df = pd.read_csv('F:/pycharm/2nd_mini_thesis/dragen-master/OutputData/2021-07-23_0/Generation_Data/grain_data_input.csv')#grain_data
    test_run.run()
    end = datetime.datetime.now()
    print('running time is',end-start)