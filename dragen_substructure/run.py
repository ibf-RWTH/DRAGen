# _*_ coding: utf-8 _*_
"""
Time:     2021/6/18 0:29
Author:   Linghao Kong
Version:  V 0.1
File:     run
Describe: Write during the internship at IEHK RWTH"""
import pandas as pd
from dragen_substructure.substructure import Grain,rve_substruct_plotter
import numpy as np
from dragen_substructure.data import save_data
import logging
import logging.handlers
import os

class Run():

    def __init__(self,box_size,n_pts,root_dir,a = 2,b = 2,c = 3,x_0 = 0,y_0 = 0,z_0 = 0,grainID = None,OR='KS'):

        self.box_size = box_size
        self.n_pts = n_pts
        self.a = a
        self.b = b
        self.c = c
        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0
        self.grainID = grainID
        self.OR = OR
        self.logger = logging.getLogger('RVE-substruct')
        self.root_dir = root_dir

    def setup_logging(self):

        LOGS_DIR = self.root_dir + '/Logs/'
        if not os.path.isdir(LOGS_DIR):
            os.makedirs(LOGS_DIR)
        f_handler = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(LOGS_DIR, 'dragen-logs'), when='midnight')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        f_handler.setFormatter(formatter)
        self.logger.addHandler(f_handler)
        self.logger.setLevel(level=logging.DEBUG)

    def run(self,rve_df,grains_df,equiv_d, mu, sigma,lower=None,upper=None,plt_name = None,save= True,plot=False,store_path = None,filename = None,fig_path = None):

        self.setup_logging()
        self.logger.info('------------------------------------------------------------------------------')
        self.logger.info('substructure generation begins')
        self.logger.info('------------------------------------------------------------------------------')
        rve_data = pd.DataFrame()

        for i in range(len(grains_df)):

            grain_data = grains_df.iloc[i]
            phaseID = int(grain_data['phaseID'])
            grain_id = grain_data['GrainID']
            x = rve_df[rve_df['GrainID'] == grain_id + 1]['x'].to_numpy().reshape((-1, 1)) #needs + 1 finally
            y = rve_df[rve_df['GrainID'] == grain_id + 1]['y'].to_numpy().reshape((-1, 1))
            z = rve_df[rve_df['GrainID'] == grain_id + 1]['z'].to_numpy().reshape((-1, 1))

            points = np.concatenate((x, y, z), axis=1)

            if phaseID == 2:

                orientation = (grain_data['phi1'], grain_data['PHI'], grain_data['phi2'])
                grain = Grain(box_size=self.box_size, n_pts=self.n_pts, v=grain_data['final_conti_volume'], points=points,
                              phaseID=phaseID,grainID=grain_id,orientation=orientation)

                grain.gen_packets(equiv_d)
                grain.gen_blocks(mu, sigma,lower=lower,upper=upper)
                grain.block_orientation_assignment()
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

        if save:

            if filename:

                save_data(rve_data,store_path,filename)

            else:

                save_data(rve_data,store_path)

        if plot:

            for name in plt_name:

                rve_substruct_plotter(rve_data,name,fig_path)

        self.logger.info('substructure generation successful')
        self.logger.info('------------------------------------------------------------------------------')

        return rve_data

if __name__ == '__main__':

    test_run = Run(30,60,'F:/pycharm/2nd_mini_thesis/substructure')
    rve_data = pd.read_csv('F:/pycharm/2nd_mini_thesis/dragen-master/OutputData/2021-07-23_0/substruct_data_abq.csv') #changes into rve data
    grains_df = pd.read_csv('F:/pycharm/2nd_mini_thesis/dragen-master/OutputData/2021-07-23_0/Generation_Data/grain_data_input.csv')#grain_data
    test_run.run(rve_data,grains_df,6.8,0.5,0.01,store_path='./',plt_name=['packet','block'],plot=True,fig_path='D:/')