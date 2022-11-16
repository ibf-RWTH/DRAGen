# _*_ coding: utf-8 _*_
"""
Time:     2021/6/18 0:29
Author:   Linghao Kong
Version:  V 0.1
File:     run
Describe: Write during the internship at IEHK RWTH"""

from dragen.substructure.substructure import plot_rve_subs
from dragen.substructure.data import save_data
from dragen.substructure.substructure import gen_blocks, compute_bt, gen_packets
from scipy.stats import moment
from scipy.stats import gaussian_kde
from dragen.substructure.DataParser import DataParser
from dragen.substructure.Crystallograhy import CrystallInfo
import pandas as pd
import numpy as np
from dragen.utilities.InputInfo import RveInfo
import matplotlib.pyplot as plt
def record_orientations(grains_df):
    for i in range(int(grains_df['GrainID'].max())):
        CrystallInfo.gid2orientation.append(
            (grains_df.iloc[0]['phi1'], grains_df.iloc[0]['PHI'], grains_df.iloc[0]['phi2']))


def assign_packet_variants(_rve_data):
    print("assign packets with variants")
    pids = _rve_data['packet_id'].unique()
    for pid in pids:
        CrystallInfo.assign_variants(packet_id=str(pid))
    print("finish assignment of packets variants")


def assign_block_variants(_rve_data):
    print("assign blocks with variants")
    num_packets = _rve_data['packet_id'].max()
    for i in range(1, num_packets + 1):
        packet = _rve_data[_rve_data['packet_id'] == i].copy()
        CrystallInfo.assign_bv(packet=packet)
        _rve_data.loc[_rve_data['packet_id'] == i, 'block_variant'] = packet['block_variant']
    print("finish assignment of block variants")


def subsid2num(_rve_data, subs_key):
    subs_id = _rve_data[subs_key].unique().tolist()
    # CrystallInfo.old_pid = packet_id
    n_id = np.arange(1, len(subs_id) + 1)
    pid_to_nid = dict(zip(subs_id, n_id))
    pid_in_rve = _rve_data[subs_key].map(lambda pid: pid_to_nid[pid])
    _rve_data[subs_key] = pid_in_rve
    return subs_id


class Run():

    def __init__(self):
        self.rve_data = None
        self.data_parser = DataParser()
        self.crystall_info = CrystallInfo()

    @staticmethod
    def del_zerobt(_df: pd.DataFrame):
        sampled_df = _df.groupby('block_id', as_index=False).first()
        sampled_df.sort_values(by=['x', 'y', 'z'], inplace=True)
        sampled_df.reset_index(inplace=True)
        # get the id of block with zero thickness
        zero_ids = sampled_df.loc[sampled_df['block_thickness'] == 0, 'block_id'].tolist()
        if len(zero_ids) > 0:
            print('modifying zero block thickness...')
            for zid in zero_ids:
                # iter over ids of block with zero thickness
                idx = sampled_df[sampled_df['block_id'] == zid].index
                n = 1
                m = 1
                while True:
                    if idx + n < len(sampled_df):
                        if sampled_df.loc[idx + n, 'block_thickness'].values != 0:
                            nonzero_idx = idx + n
                            break
                        else:
                            n += 1

                    if idx - m > 0:
                        if sampled_df.loc[idx - m, 'block_thickness'].values != 0:
                            nonzero_idx = idx - m
                            break
                        else:
                            m += 1

                new_bid = sampled_df.loc[nonzero_idx, 'block_id'].values[0]
                new_bt = sampled_df.loc[nonzero_idx, 'block_thickness'].values[0]
                _df.loc[_df['block_id'] == zid, 'block_id'] = new_bid
                _df.loc[_df['block_id'] == new_bid, 'block_thickness'] = new_bt  # block id is modified here...

    def run(self, rve_df, grains_df):

        RveInfo.LOGGER.info('------------------------------------------------------------------------------')
        RveInfo.LOGGER.info('substructure generation begins')
        RveInfo.LOGGER.info('------------------------------------------------------------------------------')
        grains_df.sort_values(by=['GrainID'], inplace=True)
        print("record orientation of each grain")
        record_orientations(grains_df=grains_df)
        print("finishing recording grains orientation")
        print("compute global habit planes norm for each grain")
        grains_df.apply(lambda grain: self.crystall_info.get_global_hp_norm(GrainID=grain['GrainID'],
                                                                            orientation=(
                                                                                grain['phi1'], grain['PHI'],
                                                                                grain['phi2'])),
                        axis=1)
        print("finish computing global habit planes")
        print("start fitting distribution")
        pv_distribution = self.data_parser.parse_packet_data()
        bt_distribution = self.data_parser.parse_block_data()
        print("finish fitting distribution")

        rve_df['block_id'] = rve_df['packet_id'] = rve_df['GrainID'].astype(str)
        rve_df['block_variant'] = np.NAN
        rve_df['phi1'] = rve_df.apply(lambda data: CrystallInfo.gid2orientation[int(data['GrainID'] - 1)][0],
                                      axis=1)
        rve_df['PHI'] = rve_df.apply(lambda data: CrystallInfo.gid2orientation[int(data['GrainID'] - 1)][1],
                                     axis=1)
        rve_df['phi2'] = rve_df.apply(lambda data: CrystallInfo.gid2orientation[int(data['GrainID'] - 1)][2],
                                      axis=1)

        subs_rve = rve_df[(rve_df["phaseID"] == 2) | (rve_df["phaseID"] == 4)].copy()

        print("start packets generation")
        _rve_data = gen_packets(rve=subs_rve, pv_distribution=pv_distribution,
                                num_packets_list=self.data_parser.num_packets_list, grains_df=grains_df)
        print("finish packets generation")

        assign_packet_variants(_rve_data)
        CrystallInfo.old_pid = subsid2num(_rve_data=_rve_data, subs_key='packet_id')

        print("start blocks generation")
        _rve_data = gen_blocks(rve=_rve_data, bt_distribution=bt_distribution)
        print("finish blocks generation")

        compute_bt(rve=_rve_data)
        self.del_zerobt(_df=_rve_data)

        subsid2num(_rve_data=_rve_data, subs_key='block_id')
        assign_block_variants(_rve_data=_rve_data)

        print("compute block orientation")
        angles = _rve_data.apply(
            lambda data: CrystallInfo.comp_angle(CrystallInfo.gid2orientation[int(data['GrainID'] - 1)],
                                                 data['block_variant']), axis=1)
        _rve_data['angles'] = angles
        _rve_data['phi1'] = _rve_data.apply(lambda data: data['angles'][0], axis=1)
        _rve_data['PHI'] = _rve_data.apply(lambda data: data['angles'][1], axis=1)
        _rve_data['phi2'] = _rve_data.apply(lambda data: data['angles'][2], axis=1)
        print("finish computing block orientation")

        rve_df.loc[(rve_df["phaseID"] == 2) | (rve_df["phaseID"] == 4), 'packet_id'] = _rve_data['packet_id']
        rve_df.loc[(rve_df["phaseID"] == 2) | (rve_df["phaseID"] == 4), 'block_id'] = _rve_data['block_id']
        rve_df.loc[(rve_df["phaseID"] == 2) | (rve_df["phaseID"] == 4), 'block_thickness'] = _rve_data[
            'block_thickness']
        rve_df.loc[(rve_df["phaseID"] == 2) | (rve_df["phaseID"] == 4), 'phi1'] = _rve_data['phi1']
        rve_df.loc[(rve_df["phaseID"] == 2) | (rve_df["phaseID"] == 4), 'PHI'] = _rve_data['PHI']
        rve_df.loc[(rve_df["phaseID"] == 2) | (rve_df["phaseID"] == 4), 'phi2'] = _rve_data['phi2']

        RveInfo.rve_data_substructure = rve_df
        self.rve_data = rve_df

        rve_df.n_pts = RveInfo.n_pts
        rve_df.box_size = RveInfo.box_size
        rve_df.box_size_y = RveInfo.box_size_y
        rve_df.box_size_z = RveInfo.box_size_z

        if RveInfo.save:

            if RveInfo.filename:

                save_data(rve_df, RveInfo.store_path, RveInfo.filename)

            else:
                save_data(rve_df, RveInfo.store_path)

        if RveInfo.plot:

            for name in RveInfo.plt_name:
                plot_rve_subs(rve_df, name, RveInfo.fig_path)

        RveInfo.LOGGER.info('substructure generation successful')
        RveInfo.LOGGER.info('------------------------------------------------------------------------------')
        rve_df.to_csv(r"F:\codes\DRAGen\dragen\test\results\rve_data.csv")
        return rve_df

    def post_processing(self, k, sigma=2):
        rve_data = RveInfo.rve_data_substructure

        def gaussian_kernel(x1, x2, sigma=2):
            return np.exp(-np.power(x1 - x2, 2).sum() / (2 * sigma ** 2))

        def k_moments(x, k):
            m_list = []
            for i in range(k):
                if i == 0:
                    m = np.mean(x)
                else:
                    m = moment(x, i + 1)
                m_list.append(m)
            return np.array(m_list)

        pag_path = RveInfo.store_path + '/Generation_Data/grain_data_output.csv'
        pag_df = pd.read_csv(pag_path)
        discrete_vol = pag_df['final_discrete_volume']
        n_pag = len(pag_df)
        mean_pagvol = np.mean(discrete_vol)
        std_pagvol = np.std(discrete_vol)

        n_pak = int(rve_data['packet_id'].max())
        mean_pakvol = RveInfo.box_size ** 3 / n_pak
        variance_pakvol = 0
        for i in range(1, n_pak + 1):
            pakvol = len(rve_data[rve_data['packet_id'] == i]) / len(rve_data) * RveInfo.box_size ** 3
            variance_pakvol += (pakvol - mean_pakvol) ** 2 / n_pak

        std_pakvol = variance_pakvol ** 0.5

        n_block = int(rve_data['block_id'].max())
        bt_df = rve_data.groupby(['block_id']).first()
        mean_bt = bt_df['block_thickness'].mean()
        std_bt = bt_df['block_thickness'].std()

        # compute MMD
        measured_bt = pd.read_csv(RveInfo.subs_file)['block_thickness']
        generated_bt = bt_df['block_thickness']

        # compute k-moments
        measured_bt_kmoments = k_moments(measured_bt, k)
        gen_bt_kmoments = k_moments(generated_bt, k)

        # get MMD
        MMD = gaussian_kernel(gen_bt_kmoments, measured_bt_kmoments, sigma)

        result_path = RveInfo.store_path + '/Postprocessing/result.txt'
        with open(result_path, 'w') as f:
            f.write('Parent Austenitic Grains Statistical Info:\n')
            f.write('total number: {}\n'.format(n_pag))
            f.write('average(volume): {}\n'.format(mean_pagvol))
            f.write('standard variance(volume): {}\n'.format(std_pagvol))
            f.write('\n')
            f.write('Packets Statistical Info:\n')
            f.write('total number: {}\n'.format(n_pak))
            f.write('average(volume): {}\n'.format(mean_pakvol))
            f.write('standard variance(volume): {}\n'.format(std_pakvol))
            f.write('\n')
            f.write('Blocks Statistical Info:\n')
            f.write('total number: {}\n'.format(n_block))
            f.write('average(thickness): {}\n'.format(mean_bt))
            f.write('standard variance(thickness): {}\n'.format(std_bt))
            f.write('MMD: {}\n'.format(MMD))
            f.write('\n')
            if MMD <= 0.01:
                f.write('##warning: the MMD is too small, please check statistical features!')

        if RveInfo.subs_file_flag:
            bt_df = pd.read_csv(RveInfo.subs_file)
            measured_bt = np.sort(bt_df['block_thickness'].to_numpy())
            kernel1 = gaussian_kde(measured_bt)
            blocks = rve_data.groupby('block_id').first()
            generated_bt = np.sort(blocks['block_thickness'].to_numpy())
            kernel2 = gaussian_kde(generated_bt)
            plt.plot(measured_bt, kernel1(measured_bt), label='Real_Distribution')
            plt.plot(generated_bt, kernel2(generated_bt), label='DRAGen_Distribution')
            plt.xlabel('Block_Thickness({}m)'.format(r'$\mu$'), fontsize=15)
            plt.ylabel('Frequency', fontsize=15)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(fontsize=12.5, loc=1)
            plt.savefig(RveInfo.store_path + '/Postprocessing/compare_distribution.png')

        RveInfo.LOGGER.info('substructure postprocessing successful')


if __name__ == '__main__':
    # import datetime
    #
    # start = datetime.datetime.now()
    # test_run = Run(20, 40, 'F:/pycharm/2nd_mini_thesis/substructure')
    # rve_data = pd.read_csv(
    #     'F:/pycharm/2nd_mini_thesis/dragen-master/OutputData/2021-09-23_0/substruct_data_abq.csv')  # changes into rve data
    # # grains_df = pd.read_csv('F:/pycharm/2nd_mini_thesis/dragen-master/OutputData/2021-09-23_0/Generation_Data/grain_data_input.csv')#grain_data
    # # test_run.run()
    # # end = datetime.datetime.now()
    # # print('running time is',end-start)
    # df = rve_data[rve_data['block_id'] == 1]
    # print(df)
    subs_run = Run()
    RveInfo.subs_file_flag = True
    RveInfo.block_file = r"/ExampleInput/example_block_inp.csv"
    block_data = pd.read_csv(RveInfo.block_file)
    RveInfo.bt_min = block_data['block_thickness'].min()
    RveInfo.bt_max = block_data['block_thickness'].max()
    rve_df = pd.read_csv(r"X:\DRAGen\DRAGen\dragen\test\results\periodic_rve_df.csv")
    grains_df = pd.read_csv(r"/dragen/test/results/grains_df.csv")
    rve_data = subs_run.run(rve_df=rve_df, grains_df=grains_df)
    print(rve_data)
    rve_data.to_csv(r"X:\DRAGen\DRAGen\dragen\test\results\test_result.csv")
