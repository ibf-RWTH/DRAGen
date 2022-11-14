# _*_ coding: utf-8 _*_
"""
Time:     2021/6/18 0:29
Author:   Linghao Kong
Version:  V 0.1
File:     run
Describe: Write during the internship at IEHK RWTH"""

from dragen.substructure.substructure import plot_rve_subs
from dragen.substructure.data import save_data
from dragen.substructure.substructure import Grain, gen_blocks, compute_bt
from scipy.stats import moment
from scipy.stats import gaussian_kde
from dragen.stats.preprocessing import *
from dragen.substructure.DataParser import block_data_parser

class Run():

    def __init__(self):
        self.rve_data = None

    @staticmethod
    def get_orientations(block_df, grain_id):
        blocks = block_df[block_df['grain_id'] == grain_id + 1]  # +1...
        n_pack = len(list(set(blocks['packet_id'])))
        groups = blocks.groupby('packet_id')

        ori_dflist = []
        for name, group in groups:
            ori_df = group[['phi1', 'PHI', 'phi2']]
            ori_dflist.append(ori_df)

        n_pack = np.arange(n_pack)

        n_pack_to_ori = dict(zip(n_pack, ori_dflist))

        return n_pack_to_ori

    @staticmethod
    def get_bt_distribution(block_df):

        average_bt = block_df['block_thickness'].mean()
        RveInfo.t_mu = average_bt
        return average_bt

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
        _rve_data = pd.DataFrame()
        if RveInfo.subs_file_flag:
            assert RveInfo.block_file is not None, 'no substructure file given'
            block_df = pd.read_csv(RveInfo.block_file)
            self.get_bt_distribution(block_df)
            RveInfo.t_mu *= RveInfo.decreasing_factor

        for i in range(len(grains_df)):

            grain_data = grains_df.iloc[i]
            phaseID = int(grain_data['phaseID'])
            grain_id = grain_data['GrainID']
            x = rve_df[rve_df['GrainID'] == grain_id]['x'].to_numpy().reshape((-1, 1))
            y = rve_df[rve_df['GrainID'] == grain_id]['y'].to_numpy().reshape((-1, 1))
            z = rve_df[rve_df['GrainID'] == grain_id]['z'].to_numpy().reshape((-1, 1))

            points = np.concatenate((x, y, z), axis=1)

            if phaseID == 2 or 4:
                orientation = (grain_data['phi1'], grain_data['PHI'], grain_data['phi2'])
                grain = Grain(v=grain_data['final_conti_volume'], points=points,
                              phaseID=phaseID, grainID=grain_id, orientation=orientation)

                if RveInfo.subs_file_flag:
                    old_gid = grain_data['old_gid']
                    blocks = block_df[block_df['grain_id'] == old_gid + 1]
                    n_pack = len(list(set(blocks['packet_id'])))
                    orientations = self.get_orientations(block_df, old_gid)
                    grain.gen_packs(n_pack=n_pack, orientations=orientations)

                else:
                    assert RveInfo.equiv_d is not None, 'no valid definition for equiv_d'
                    assert RveInfo.p_sigma is not None, 'no valid definition for p_sigma'
                    assert RveInfo.t_mu is not None, 'no valid definition for t_mu'
                    # grain.gen_packs(RveInfo.equiv_d, sigma=RveInfo.p_sigma, block_thickness=RveInfo.t_mu,
                    #                b_sigma=RveInfo.b_sigma, lower_t=RveInfo.lower, upper_t=RveInfo.upper,
                    #                circularity=RveInfo.circularity) # check later
                _rve_data = pd.concat([_rve_data, grain.points_data])

            else:
                grain_data = pd.DataFrame(points, columns=['x', 'y', 'z'])
                grain_data['GrainID'] = grain_id
                grain_data['phaseID'] = phaseID
                grain_data['packet_id'] = grain_id
                grain_data['block_id'] = grain_id
                grain_data['block_orientation'] = np.NaN
                _rve_data = pd.concat([_rve_data, grain_data])

        RveInfo.rve_data_substructure = _rve_data
        self.rve_data = _rve_data

        # self.del_zerobt(_rve_data)  # del blocks with 0 thickness

        # _rve_data.loc[_rve_data['block_id'].isnull(), 'block_id'] = _rve_data[_rve_data['block_id'].isnull()][
        #                                                                 'packet_id'] + '0'
        bt_distribution = block_data_parser()
        packet_id = _rve_data['packet_id'].unique().tolist()
        n_id = np.arange(1, len(packet_id) + 1)
        pid_to_nid = dict(zip(packet_id, n_id))
        pid_in_rve = _rve_data['packet_id'].map(lambda pid: pid_to_nid[pid])
        _rve_data['packet_id'] = pid_in_rve
        print("start blocks generation")
        _rve_data = gen_blocks(rve=_rve_data, bt_distribution=bt_distribution)
        print("finish blocks generation")

        block_id = _rve_data['block_id'].unique().tolist()
        n2_id = np.arange(1, len(block_id) + 1)
        bid_to_nid = dict(zip(block_id, n2_id))
        bid_in_rve = _rve_data['block_id'].map(lambda bid: bid_to_nid[bid])
        _rve_data['block_id'] = bid_in_rve

        compute_bt(rve=_rve_data)
        self.del_zerobt(_df=_rve_data)

        _rve_data.n_pts = RveInfo.n_pts
        _rve_data.box_size = RveInfo.box_size
        _rve_data.box_size_y = RveInfo.box_size_y
        _rve_data.box_size_z = RveInfo.box_size_z

        if RveInfo.save:

            if RveInfo.filename:

                save_data(_rve_data, RveInfo.store_path, RveInfo.filename)

            else:
                save_data(_rve_data, RveInfo.store_path)

        if RveInfo.plot:

            for name in RveInfo.plt_name:
                plot_rve_subs(_rve_data, name, RveInfo.fig_path)

        RveInfo.LOGGER.info('substructure generation successful')
        RveInfo.LOGGER.info('------------------------------------------------------------------------------')
        return _rve_data

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

