# _*_ coding: utf-8 _*_
"""
Time:     2021/6/16 14:00
Author:   Linghao Kong
Version:  V 0.1
File:     rve_substructure(updated).py
Describe: Write during the internship at IEHK RWTH
"""
import pandas as pd
def save_data(data,store_path,filename='substruct_data_abq.csv'):

    filename = store_path + '/' + filename
    data.to_csv(filename)

def load_data(filename):

    data = pd.read_csv(filename)
    return data

if __name__ == '__main__':
    # grains_x, grains_y, grains_z, ic = rve_load(
    #     'F:/pycharm/2nd_mini_thesis/dragen-master/dragen/OutputData/2021-06-10_0/RVE_Numpy.npy', box_size=20, n_pts=40)
    # grains_list = grains_in_rve(grains_x, grains_y, grains_z, ic,
    #                             'F:/pycharm/2nd_mini_thesis/dragen-master/dragen/OutputData/2021-06-10_0/Generation_Data/grain_data_output_conti.csv')
    # data = grains_list
    # save_data(data,'./data.pkl')

    data = load_data('F:/pycharm/2nd_mini_thesis/dragen-master/dragen/test_run.pkl')
    for grain in data:

        print(grain.points_data)























