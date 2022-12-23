import sys
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from dragen.utilities.InputInfo import RveInfo
from scipy.stats import lognorm, truncnorm, uniform
import matplotlib.pyplot as plt
from typing import Tuple
from dragen.substructure.DataParser import SubsDistribution, DataParser, subs_sampler


def lc_to_gc(orientation, habit_plane_list):
    R1 = np.array([[np.cos(np.deg2rad(orientation[0])), -np.sin(np.deg2rad(orientation[0])), 0],
                   [np.sin(np.deg2rad(orientation[0])), np.cos(np.deg2rad(orientation[0])), 0],
                   [0, 0, 1]], dtype=object)

    R2 = np.array([[1, 0, 0],
                   [0, np.cos(np.deg2rad(orientation[1])), -np.sin(np.deg2rad(orientation[1]))],
                   [0, np.sin(np.deg2rad(orientation[1])), np.cos(np.deg2rad(orientation[1]))]], dtype=object)

    R3 = np.array([[np.cos(np.deg2rad(orientation[2])), -np.sin(np.deg2rad(orientation[2])), 0],
                   [np.sin(np.deg2rad(orientation[2])), np.cos(np.deg2rad(orientation[2])), 0],
                   [0, 0, 1]], dtype=object)

    result = np.dot(R3, R2)
    R = np.matrix(np.dot(result, R1))
    R = R.astype(float)  # new error has to convert data type
    R_I = R.I

    habit_plane_list = np.array(habit_plane_list)
    transferred_hp_list = (R_I.dot(habit_plane_list.T)).T

    return transferred_hp_list


def choose_norm(orientation: tuple):
    hp_normal_list = lc_to_gc(orientation=orientation,
                              habit_plane_list=[[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    index = np.random.randint(4)
    chosen_norm = hp_normal_list[index, ...]
    return chosen_norm


def recursive_bisect(grain: pd.DataFrame, num_grid_points: list, chosen_norm: np.ndarray):
    points_data = grain.copy()
    i = 0
    for num in num_grid_points:
        d = -(points_data['x'] * chosen_norm[0, 0] + points_data['y'] * chosen_norm[0, 1] +
              points_data['z'] * chosen_norm[0, 2])
        d.sort_values(inplace=True)
        chosen_d = d.iloc[num - 1]

        points_idx = points_data[points_data['x'] * chosen_norm[0, 0] +
                                 points_data['y'] * chosen_norm[0, 1] +
                                 points_data['z'] * chosen_norm[0, 2] + chosen_d >= 0].index
        points_data.loc[points_idx, 'packet_id'] = str(
            int(points_data.loc[points_idx, 'GrainID'].iloc[0])) + 'p' + str(i)
        packet_df = points_data.loc[points_idx]
        grain['packet_id'] = packet_df['packet_id']
        points_data.drop(packet_df.index, inplace=True)
        i += 1


def gen_packets(rve: pd.DataFrame, grains_df: pd.DataFrame, pv_distribution: SubsDistribution,
                num_packets_list: list = None) -> pd.DataFrame:
    """
    generate packets in rve
    :param num_packets_list: the number of packets in each grain
    :param grains_df: dataframe containing information of grains
    :param rve: the rve to be subdivided into packets
    :param pv_distribution: statistical distribution of packet volume
    :return: rve has packets
    """
    num_pags = int(rve['GrainID'].max())

    for i in range(1, num_pags + 1):
        grain = rve[rve['GrainID'] == i]
        grain_data = grains_df.iloc[i - 1]
        phaseID = int(grain_data['phaseID'])
        orientation = (grain_data['phi1'], grain_data['PHI'], grain_data['phi2'])
        grain_volume = grain_data['final_discrete_volume']
        # habit plane norm
        chosen_norm = choose_norm(orientation=orientation)
        # sample packet volumes
        if not RveInfo.subs_file_flag:
            pv_list = subs_sampler(subs_distribution=pv_distribution, y=grain_volume,
                                   interval=[RveInfo.pv_min, RveInfo.pv_max])
            # convert packet volumes to number of grid points in the grain
            num_grid_points = np.asarray(pv_list) / grain_volume * len(grain)
        else:
            num_packets = num_packets_list[i - 1]
            print(num_packets)
            num_grid_points = [np.random.randint(20, len(grain)) for _ in range(num_packets - 1)]
            num_grid_point = len(grain) - sum(num_grid_points)
            num_grid_points.append(num_grid_point)

        if len(num_grid_points) > 1:
            recursive_bisect(grain=grain, num_grid_points=num_grid_points, chosen_norm=chosen_norm)
        else:
            rve.loc[rve['GrainID'] == i, 'GrainID'] = grain['GrainID']

    return rve


if __name__ == "__main__":
    RveInfo.file_dict = dict()
    RveInfo.subs_file_flag = True
    RveInfo.file_dict[RveInfo.PHASENUM["Martensite"]] = r"F:\codes\DRAGen\ExampleInput\example_pag_inp.csv"
    RveInfo.block_file = r"F:\codes\DRAGen\ExampleInput\example_block_inp.csv"
    data_parser = DataParser()
    pv_distribution = data_parser.parse_packet_data()
    rve = pd.read_csv(r"F:\codes\DRAGen\OutputData\2022-11-16_000\substruct_data.csv")
    grains_df = pd.read_csv(r"F:\codes\DRAGen\OutputData\2022-11-16_000\Generation_Data\grain_data_output.csv")
    gen_packets(rve=rve, grains_df=grains_df, pv_distribution=pv_distribution,
                num_packets_list=data_parser.num_packets_list)
