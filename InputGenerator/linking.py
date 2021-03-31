import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
import datetime
from geomloss import SamplesLoss
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Das scheint das beste Format für das GAN zu sein.
class Reconstructor:

    def __init__(self, ab, cb, ca, threshold=0.05, surrounding=0.05, calc_mean=False, drop=False):
        self.ab = self.adjust_axis(ab)
        self.cb = self.adjust_axis(cb)
        self.ca = self.adjust_axis(ca)
        self.threshold = threshold
        self.surrounding = surrounding
        self.calc_mean = calc_mean
        self.mean_percentage = 0
        self.drop = drop

        # New properties for bot
        self.result_df = pd.DataFrame()
        self.results = pd.DataFrame()
        self.slope_dict = dict()
        self.rve_inp = pd.DataFrame()

    @staticmethod
    def adjust_axis(df):
        # locations: [Area, Aspect Ratio, Slope] - Sind so fixed
        df2 = df.copy().dropna(axis=0)
        columns = df2.columns
        df2['Axes1'] = np.nan
        df2['Axes2'] = np.nan
        df2['Axes1'] = (df2[columns[0]] * df2[columns[1]] / np.pi) ** 0.5  # Major
        df2['Axes2'] = df2[columns[0]] / (np.pi * df2['Axes1'])
        # Switch axis in 45 - 135
        for j, row in df2.iterrows():
            if 45 < row[2] <= 135:
                temp1 = df2['Axes2'].iloc[j]
                temp2 = df2['Axes1'].iloc[j]
                df2['Axes1'].iloc[j] = temp1
                df2['Axes2'].iloc[j] = temp2
            else:
                pass

        return df2

    @staticmethod
    def sample_point(df):
        index = np.random.randint(low=0, high=len(df))
        col = df.columns
        a = df[col[-2]].iloc[index]
        b = df[col[-1]].iloc[index]
        slope = df[col[2]].iloc[index]
        df2 = df.drop(labels=df.index[index], axis=0)
        return (a, b), slope, df2

    @staticmethod
    def find_corresponding_value_per(df, sort_by, lookup, value, surr):
        """
        :param df:
        :param sort_by:
        :param lookup:
        :param value:
        :param surr:
        :return:
        find corresponding value by percentage deviation
        """
        global i
        msurr = -1 * surr
        values = dict()
        test = df.copy()
        sorted_df = test.sort_values(by=sort_by, axis=0)
        header = sorted_df.columns
        sorted_df['Deviation'] = (sorted_df[sort_by] - value) / value
        for i in range(1, len(sorted_df)):
            if msurr < sorted_df['Deviation'].iloc[i - 1] < surr:
                # Only Value and Slope
                values.update(
                    {sorted_df[sort_by].iloc[i - 1]: (sorted_df[lookup].iloc[i - 1], sorted_df[header[2]].iloc[i - 1])})
            else:
                pass

        return values

    def run(self, n_points, save_state=False):
        starttime = time.time()
        percentages = list()
        i = 0
        n_iter = 0
        df = self.ab.copy()
        cb = self.cb.copy()
        ca = self.ca.copy()
        result_storage = list()
        col_new = ['a_final', 'b_final', 'c_final', 'SlopeAB',
                   'SlopeCB', 'SlopeCA', 'z_1 (c)', 'z_2 (c)',
                   'Percentage Deviation', 'a-Difference', 'b-Difference']

        # Start running
        while i <= n_points:
            point, slope, df = self.sample_point(df)
            print('Remaining points: {}'.format(len(df)))
            if len(df) == 0:
                print('No Points remaining. Stop iterating')
                break

            # find c from b - here c is on the x axis
            dict1 = self.find_corresponding_value_per(df=cb, sort_by='Axes2', lookup='Axes1',
                                                      value=point[1], surr=self.surrounding)

            # find c from a - here c is on the x axis
            dict2 = self.find_corresponding_value_per(df=ca, sort_by='Axes2', lookup='Axes1',
                                                      value=point[0], surr=self.surrounding)
            # This is only 'compound' value. Not a real "c"
            # But the convergence shows that the approach is correct
            # Quadratic scaling here - maybe better ideas?
            if dict1.__len__() != 0 and dict2.__len__() != 0:
                diff = dict()
                for k, v in dict1.items():
                    for k1, v1 in dict2.items():
                        diff.update({(k, k1): (v[0], v1[0], round(abs(v[0] - v1[0]), 5), v[1], v1[1])})
                x_arr = np.array(list(diff.values()))
                y_arr = np.array(list(diff.keys()))
                min_abs_diff = round(np.amin(x_arr[:, 2]), 5)
                index_min = int(np.argmin(x_arr[:, 2]))
                """
                y_arr indices: b, a
                x_arr indices: z_b, z_a, diff, slopecb, slopeca
                """
                mean = np.mean([x_arr[index_min, 0], x_arr[index_min, 1]]).item()
                slopecb = x_arr[index_min, 3]
                slopeca = x_arr[index_min, 4]
                mean_c_value = round(mean, 5)
                percentage = round(min_abs_diff / mean_c_value, 5)
                a_diff = y_arr[index_min, 1] - point[0]
                b_diff = y_arr[index_min, 0] - point[1]
                # Calc the mean value
                if self.calc_mean:
                    x_a = np.mean([y_arr[index_min, 1], point[0]]).item()
                    y_b = np.mean([y_arr[index_min, 0], point[1]]).item()
                    new_point = (round(x_a, 5), round(y_b, 5), mean_c_value)
                else:
                    new_point = (round(point[0], 5), round(point[1], 5), mean_c_value)

                # Print out the results
                print('Minimal difference between both z: {} | Mean of both z: {} | Percentage deviation: {}%'
                      .format(min_abs_diff, mean_c_value, round(percentage * 100, 2)))

                if percentage <= self.threshold:
                    print('Difference small enough: Add Point')
                    print('The new point is: x={} / y={} / z={}'.format(new_point[0], new_point[1], new_point[2]))
                    i += 1
                    # Drop values from df
                    if self.drop:
                        indexcb = cb[cb['Axes1'] == x_arr[index_min, 0].item()].index
                        indexca = ca[ca['Axes2'] == x_arr[index_min, 1].item()].index
                        cb = cb.drop(labels=indexcb, inplace=False)
                        ca = ca.drop(labels=indexca, inplace=False)
                    # Reihenfolge direkt für
                    temp = [new_point[0], new_point[1],  mean_c_value, slope, slopecb, slopeca, x_arr[index_min, 0],
                            x_arr[index_min, 1],
                            round(percentage * 100, 2), a_diff, b_diff]
                    result_storage.append(temp)
                else:
                    print('Difference is to big. No convergence')
                    print('z1={} | z2={}'.format(x_arr[index_min, 0], x_arr[index_min, 1]))
                    print('The original point was: {} | {}'.format(round(point[0], 5), round(point[1], 5)))
                n_iter += 1
                print('Iterations: {}'.format(n_iter))
                print('Number of points: {}'.format(i))
                print('----------------------------------------------------')
                percentages.append(percentage)
            else:
                print('All points to far away - Next Point')
                print(point[0], point[1])
                print('----------------------------------------------------')
                n_iter += 1

        self.mean_percentage = round((np.array(percentages).sum() / len(percentages)) * 100, 3)
        results = np.array(result_storage)
        self.results = pd.DataFrame(results, columns=col_new)
        self.result_df = self.results.drop(labels=(col_new[6:]), axis=1)
        if save_state:
            self.result_df.to_hdf('Results_{}_{}_{}_{}_{}.h5'.format(self.threshold, self.surrounding, n_points,
                                                                     self.cb.__len__(), self.calc_mean),
                                  key='results')
        print('----------------------------------------------------')
        elapsed = time.time() - starttime
        print(str(datetime.timedelta(seconds=elapsed)))

    def plot_comparison(self, kind='kde', close=False):
        fig, ax = plt.subplots(2, 3, 'none', figsize=(18, 10))
        converged_ = self.result_df
        length = converged_.__len__()
        if kind == 'kde':
            # Plot the fake points in three constellations
            sns.kdeplot(data=converged_, x='a_final', y='b_final',
                        ax=ax[0][0])
            sns.kdeplot(data=converged_, x='c_final', y='b_final',
                        ax=ax[0][1])
            sns.kdeplot(data=converged_, x='c_final', y='a_final',
                        ax=ax[0][2])
            ax[0][0].set_title('axb')
            ax[0][1].set_title('cxb')
            ax[0][2].set_title('cxa')
            ax[0][0].set_xlim(0, 11)
            ax[0][1].set_xlim(0, 15)
            ax[0][2].set_xlim(0, 16)
            ax[0][0].set_ylim(0, 14)
            ax[0][1].set_ylim(0, 21)
            ax[0][2].set_ylim(0, 21)

            sns.kdeplot(data=self.ab, x='Axes1', y='Axes2', ax=ax[1][0])
            sns.kdeplot(data=self.cb, x='Axes1', y='Axes2', ax=ax[1][1])
            sns.kdeplot(data=self.ca, x='Axes1', y='Axes2', ax=ax[1][2])
            ax[1][0].set_title('X vs Y')
            ax[1][1].set_title('Z vs Y')
            ax[1][2].set_title('Z vs X')
            ax[1][0].set_xlim(0, 11)
            ax[1][1].set_xlim(0, 15)
            ax[1][2].set_xlim(0, 16)
            ax[1][0].set_ylim(0, 14)
            ax[1][1].set_ylim(0, 21)
            ax[1][2].set_ylim(0, 21)

        elif kind == 'hist':
            # Plot the fake points in three constellations
            sns.histplot(data=converged_, x='a_final', y='b_final',
                         ax=ax[0][0])
            sns.histplot(data=converged_, x='c_final', y='b_final',
                         ax=ax[0][1])
            sns.histplot(data=converged_, x='c_final', y='a_final',
                         ax=ax[0][2])
            ax[0][0].set_title('axb')
            ax[0][1].set_title('cxb')
            ax[0][2].set_title('cxa')
            ax[0][0].set_xlim(0, 11)
            ax[0][1].set_xlim(0, 15)
            ax[0][2].set_xlim(0, 16)
            ax[0][0].set_ylim(0, 14)
            ax[0][1].set_ylim(0, 21)
            ax[0][2].set_ylim(0, 21)

            sns.histplot(data=self.ab.sample(n=length, random_state=1), x='Axes1', y='Axes2', ax=ax[1][0])
            sns.histplot(data=self.cb.sample(n=length, random_state=1), x='Axes1', y='Axes2', ax=ax[1][1])
            sns.histplot(data=self.ca.sample(n=length, random_state=1), x='Axes1', y='Axes2', ax=ax[1][2])
            ax[1][0].set_title('X vs Y')
            ax[1][1].set_title('Z vs Y')
            ax[1][2].set_title('Z vs X')
            ax[1][0].set_xlim(0, 11)
            ax[1][1].set_xlim(0, 15)
            ax[1][2].set_xlim(0, 16)
            ax[1][0].set_ylim(0, 14)
            ax[1][1].set_ylim(0, 21)
            ax[1][2].set_ylim(0, 21)

        elif kind == 'reg':
            # Plot the fake points in three constellations
            sns.regplot(data=converged_, x='a_final', y='b_final',
                        ax=ax[0][0])
            sns.regplot(data=converged_, x='c_final', y='b_final',
                        ax=ax[0][1])
            sns.regplot(data=converged_, x='c_final', y='a_final',
                        ax=ax[0][2])
            ax[0][0].set_title('axb')
            ax[0][1].set_title('cxb')
            ax[0][2].set_title('cxa')
            ax[0][0].set_xlim(0, 11)
            ax[0][1].set_xlim(0, 15)
            ax[0][2].set_xlim(0, 16)
            ax[0][0].set_ylim(0, 14)
            ax[0][1].set_ylim(0, 21)
            ax[0][2].set_ylim(0, 21)

            sns.regplot(data=self.ab.sample(n=length, random_state=1), x='Axes1', y='Axes2', ax=ax[1][0])
            sns.regplot(data=self.cb.sample(n=length, random_state=1), x='Axes1', y='Axes2', ax=ax[1][1])
            sns.regplot(data=self.ca.sample(n=length, random_state=1), x='Axes1', y='Axes2', ax=ax[1][2])
            ax[1][0].set_title('X vs Y')
            ax[1][1].set_title('Z vs Y')
            ax[1][2].set_title('Z vs X')
            ax[1][0].set_xlim(0, 11)
            ax[1][1].set_xlim(0, 15)
            ax[1][2].set_xlim(0, 16)
            ax[1][0].set_ylim(0, 14)
            ax[1][1].set_ylim(0, 21)
            ax[1][2].set_ylim(0, 21)

        else:
            print('No supported plot type: {}'.format(kind))

        fig.suptitle('Threshold: {} | Range {}'.format(self.threshold, self.surrounding), fontsize=16)
        plt.savefig('Results_{}_{}_{}_{}_{}.png'.format(self.threshold, self.surrounding, kind, self.drop,
                                                        self.calc_mean))
        if close:
            plt.close()

    def compute_slopes(self):
        # First predict the real slopes
        tdxbn_x_real = self.ab['Axes1'].to_numpy().reshape(-1, 1)
        tdxbn_y_real = self.ab['Axes2'].to_numpy().reshape(-1, 1)
        rdxbn_x_real = self.cb['Axes1'].to_numpy().reshape(-1, 1)
        rdxbn_y_real = self.cb['Axes2'].to_numpy().reshape(-1, 1)
        rdxtd_x_real = self.ca['Axes1'].to_numpy().reshape(-1, 1)
        rdxtd_y_real = self.ca['Axes2'].to_numpy().reshape(-1, 1)

        # Fake slopes
        data = self.result_df
        tdxbn_x_fake = data['a_final'].to_numpy().reshape(-1, 1)
        tdxbn_y_fake = data['b_final'].to_numpy().reshape(-1, 1)
        rdxbn_x_fake = data['c_final'].to_numpy().reshape(-1, 1)
        rdxbn_y_fake = data['b_final'].to_numpy().reshape(-1, 1)
        rdxtd_x_fake = data['c_final'].to_numpy().reshape(-1, 1)
        rdxtd_y_fake = data['a_final'].to_numpy().reshape(-1, 1)
        all = [[tdxbn_x_real, tdxbn_y_real, tdxbn_x_fake, tdxbn_y_fake],
               [rdxbn_x_real, rdxbn_y_real, rdxbn_x_fake, rdxbn_y_fake],
               [rdxtd_x_real, rdxtd_y_real, rdxtd_x_fake, rdxtd_y_fake]
               ]
        names = ['tdxbn', 'rdxbn', 'rdxtd']
        for i in range(3):
            self.slope_dict.update({'{}'.format(names[i]): (LinearRegression().fit(all[i][0], all[i][1]).coef_,
                                                            LinearRegression().fit(all[i][2], all[i][3]).coef_)})

    def get_rve_input(self, bs):
        max_volume = bs*bs*bs
        grain_vol = 0
        data = self.result_df.copy()

        inp_list = list()
        while grain_vol < max_volume:
            idx = np.random.randint(0, data.__len__())
            grain = data[['a_final', 'b_final', 'c_final']].iloc[idx].tolist()
            data = data.drop(labels=data.index[idx], axis=0)
            vol = 4/3 * np.pi * grain[0] * grain[1] * grain[2]
            grain_vol += vol
            inp_list.append([grain[0], grain[1], grain[2], vol])
            #print(data.__len__())

        # Del last if to big:
        if grain_vol >= max_volume:
            inp_list.pop(-1)

        header = ['a', 'b', 'c', 'volume']
        self.rve_inp = pd.DataFrame(inp_list, columns=header)














