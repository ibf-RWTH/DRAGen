# _*_ coding: utf-8 _*_
"""
Time:     2021/8/9 21:23
Author:   Linghao Kong
Version:  V 0.1
File:     predictor
Describe: Write during the internship at IEHK RWTH"""

import pandas as pd
import numpy as np

bainite_data = pd.read_csv('./trainning/bainite.csv',names=['block_id','grain_id','packet_id','phi1','PHI','phi2','block_thickness'])
bainite_data.dropna(how='any',inplace=True)

pag_data = pd.read_csv('./trainning/pag_info.csv',names=['a','b','alpha','phi1','PHI','phi2','grain_id'])
pag_data['c'] = pag_data['b']
pag_data['volume'] = 4/3*np.pi*pag_data['a']*pag_data['b']*pag_data['c']
order = ['a','b','c','alpha','phi1','PHI','phi2','volume','grain_id']
pag_data = pag_data[order]
#filter
filtered_pag_data = pd.DataFrame()
for i in range(len(pag_data)):
    if pag_data.iloc[i]['grain_id'] in list(set(bainite_data['grain_id'])):
        filtered_pag_data = filtered_pag_data.append(pag_data.iloc[i])

pag_data = filtered_pag_data[order]
#re-index block_id
old_bid = list(set(bainite_data['block_id']))
new_bid = np.arange(1,len(old_bid)+1)
oldb_to_newb = dict(zip(old_bid,new_bid))

new_blockid = bainite_data['block_id'].map(lambda bid: oldb_to_newb[bid])
#re-index pag_id
old_bid = list(set(bainite_data['grain_id']))
new_bid = np.arange(1,len(old_bid)+1)
oldpg_to_newpg = dict(zip(old_bid,new_bid))

new_pagid = bainite_data['grain_id'].map(lambda gid: oldpg_to_newpg[gid])
new_pagid2 = pag_data['grain_id'].map(lambda gid: oldpg_to_newpg[gid])
bainite_data['block_id'] = new_blockid
bainite_data['grain_id'] = new_pagid
pag_data['grain_id'] = new_pagid2
bainite_data.sort_values(by=['grain_id','packet_id'],inplace=True)

# re-index pak-id
n = 1
pid_list = [n]
for i in range(1,len(bainite_data)):
    if bainite_data.iloc[i]['grain_id'] != bainite_data.iloc[i-1]['grain_id']:
        n += 1

    if bainite_data.iloc[i]['grain_id'] == bainite_data.iloc[i-1]['grain_id'] and \
       bainite_data.iloc[i]['packet_id'] != bainite_data.iloc[i - 1]['packet_id']:
        n += 1

    pid_list.append(n)

bainite_data['packet_id'] = pid_list

small_idx = np.array(pag_data[pag_data['volume']<1].index.tolist())
small_gid = small_idx + 1

pag_data.drop(pag_data[pag_data['volume']<1].index,inplace=True)
# block_data.drop(block_data[])
for gid in small_gid:
    bainite_data.drop(bainite_data[bainite_data['grain_id'] == gid].index,inplace=True)

#re-index pag id
old_bid = list(set(bainite_data['grain_id']))
new_bid = np.arange(1,len(old_bid)+1)
oldpg_to_newpg = dict(zip(old_bid,new_bid))

new_pagid = bainite_data['grain_id'].map(lambda gid: oldpg_to_newpg[gid])
bainite_data['grain_id'] = new_pagid
#re-index packet id
n = 1
pid_list = [n]
for i in range(1,len(bainite_data)):
    if bainite_data.iloc[i]['grain_id'] != bainite_data.iloc[i-1]['grain_id']:
        n += 1

    if bainite_data.iloc[i]['grain_id'] == bainite_data.iloc[i-1]['grain_id'] and \
       bainite_data.iloc[i]['packet_id'] != bainite_data.iloc[i - 1]['packet_id']:
        n += 1

    pid_list.append(n)

bainite_data['packet_id'] = pid_list
pag_data.to_csv('F:/pycharm/2nd_mini_thesis/dragen-master/ExampleInput/example_pag_inp2.csv')
bainite_data.to_csv('F:/pycharm/2nd_mini_thesis/dragen-master/ExampleInput/example_block_inp2.csv')



