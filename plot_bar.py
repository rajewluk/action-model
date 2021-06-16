from pathlib import Path
from os import walk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
import configparser
import ast
import math
import sys
from plot_shared import *


def print_col_stats(collection, col_name, data_file):
    df_col = data_file[col_name]
    agg_mean = df_col.aggregate('mean')
    agg_std = df_col.aggregate('std')
    agg_var = df_col.aggregate('var')
    print("{}\t{:.3f}\t{:.3f}\t{:.3f}".format(col_name, agg_mean, agg_std, agg_var))
    pass


plan_config = configparser.ConfigParser()
plan_config.read_file(open('plan.cfg'))
result_root = plan_config['Plot'].get('result-root')
sequence_numbers = plan_config['Plot'].get('sequence-numbers')
if sequence_numbers is not None:
    tmp_values = ast.literal_eval(sequence_numbers)
    if len(tmp_values) == 0:
        sequence_numbers = None
    else:
        sequence_numbers = tmp_values

files = get_csv_files(None, result_root)
files = filter_csv_files(files, sequence_numbers)

xs = list()
mins = list()
maxes = list()
means = list()
stds = list()
col = "S1_AVG_SLA"
col_fp = "S1_LB_FP"
col_act_1 = "S1_LB_ACT_1"
col_act_2 = "S1_LB_ACT_2"
col_dns_fp = "S1_DNS_FP"
col_dns_act_1 = "S1_DNS_ACT_1"
col_dns_act_2 = "S1_DNS_ACT_2"

cnt = 3
mode_count = 0
for file in files:
    c = cnt % 3
    if c == 0:
        cnt = 1
        mode_count += 1
    else:
        cnt += 1
    df = pd.read_csv(file, delimiter=";", index_col='Iter', decimal=",")
    file_name = file.name[:-4]
    seq_n = int(file_name[file_name.rfind('_') + 1:])
    mode = file.parts[-2][file.parts[-2].rfind('-') + 1:]
    # combination = file.parts[-2].replace("rnd", "v").replace("fix", "f").replace("joint", "JRAS").replace("no_cl", "CRA").replace("cl", "CLO")
    combination = file.parts[-2].replace("rnd-", "v").replace("fix-", "f").\
        replace("joint", "JRAS  ").replace("no_cl", "CRA  ").replace("cl", "CLO  ")
    print(combination)
    df_col = df[col]
    agg_mean = df_col.aggregate('mean')
    agg_min = df_col.aggregate('min')
    agg_max = df_col.aggregate('max')
    agg_std = df_col.aggregate('std')
    mins.append(agg_min)
    maxes.append(agg_max)
    means.append(agg_mean)
    stds.append(agg_std)
    xs.append(combination)
    if cnt == 3:
        label = ""
        for m in range(mode_count):
            label += " "
        xs.append(label)
        mins.append(agg_min)
        maxes.append(agg_min)
        means.append(agg_min)
        stds.append(0)
    print_col_stats(combination, col_fp, df)
    print_col_stats(combination, col_dns_fp, df)
    print_col_stats(combination, col_act_1, df)
    print_col_stats(combination, col_dns_act_1, df)
    print_col_stats(combination, col_act_2, df)
    print_col_stats(combination, col_dns_act_2, df)


# construct some data like what you have:
mins = np.asarray(mins)
maxes = np.asarray(maxes)
means = np.asarray(means)
stds = np.asarray(stds)


plt.figure(figsize=(10, 6))
# plt.plot(xs, np.ones(16) * mins.max())
plt.errorbar(xs, means, stds, fmt='ok', lw=3)
plt.errorbar(xs, means, [means - mins, maxes - means],
             fmt='.k', ecolor='gray', lw=1)

#plt.xlim(-1, 12)

plt.show()


