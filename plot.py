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


def combine_func(left, right):
    return (left - right)/right*100


def print_statistic_row(series_name, agg_mean, agg_median, agg_min, agg_max):
    print('| {:>21} | {:10.2f} | {:10.2f} | {:10.2f} | {:10.2f} |'.format(series_name, agg_mean, agg_median, agg_min,
                                                                          agg_max))


def select_best_result(results):
    if len(results) == 0:
        return None
    elif len(results) == 1:
        return results[0]["values"]
    else:
        return results[-1]["values"]


def print_line():
    print("-----------------------------------------------------------------------------")


def compare_results(title, res_1, res_2, res_3=None, total=False):
    res = [0, 0, 0, 0]
    for i in range(4):
        res[i] = res_1[i] - res_2[i]
        if res_3 is not None:
            if total:
                res[i] = res[i] / (res_2[i] - res_3[i]) * 100
            else:
                res[i] = res[i] / (res_3[i]) * 100
    print_statistic_row(title, res[0], res[1], res[2], res[3])
    print_line()


def print_statistics(statistics):
    joint = list()
    cl = list()
    no_cl = list()
    for key in statistics:
        if statistics[key]["mode"] == "joint":
            joint.append(statistics[key])
        elif statistics[key]["mode"] == "cl":
            cl.append(statistics[key])
        elif statistics[key]["mode"] == "no_cl":
            no_cl.append(statistics[key])
    joint = select_best_result(joint)
    cl = select_best_result(cl)
    no_cl = select_best_result(no_cl)
    if joint is not None:
        if no_cl is not None:
            compare_results("joint-no_cl", joint, no_cl)
            compare_results("% joint-no_cl / no_cl", joint, no_cl, no_cl)
        if cl is not None:
            compare_results("joint-cl", joint, cl)
            compare_results("% joint-cl / cl", joint, cl, cl)
            if no_cl is not None:
                compare_results("cl-no_cl", cl, no_cl)
                compare_results("% cl-no_cl / no_cl", cl, no_cl, no_cl)
                compare_results("% joint-cl / no_cl", joint, cl, no_cl)
                compare_results("% joint%-cl% / cl%", joint, cl, no_cl, True)


def gen_series_name(pattern, mode_name, seq_n_val, col_name):
    series_name_res = pattern.format(mode=mode_name, seq_n = seq_n_val, col_1=col_name.split('_')[0],
                                     col_2=col_name.split('_')[1], col_3=col_name.split('_')[2]).lower()

    series_name_res = series_name_res.replace("joint", "JRAS").replace("no_cl", "CRA").replace("cl", "CLO")
    return series_name_res


plan_config = configparser.ConfigParser()
plan_config.read_file(open('plan.cfg'))

result_folders = plan_config['Plot'].get('result-sets')
result_root = plan_config['Plot'].get('result-root')
res_folder_name = plan_config['Simulation'].get('results-folder')
if result_folders is not None:
    tmp_values = ast.literal_eval(result_folders)
    if len(tmp_values) == 0:
        result_folders = [res_folder_name]
    else:
        result_folders = tmp_values
else:
    result_folders = [res_folder_name]

files = get_csv_files(result_folders, result_root)
diff_files = []
diff_folder = plan_config['Plot'].get('result-diff-set')
if diff_folder is not None:
    diff_files = get_csv_files([diff_folder], result_root)

plot_y_columns = plan_config['Plot'].get('y-columns')
plot_y_columns = ast.literal_eval(plot_y_columns)
plot_series_labels = plan_config['Plot'].get('series-labels')
plot_series_labels = ast.literal_eval(plot_series_labels)

sequence_numbers = plan_config['Plot'].get('sequence-numbers')
if sequence_numbers is not None:
    tmp_values = ast.literal_eval(sequence_numbers)
    if len(tmp_values) == 0:
        sequence_numbers = None
    else:
        sequence_numbers = tmp_values

main_col_name = plot_y_columns[0][0]
y_limits = [sys.maxsize * 2 + 1, -(sys.maxsize * 2 + 1)]
diff_df = None
if len(diff_files) > 0:
    diff_df = pd.read_csv(diff_files[0], delimiter=";", index_col='Iter', decimal=",")
    print("{:>15} ({} - {}/{}) * 100%".format('', main_col_name, main_col_name, main_col_name))
else:
    print("{:>50}".format(main_col_name))

print_line()
print('| {:>21} | {:>5}  AVG | {:>5}  MED | {:>5}  MIN | {:>5}  MAX |'.format('SERIES', '', '', '', ''))
print_line()


series_size = 0
series = []
chart_size = len(plot_y_columns)
for i in range(chart_size):
    series.append({})

files = filter_csv_files(files, sequence_numbers)

statistics = dict()
for file in files:
    df = pd.read_csv(file, delimiter=";", index_col='Iter', decimal=",")
    file_name = file.name[:-4]
    seq_n = int(file_name[file_name.rfind('_') + 1:])
    mode = file.parts[-2][file.parts[-2].rfind('-') + 1:]
    series_name = gen_series_name(plot_series_labels[0], mode, seq_n, main_col_name)
    df_col = df[main_col_name]
    if diff_df is not None:
        df_col = df_col.combine(diff_df[main_col_name], combine_func)
    agg_mean = df_col.aggregate('mean')
    agg_median = df_col.aggregate('median')
    agg_min = df_col.aggregate('min')
    agg_max = df_col.aggregate('max')
    statistics[series_name] = {"mode": mode, "values": [agg_mean, agg_median, agg_min, agg_max]}
    print_statistic_row(series_name, agg_mean, agg_median, agg_min, agg_max)
    print_line()
    y_limits[0] = np.min([y_limits[0], df_col.min()])
    y_limits[1] = np.max([y_limits[1], df_col.max()])
    series[0][series_name] = df_col
    for i in range(1, chart_size):
        for col_name in plot_y_columns[i]:
            series_col_name = col_name
            series_name = gen_series_name(plot_series_labels[i], mode, seq_n, series_col_name)
            series[i][series_name] = df[series_col_name]
    series_size = df_col.count()

print_statistics(statistics)

x_limits = [1, series_size]

plot_title = plan_config['Plot'].get('title')
plot_x_title = plan_config['Plot'].get('x-title')
plot_y_titles = plan_config['Plot'].get('y-titles')
if plot_y_titles is not None:
    plot_y_titles = ast.literal_eval(plot_y_titles)
plot_x_limit = plan_config['Plot'].get('x-limit')
plot_y_limit = plan_config['Plot'].get('y-limit')

if plot_x_limit is not None:
    tmp_values = ast.literal_eval(plot_x_limit)
    if len(tmp_values) == 2:
        x_limits = tmp_values

if plot_y_limit is not None:
    tmp_values = ast.literal_eval(plot_y_limit)
    if len(tmp_values) == 2:
        y_limits = tmp_values

fig = plt.figure(figsize=(10, 6))
if chart_size == 1:
    ax = [plt.subplot(1, 1, 1)]
    #plt.grid()
else:
    ax = list()
    ax.append(plt.subplot(2, 1, 1))
    #plt.grid()
    plot_size = math.pow(2, chart_size - 1)
    for i in range(1, chart_size):
        ax.append(plt.subplot(plot_size, 1, plot_size / 2 + i))

fig.set_size_inches(10, 6)
ax[0].axis([x_limits[0], x_limits[1], y_limits[0], y_limits[1]])
ax[0].set_title(plot_title)
ax[chart_size - 1].set_xlabel(plot_x_title)
x = range(1, series_size + 1)
for i in range(chart_size):
    for serie in series[i]:
        ax[i].plot(x, series[i][serie], label=serie)
    ax[i].set_xlim(x_limits)
    if plot_y_titles is not None:
        ax[i].set_ylabel(plot_y_titles[i])

# plt.yscale('log')
for i in range(len(plot_series_labels)):
    if plot_series_labels[i] != "":
        ax[i].legend()

slider_size = 0
if plan_config['Plot'].get('slider-size') is not None:
    slider_size = int(plan_config['Plot'].get('slider-size'))
prev_pos = [0]
s_pos = None


def update(val):
    pos = math.floor(s_pos.val)
    if pos == prev_pos[0]:
        return
    prev_pos[0] = pos
    x_min = pos - slider_size
    x_max = pos + slider_size
    if x_min < x_limits[0]:
        x_min = x_limits[0]
        x_max = np.min([x_min + slider_size*2, x_limits[1]])
    elif x_max > x_limits[1]:
        x_max = x_limits[1]
        x_min = np.max([x_max - slider_size * 2, x_limits[0]])

    for axis in ax:
        axis.set_xlim([x_min, x_max])
    fig.canvas.draw_idle()


if slider_size > 0:
    ax_pos = plt.axes([0.2, 0.1, 0.65, 0.03])
    s_pos = Slider(ax_pos, '', x_limits[0], x_limits[1], x_limits[0], valfmt='%1.0f')
    s_pos.on_changed(update)

plt.show()
