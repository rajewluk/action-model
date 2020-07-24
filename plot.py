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


def get_csv_files(folder_names):
    csv_files = []
    for folder in folder_names:
        p = Path("results/{}".format(folder))
        for r, d, f in walk("results"):
            if p.name == Path(r).name:
                for file in f:
                    if '.csv' in file:
                        csv_files.append(p.joinpath(file))
    return csv_files


def combine_func(left, right):
    return (left - right)/right*100


plan_config = configparser.ConfigParser()
plan_config.read_file(open('plan.cfg'))

result_folders = plan_config['Plot'].get('result-sets')
res_folder_name = plan_config['Simulation'].get('results-folder')
if result_folders is not None:
    tmp_values = ast.literal_eval(result_folders)
    if len(tmp_values) == 0:
        result_folders = [res_folder_name]
    else:
        result_folders = tmp_values
else:
    result_folders = [res_folder_name]

files = get_csv_files(result_folders)
diff_files = []
diff_folder = plan_config['Plot'].get('result-diff-set')
if diff_folder is not None:
    diff_files = get_csv_files([diff_folder])

plot_y_columns = plan_config['Plot'].get('y-columns')
plot_y_columns = ast.literal_eval(plot_y_columns)
plot_series_label = plan_config['Plot'].get('series-label')

sequence_numbers = plan_config['Plot'].get('sequence-numbers')
if sequence_numbers is not None:
    tmp_values = ast.literal_eval(sequence_numbers)
    if len(tmp_values) == 0:
        sequence_numbers = None
    else:
        sequence_numbers = tmp_values

y_limits = [sys.maxsize * 2 + 1, -(sys.maxsize * 2 + 1)]
diff_df = None
if len(diff_files) > 0:
    diff_df = pd.read_csv(diff_files[0], delimiter=";", index_col='Iter', decimal=",")
    print("{:>15} ({} - {}/{}) * 100%".format('', plot_y_columns[0], plot_y_columns[0], plot_y_columns[0]))
else:
    print("{:>50}".format(plot_y_columns[0]))

print('| {:>20} | {:>5}  AVG | {:>5}  MED | {:>5}  MIN | {:>5}  MAX |'.format('SERIES', '', '', '', ''))
series_size = 0
series = []
chart_size = len(plot_y_columns)
for i in range(chart_size):
    series.append({})
for file in files:
    df = pd.read_csv(file, delimiter=";", index_col='Iter', decimal=",")
    file_name = file.name[:-4]
    seq_n = int(file_name[file_name.rfind('_') + 1:])
    if sequence_numbers is not None and seq_n not in sequence_numbers:
        continue
    mode = file.parts[1][file.parts[1].rfind('-') + 1:]
    series_name = plot_series_label.format(mode, seq_n)
    df_col = df[plot_y_columns[0]]
    if diff_df is not None:
        df_col = df_col.combine(diff_df[plot_y_columns[0]], combine_func)
    agg_mean = df_col.aggregate('mean')
    agg_avg = df_col.aggregate('median')
    agg_min = df_col.aggregate('min')
    agg_max = df_col.aggregate('max')
    print('| {:>20} | {:10.2f} | {:10.2f} | {:10.2f} | {:10.2f} |'.format(series_name, agg_mean, agg_avg, agg_min, agg_max))
    y_limits[0] = np.min([y_limits[0], df_col.min()])
    y_limits[1] = np.max([y_limits[1], df_col.max()])
    series[0][series_name] = df_col
    for i in range(1, chart_size):
        series[i][series_name] = df[plot_y_columns[i]]
    series_size = df_col.count()

x_limits = [1, series_size]

plot_title = plan_config['Plot'].get('title')
plot_x_title = plan_config['Plot'].get('x-title')
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

# plt.yscale('log')
if plot_series_label != "":
    ax[0].legend()

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
