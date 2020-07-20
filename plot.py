from pathlib import Path
from os import walk
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import configparser
import ast

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
files = []
for folder in result_folders:
    p = Path("results/{}".format(folder))
    for r, d, f in walk("results"):
        if p.name == Path(r).name:
            for file in f:
                if '.csv' in file:
                    files.append(p.joinpath(file))

series = {}
plot_y_column = plan_config['Plot'].get('y-column')
plot_series_label = plan_config['Plot'].get('series-label')
sequence_numbers = plan_config['Plot'].get('sequence-numbers')
if sequence_numbers is not None:
    tmp_values = ast.literal_eval(sequence_numbers)
    if len(tmp_values) == 0:
        sequence_numbers = None
    else:
        sequence_numbers = tmp_values
series_size = 0
for file in files:
    df = pd.read_csv(file, delimiter=";", index_col='Iter', decimal=",")
    file_name = file.name[:-4]
    seq_n = int(file_name[file_name.rfind('_') + 1:])
    if sequence_numbers is not None and seq_n not in sequence_numbers:
        continue
    mode = file.parts[1][file.parts[1].rfind('-') + 1:]
    series_name = plot_series_label.format(mode, seq_n)
    series[series_name] = df[plot_y_column]
    series_size = df[plot_y_column].count()

plot_title = plan_config['Plot'].get('title')
plot_x_title = plan_config['Plot'].get('x-title')

plt.figure(figsize=(10, 6))
plt.title(plot_title)
plt.xlabel(plot_x_title)
plt.title(plot_title)
x = range(1, series_size + 1)
for serie in series:
    plt.plot(x, series[serie], label=serie)

#plt.yscale('log')
if plot_series_label != "":
    plt.legend()
plt.grid()
plt.show()
