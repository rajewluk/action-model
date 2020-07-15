from pathlib import Path
from os import walk
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import configparser

plan_config = configparser.ConfigParser()
plan_config.read_file(open('plan.cfg'))

res_folder_name = plan_config['Simulation'].get('results-folder')
files = []
p = Path("results/{}".format(res_folder_name))
for r, d, f in walk("results"):
    if p.name == Path(r).name:
        for file in f:
            if '.csv' in file:
                files.append(p.joinpath(file))

series = {}
plot_y_column = plan_config['Plot'].get('y-column')
plot_series_label = plan_config['Plot'].get('series-label')
series_size = 0
for file in files:
    df = pd.read_csv(file, delimiter=";", index_col='Iter', decimal=",")
    file_name = file.name[:-4]
    seq_n = file_name[file_name.rfind('_') + 1:]
    series[plot_series_label.format(seq_n)] = df[plot_y_column]
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
