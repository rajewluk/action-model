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


def get_csv_files(folder_names, root_folder=None):
    csv_files = []
    if folder_names is None:
        folder_names = ['']

    for folder in folder_names:
        location = folder
        if root_folder is not None:
            location = f"{root_folder}/{folder}"
        p = Path("results/{}".format(location))
        p_str = str(p)
        for r, d, f in walk("results"):
            if r.startswith(p_str):
                for file in f:
                    if '.csv' in file:
                        file_path = Path(r)
                        csv_files.append(file_path.joinpath(file))
    return csv_files


def filter_csv_files(files, sequence_numbers):
    new_files = list()
    file_dict = dict()
    for file in files:
        folder = str(file.parent)
        if folder not in file_dict:
            file_dict[folder] = {"index": 10000000, "files": list()}
        file_name = file.name[:-4]
        seq_n = int(file_name[file_name.rfind('_') + 1:])
        if sequence_numbers is not None:
            for i in range(len(sequence_numbers)):
                sequence_numbers_sect = sequence_numbers[i]
                if seq_n in sequence_numbers_sect:
                    if file_dict[folder]["index"] > i:
                        file_dict[folder] = {"index": i, "files": list()}
                        file_dict[folder]["files"].append(file)
                        break
                    elif file_dict[folder]["index"] == i:
                        file_dict[folder]["files"].append(file)
                        break

    for key in file_dict:
        for file in file_dict[key]["files"]:
            new_files.append(file)

    return new_files

