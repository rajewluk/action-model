from pathlib import Path
from os import walk
import numpy as np
import csv


def get_output_files(root_folder):
    output_files = {}
    for r, d, f in walk(root_folder):
        for file in f:
            if '.log' in file:
                file_path = Path(r)
                if file_path.name not in output_files:
                    output_files[file_path.name] = []
                output_files[file_path.name].append(file_path.joinpath(file))

    return output_files


def get_time_lines(output_files):
    lines = {}
    for mode in output_files:
        for file in output_files[mode]:
            with open(file, 'r') as output_file:
                for last_line in output_file:
                    pass
            if mode not in lines:
                lines[mode] = []
            lines[mode].append(last_line.replace("\n", ""))

    return lines


def parse_times(lines):
    times = {}
    for mode in lines:
        for time_result in lines[mode]:
            if mode not in times:
                times[mode] = []
            # print("{} {}".format(mode, time_result))
            try:
                time_res = time_result.split(",")[1].replace(" minutes", "")
                times[mode].append(int(time_res))
            except:
                print("{}: Incomplete".format(mode))
    return times


def get_statistics(times_set):
    with open("times.csv", 'w', newline='') as res_file:
        res_writer = csv.writer(res_file, delimiter=";")
        res_writer.writerow(['Mode', 'Result', 'Var', 'Std', 'Min', 'Max'])
        res_file.flush()
        for mode in times_set:
            calc_set = np.array(times_set[mode])
            print("{:13} [{}]: {} {} {} {} {}".format(mode, len(times_set[mode]), calc_set.mean(), calc_set.var(),
                                                      calc_set.std(), calc_set.min(), calc_set.max()))
            res_writer.writerow([mode,
                                 "{:.1f}".format(np.array(calc_set.mean()).replace(".", ",")),
                                 "{:.1f}".format(np.array(calc_set.var()).replace(".", ",")),
                                 "{:.1f}".format(np.array(calc_set.std()).replace(".", ",")),
                                 "{:.1f}".format(np.array(calc_set.min()).replace(".", ",")),
                                 "{:.1f}".format(np.array(calc_set.max()).replace(".", ","))])


files = get_output_files("time_simulations")
# files = get_output_files("results/times")
time_lines = get_time_lines(files)
times_results = parse_times(time_lines)
get_statistics(times_results)
