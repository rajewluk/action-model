from pathlib import Path
from shutil import copyfile
from shutil import rmtree
import configparser
import minizinc
import asyncio
import random
import decimal
import time
import math
import ast
import csv
import numpy as np
import sys
import datetime


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def print_config(config):
    for section in dict(config.items()).keys():
        if len(config.items(section)) > 0:
            print("[{}]".format(section))
            print(dict(config.items(section)))


def load_config(config_name, main_config):
    new_config = configparser.ConfigParser(comment_prefixes='/', allow_no_value=True)
    try:
        new_config.read_file(open(config_name))
    except FileNotFoundError:
        print("NO CONFIGURATION FILE {}. SET TO DEFAULTS. PLEASE VERIFY".format(config_name))
        if main_config:
            new_config['Slots'] = {}
            new_config['Slots']['multiplier'] = '3'
            new_config['Slots']['iterations'] = '5'
            new_config['Slots']['max-remaining'] = '2'
            new_config['TrafficDemands'] = {}
            new_config['TrafficDemands']['fixed'] = 'yes'
            new_config['TrafficDemands']['level'] = '100'
            new_config['TrafficDemands']['change-percent'] = '10'
            new_config['TrafficDemands']['min-demand'] = '5'
            new_config['TrafficDemands']['trend-threshold'] = '4'
            new_config['ActionDemands'] = {}
            new_config['ActionDemands']['fixed'] = 'yes'
            new_config['ActionDemands']['level'] = '15'
            new_config['ActionDemands']['change-percent'] = '10'
            new_config['ActionDemands']['trend-threshold'] = '4'
            new_config['Allocation'] = {}
            new_config['Allocation']['optimize-resources'] = 'yes'
            new_config['Allocation']['service-alternating'] = 'no'
            new_config['Allocation']['allocate-actions'] = 'yes'
            new_config['Allocation']['action-feedback-delay'] = '0'
            new_config['Allocation']['initial-demand-level'] = '80'
            new_config['Allocation']['seed-base'] = '0'
            new_config['Allocation']['joint-actions-allocation'] = 'no'
            new_config['Statistics'] = {}
            new_config['Statistics']['max-sla'] = '10000'
        else:
            new_config['Simulation'] = {}
            new_config['Simulation']['results-folder'] = 'example'
            new_config['Simulation']['append-results'] = 'yes'
            new_config['Simulation']['completed'] = 'no'
            new_config['Plot'] = {}
            new_config['Plot']['y-columns'] = "['S1_AVG_SLA']"
            new_config['Plot']['title'] = "Average SLA"
            new_config['Plot']['x-title'] = "Iteration"
            new_config['Plot']['series-labels'] = "level {}"

            new_config['Slots'] = {}
            new_config['TrafficDemands'] = {}
            new_config['TrafficDemands']['fixed'] = 'yes'
            new_config['ActionDemands'] = {}
            new_config['ActionDemands']['fixed'] = 'yes'
            new_config['Allocation'] = {}
        with open(config_name, 'w') as configfile:
            new_config.write(configfile)
        exit()

    return new_config


def get_last_slot(tab, dimensions, level=1):
    if level == 1:
        tab = tab.copy()
    if level == dimensions:
        return tab[-1]
    else:
        for i in range(len(tab)):
            tab[i] = get_last_slot(tab[i], dimensions, level + 1)
        return tab


def remove_first_slot(tab, dimensions, level=1):
    if level == 1:
        tab = tab.copy()
    if level == dimensions:
        return tab[1:]
    else:
        for i in range(len(tab)):
            tab[i] = remove_first_slot(tab[i], dimensions, level + 1)
        return tab


def localize_floats(row):
    return [
        str(el).replace('.', ',') if isinstance(el, float) else el
        for el in row
    ]


decimal_ctx = decimal.Context()
decimal_ctx.prec = 20


def non_exp_repr(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = decimal_ctx.create_decimal(repr(f))
    return format(d1, 'f')


def round_sig(number, sig):
    if number == 0:
        return 0
    else:
        sig_number = sig - 1
        counter = -1
        value = number
        correction = 0
        str_val = non_exp_repr(value)
        if str_val.count('.'):
            correction += 1
            if str_val.count('-'):
                correction += 1
            value = 0
            while sig_number < sig and value != number:
                counter += 1
                value = round(number, counter)
                str_val = non_exp_repr(value)
                sig_number = len(str_val) - correction - str_val.count('0')
                # print("{} - {}".format(value, sig_number))
        return value


def get_cl_action_feedback(iteration, action_feedback_delay, action_demand_history, allocate_actions):
    action_feedback = [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                       [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]

    if 0 < action_feedback_delay < iteration and not allocate_actions:
        reference_data = action_demand_history[iteration - 1 - action_feedback_delay]['initialActionCounter']
        for s in range(2):
            for l in range(5):
                for f in range(2):
                    action_feedback[s][l][f] = reference_data[s][l][f][0] + reference_data[s][l][f][1]

    return action_feedback


def get_action_arrival_times(iterations, action_slots_number, lambdas, seed_base):
    random.seed(seed_base + 1000)
    action_arrival_times = [[[[None, None], [None, None]], [[None, None], [None, None]], [[None, None], [None, None]], [[None, None], [None, None]], [[None, None], [None, None]]],
                     [[[None, None], [None, None]], [[None, None], [None, None]], [[None, None], [None, None]], [[None, None], [None, None]], [[None, None], [None, None]]]]
    for s in range(2):
        for f in range(2):
            for l in range(5):
                for a in range(2):
                    action_arrival_times[s][l][f][a] = list()
                    action_arrival_times[s][l][f][a].append([0, 0])
                    last_arrival_time = 0
                    while True:
                        time_offset = -math.log(1 - random.random()) / lambdas[a]
                        last_arrival_time = math.ceil(last_arrival_time + time_offset)
                        iteration = math.ceil(last_arrival_time / action_slots_number)
                        if iteration <= iterations:
                            action_arrival_times[s][l][f][a].append([iteration,
                                                                     last_arrival_time - (iteration - 1)*action_slots_number])
                        else:
                            break
    return action_arrival_times


def get_initial_action_information(iteration, joint_actions_allocation, initial_function_placement,
                                   total_action_counter, action_slots_number, action_demand_level,
                                   change_action_demands, action_trend_threshold, action_change_percent,
                                   prev_action_information, action_arrival_time, iterations, seed_base):
    new_procedure = False
    action_demand = [[[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                     [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]]
    initial_action_counter = [[[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                              [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]]

    result = {
        "actionDemands": action_demand,
        "initialActionCounter": initial_action_counter
    }

    max_break_counter = 100
    random.seed(seed_base + iteration)

    new_demand = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    if new_procedure:
        if joint_actions_allocation:
            print("Action Demands")
        else:
            print("Action Schedule")
            for s in range(2):
                for f in range(2):
                    for l in range(5):
                        for a in range(2):
                            arrival_info = 0
                            for arrival in action_arrival_time[s][l][f][a]:
                                if arrival[0] == iteration:
                                    batch_size = action_demand_level * initial_function_placement[s][l][f] * \
                                                 math.exp(random.randint(0, 1000) / 151) / 75100 / action_slots_number
                                    arrival_info = arrival_info + batch_size
                                if arrival[0] > iteration:
                                    break
                            arrival_info = math.ceil(arrival_info)
                            if iteration > 1:
                                prev_required_demand = prev_action_information["initialActionCounter"][s][l][f][a]
                                diff = prev_required_demand - math.floor(
                                    total_action_counter[s][l][f][a] / action_slots_number)
                                # required_demand += diff
                                if diff > 0:
                                    print("ACT {} Differ[{},{}]) = {}".format(a, s, f, diff))
                                    arrival_info = arrival_info + diff
                            final_batch_size = min([initial_function_placement[s][l][f], arrival_info])
                            if final_batch_size != arrival_info:
                                print("Requested batch size exceeds the function limit")
                            initial_action_counter[s][l][f][a] = final_batch_size
                            new_demand[s][f][a] = new_demand[s][f][a] + final_batch_size
            for s in range(2):
                print("[{}] Service {} Action Demands: [[{},{}], [{},{}]]".format(iteration, s,
                                                                                  new_demand[s][
                                                                                      0][0] * action_slots_number,
                                                                                  new_demand[s][
                                                                                      1][0] * action_slots_number,
                                                                                  new_demand[s][
                                                                                      0][1] * action_slots_number,
                                                                                  new_demand[s][
                                                                                      1][1] * action_slots_number))
    else:
        if joint_actions_allocation:
            print("Action Demands")
            a = 1
            for s in range(2):
                for f in range(2):
                    required_demand = action_demand_level * (f + 1) * action_slots_number  # twice the amount for DNS
                    if iteration > 1:
                        diff = required_demand
                        for l in range(5):
                            diff = diff - total_action_counter[s][l][f][a]
                        required_demand += diff
                    counter = 0
                    while new_demand[s][f][a] < required_demand:
                        counter += 1
                        if counter > max_break_counter:
                            raise Exception("Cannot allocate requested action demand {}".format(required_demand))
                        l = random.randint(0, 4)
                        if action_demand[s][l][f][a] < 100000: #(initial_function_placement[s][l][f]*action_slots_number):
                            action_demand[s][l][f][a] += 1
                            new_demand[s][f][a] += 1
                            counter = 0

            print("Service {} Action Demands: {}".format(0, action_demand[0]))
            print("Service {} Action Demands: {}".format(1, action_demand[1]))
            print("Action {} Demands: {}".format(0, new_demand[0]))
            print("Action {} Demands: {}".format(1, new_demand[1]))

        else:
            print("Action Schedule")

            threshold_indicator = iteration % action_trend_threshold
            if threshold_indicator == 0:
                threshold_indicator = action_trend_threshold
            step_indicator = math.floor((iteration - 1) / action_trend_threshold) + 1
            direction_indicator = 1
            if step_indicator % 2 == 0:
                direction_indicator = -1
            a = 1
            for s in range(2):
                for f in range(2):
                    total_placement = 0
                    for l in range(5):
                        total_placement += initial_function_placement[s][l][f]
                    # required_demand = action_demand_level * 0.01  for % change
                    required_demand = action_demand_level * (f + 1)  # twice the amount for DNS
                    if change_action_demands and iteration > 1:
                        step = required_demand * action_change_percent / 100.0
                        change = direction_indicator * (threshold_indicator - 1) * step
                        if step_indicator % 2 == 0:
                            change += (action_trend_threshold - 1) * step
                        required_demand = required_demand + change
                        # required_demand = math.ceil(total_placement * required_demand)  for % change
                        required_demand = math.ceil(required_demand)
                        required_demand = required_demand + direction_indicator * random.randint(0, 1)
                        if required_demand < 0:
                            required_demand = 0
                            # raise Exception("Negative Traffic Generated")
                    else:
                        # required_demand = math.ceil(total_placement * required_demand)  for % change
                        required_demand = math.ceil(required_demand)
                    if iteration > 1:
                        total_diff = 0
                        for l in range(5):
                            total_diff += prev_action_information["initialActionCounter"][s][l][f][a] - math.floor(total_action_counter[s][l][f][a] / action_slots_number)
                        required_demand += total_diff
                        if total_diff != 0:
                            print("ACT {} Differ[{},{}]) = {}".format(a, s, f, total_diff))
                    print("ACT {} Demand[{},{}]) = {}".format(a, s, f, required_demand))
                    counter = 0
                    location_index_map = list()
                    for l in range(5):
                        for k in range(initial_function_placement[s][l][f]):
                            location_index_map.append(l)
                    if change_action_demands:
                        random.seed(seed_base + iteration)
                    else:
                        random.seed(seed_base + 1000)
                    while new_demand[s][f][a] < required_demand:
                        counter += 1
                        if counter > max_break_counter:
                            raise Exception("Cannot allocate requested action demand {}".format(required_demand))
                        l = random.choice(location_index_map)
                        if initial_action_counter[s][l][f][a] < initial_function_placement[s][l][f]:
                            initial_action_counter[s][l][f][a] += 1
                            new_demand[s][f][a] += 1
                            counter = 0

            for s in range(2):
                print("[{}] Service {} Action Demands: [[{},{}], [{},{}]]".format(iteration, s,
                                                                                  new_demand[s][
                                                                                      0][0] * action_slots_number,
                                                                                  new_demand[s][
                                                                                      1][0] * action_slots_number,
                                                                                  new_demand[s][
                                                                                      0][1] * action_slots_number,
                                                                                  new_demand[s][
                                                                                      1][1] * action_slots_number))

    test = False
    if test and iteration == 1:
        next_demand = result
        for i in range(iterations):
            if joint_actions_allocation:
                next_demand = get_initial_action_information(2 + i, joint_actions_allocation,
                                                             initial_function_placement,
                                                             next_demand["actionDemands"],
                                                             action_slots_number, action_demand_level,
                                                             change_action_demands,
                                                             action_trend_threshold, action_change_percent, next_demand,
                                                             action_arrival_time, iterations, seed_base)
            else:
                total_actions = []
                for s in range(2):
                    total_actions.append([])
                    for l in range(5):
                        total_actions[s].append([])
                        for f in range(2):
                            total_actions[s][l].append([0, 0])
                            for a in range(2):
                                total_actions[s][l][f][a] = next_demand["initialActionCounter"][s][l][f][a] * action_slots_number
                next_demand = get_initial_action_information(2 + i, joint_actions_allocation, initial_function_placement,
                                                             total_actions, action_slots_number, action_demand_level,
                                                             change_action_demands, action_trend_threshold,
                                                             action_change_percent, next_demand, action_arrival_time,
                                                             iterations, seed_base)
        exit(0)

    return result


def gen_traffic_demand_information(iteration, traffic_min_demand, traffic_demand_level, change_traffic_demands,
                                   traffic_trend_threshold, traffic_change_percent, seed_base):
    # traffic_demands = [[40, 15, 20, 5, 25], [15, 20, 30, 15, 20]]
    # return traffic_demands
    traffic_demands = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    random.seed(seed_base + 1)

    min_demand = traffic_min_demand
    for s in range(2):
        to_allocate = traffic_demand_level - 5 * min_demand
        locations = [0, 1, 2, 3, 4]
        random.shuffle(locations)
        for l in range(5):
            if l == 4:
                traffic_demands[s][locations[l]] = min_demand + to_allocate
            else:
                rand_demand = random.randint(0, math.ceil(to_allocate*2/3))
                traffic_demands[s][locations[l]] = min_demand + rand_demand
                to_allocate -= rand_demand

    if change_traffic_demands and iteration > 1:
        threshold_indicator = iteration % traffic_trend_threshold
        if threshold_indicator == 0:
            threshold_indicator = traffic_trend_threshold
        step_indicator = math.floor((iteration - 1) / traffic_trend_threshold) + 1
        direction_indicator = 1
        if step_indicator % 2 == 0:
            direction_indicator = -1

        # print("{}: {}/{}/{}".format(iteration, threshold_indicator, step_indicator, direction_indicator))
        random.seed(seed_base + iteration)
        for s in range(2):
            for l in range(5):
                step = math.ceil(traffic_demands[s][l] * traffic_change_percent / 100.0)
                change = direction_indicator * (threshold_indicator - 1) * step
                if step_indicator % 2 == 0:
                    change += (traffic_trend_threshold - 1) * step
                traffic_demands[s][l] = traffic_demands[s][l] + change

                # if threshold_indicator != 1000:
                traffic_demands[s][l] = traffic_demands[s][l] + direction_indicator * random.randint(0, min([step, min_demand]))
                if traffic_demands[s][l] < 0:
                    traffic_demands[s][l] = 0
                    # raise Exception("Negative Traffic Generated")

    print("[{}] Traffic Demands: {}".format(iteration, traffic_demands))
    test = False
    if test and iteration == 1:
        for i in range(50):
            gen_traffic_demand_information(2 + i, traffic_min_demand, traffic_demand_level,
                                           change_traffic_demands, traffic_trend_threshold,
                                           traffic_change_percent, seed_base)
        exit(0)
    return traffic_demands


def gen_initial_function_placement(initial_demand_level, function_requirements):
    initial_function_placement = [[[4, 8], [4, 8], [4, 8], [4, 8], [4, 8]],
                                  [[4, 8], [4, 8], [4, 8], [4, 8], [4, 8]]]

    for s in range(len(initial_function_placement)):
        demand = 0
        vlb_count = 0
        while demand < initial_demand_level:
            vlb_count += 1
            max_loc_capacity = get_location_capacity(vlb_count, vlb_count*2, function_requirements)
            demand = max_loc_capacity * 5

        for l in range(5):
            initial_function_placement[s][l][0] = vlb_count
            initial_function_placement[s][l][1] = vlb_count * 2

    print("Initial Allocation: {}".format(initial_function_placement))
    return initial_function_placement


def get_location_capacity(vlb_plc, vdns_plc, function_requirements):
    max_loc_capacity = min([vlb_plc * function_requirements[0][2],
                            vdns_plc * function_requirements[1][2]])
    return max_loc_capacity


def gen_initial_demand_allocation(service_demands, initial_function_placement, function_requirements,
                                  reference_demand_allocation=None):
    # initial_demand_allocation = [[[16, 0, 0, 0, 0], [0, 15, 0, 0, 0], [0, 1, 16, 2, 0], [0, 0, 0, 5, 0],[0, 0, 0, 4, 16]],
    #                             [[15, 0, 0, 0, 0], [1, 16, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 0, 15, 0], [0, 0, 0, 1, 16]]]
    initial_demand_allocation = [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                                 [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]

    if reference_demand_allocation is None:
        for s in range(len(service_demands)):
            # print("Demands: {}".format(service_demands[s]))
            # print(initial_demand_allocation[s])
            loc_num = len(service_demands[s])
            demand = [0] * loc_num
            capacity = [0] * loc_num
            for l in range(loc_num):
                max_loc_capacity = get_location_capacity(initial_function_placement[s][l][0],
                                                         initial_function_placement[s][l][1], function_requirements)
                initial_demand_allocation[s][l] = [0] * loc_num
                initial_demand_allocation[s][l][l] = min([max_loc_capacity, service_demands[s][l]])
                demand[l] += initial_demand_allocation[s][l][l]
                capacity[l] = max_loc_capacity - initial_demand_allocation[s][l][l]
            for l in range(loc_num):
                for s_l in range(loc_num):
                    if capacity[l] == 0:
                        break
                    if demand[s_l] == service_demands[s][s_l]:
                        continue
                    initial_demand_allocation[s][s_l][l] = min(capacity[l], service_demands[s][s_l] - demand[s_l])
                    demand[s_l] += initial_demand_allocation[s][s_l][l]
                    capacity[l] -= initial_demand_allocation[s][s_l][l]

            print("Initial demand allocation ({}): ".format(s, initial_demand_allocation[s]))
    else:
        initial_demand_allocation = reference_demand_allocation.copy()
        for s in range(len(service_demands)):
            loc_num = len(service_demands[s])
            for l in range(loc_num):
                allocated_demand = 0
                for l2 in range(loc_num):
                    allocated_demand += reference_demand_allocation[s][l][l2]
                if allocated_demand > service_demands[s][l]:
                    for l2 in range(loc_num):
                        diff = allocated_demand - service_demands[s][l]
                        if l2 != l and initial_demand_allocation[s][l][l2] > 0:
                            new_allocation = int(np.max([0, initial_demand_allocation[s][l][l2] - diff]))
                            allocated_demand -= initial_demand_allocation[s][l][l2] - new_allocation
                            initial_demand_allocation[s][l][l2] = new_allocation
                            if allocated_demand == service_demands[s][l]:
                                break
                    if allocated_demand > service_demands[s][l]:
                        initial_demand_allocation[s][l][l] = service_demands[s][l]

    return initial_demand_allocation


def gen_result_file_name(config, sim_seq_number):
    slot_multiplier = config.getint("Slots", "multiplier")
    traffic_demand_level = config.getint("TrafficDemands", "level")
    action_demand_level = config.getint("ActionDemands", "level")
    change_traffic_demands = not config.getboolean("TrafficDemands", "fixed")
    change_action_demands = not config.getboolean("ActionDemands", "fixed")
    return "sn_{}-tr_{}_{}-ac_{}-{}-sq_{}.csv".format(slot_multiplier, change_traffic_demands, traffic_demand_level,
                                                      action_demand_level, change_action_demands, sim_seq_number).lower()


async def run_allocation(config, res_file):
    slot_multiplier = config.getint("Slots", "multiplier")
    max_iterations = config.getint("Slots", "iterations")
    max_remaining_slots = config.getint("Slots", "max-remaining")
    traffic_demand_level = config.getint("TrafficDemands", "level")
    traffic_min_demand = config.getint("TrafficDemands", "min-demand")
    traffic_trend_threshold = config.getint("TrafficDemands", "trend-threshold")
    traffic_change_percent = config.getfloat("TrafficDemands", "change-percent")
    action_demand_level = config.getint("ActionDemands", "level")
    action_trend_threshold = config.getint("ActionDemands", "trend-threshold")
    action_change_percent = config.getfloat("ActionDemands", "change-percent")
    joint_actions_allocation = config.getboolean("Allocation", "joint-actions-allocation")
    change_traffic_demands = not config.getboolean("TrafficDemands", "fixed")
    change_action_demands = not config.getboolean("ActionDemands", "fixed")
    optimize_resources = config.getboolean("Allocation", "optimize-resources")
    service_alternating = config.getboolean("Allocation", "service-alternating")
    initial_demand_level = config.getint("Allocation", "initial-demand-level")
    allocate_actions = config.getboolean("Allocation", "allocate-actions")
    action_feedback_delay = config.getint("Allocation", "action-feedback-delay")
    seed_base = config.getint("Allocation", "seed-base")
    max_sla = config.getfloat("Statistics", "max-sla")
    # Transform Model into a instance
    coin_bc = minizinc.Solver.lookup("coin-bc")
    # coin_bc = minizinc.Solver.load(Path("./config.msc"))
    # coin_bc = minizinc.Solver.load(Path("./config.msc"))

    # Create a MiniZinc model
    placement_model = minizinc.Model("./placement_model.mzn")
    placement_model.add_file("./model_data.dzn", False)

    orchestration_model = minizinc.Model("./orchestration_model.mzn")
    orchestration_model.add_file("./model_data.dzn", False)
    # print(coin_bc.output_configuration())

    target_placement = minizinc.Instance(coin_bc, placement_model)
    target_placement["NUM_TIME_SLOTS"] = 1
    target_placement["MAX_SLA"] = max_sla
    target_placement["ITERATION_NO"] = 1
    target_placement["MAX_REMAINING_SLOTS"] = max_remaining_slots
    target_placement["MINIMIZE_ALLOCATION"] = optimize_resources
    target_placement["ONE_SERVICE_OPT"] = service_alternating
    target_placement["JOINT_SCHEDULING"] = allocate_actions

    function_requirements = [[4, 8, 5],  # vLB_1
                             [2, 4, 2]]  # vDNS_1
    service_demands = gen_traffic_demand_information(1, traffic_min_demand, traffic_demand_level,
                                                     change_traffic_demands, traffic_trend_threshold,
                                                     traffic_change_percent, seed_base)
    initial_function_placement = gen_initial_function_placement(initial_demand_level, function_requirements)
    initial_demand_allocation = gen_initial_demand_allocation(service_demands, initial_function_placement,
                                                              function_requirements)
    remaining_time_slots = [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]

    target_placement["maxFunctionRequirements"] = function_requirements
    target_placement["serviceDemands"] = service_demands
    target_placement["initialDemandAllocation"] = initial_demand_allocation
    target_placement["initialFunctionPlacement"] = initial_function_placement
    target_placement["extraTimeSlots"] = remaining_time_slots

    # Solve the placement problem
    final_placement_solution = None
    async for result in target_placement.solutions(all_solutions=False, intermediate_solutions=True):
        if result.solution is None:
            print("No Solution")
            continue
        final_placement_solution = result.solution
        print("Intermediate Solution: {:.4f}".format(result.solution.allocationObjective))
    print("Final Solution: {:.4f}".format(round_sig(final_placement_solution.allocationObjective, 2)))
    print(final_placement_solution)

    targetDemandAllocations = final_placement_solution.targetDemandAllocations
    functionPlacementTarget = final_placement_solution.functionPlacementTarget
    targetSlaSatisfaction = final_placement_solution.targetSlaSatisfaction
    targetDemandGap = final_placement_solution.targetDemandGap
    targetDemandAllocationCost = final_placement_solution.targetDemandAllocationCost
    maxActionDuration = final_placement_solution.maxActionDuration

    action_time_slots = maxActionDuration * slot_multiplier
    num_time_slots = action_time_slots + 1
    action_arrival_time = get_action_arrival_times(max_iterations, action_time_slots, [0.5, 0.4], seed_base)
    iterations = 0
    avgSla = [0, 0]
    totalPlacement = [[0, 0], [0, 0]]
    totalActionCounter = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    lastActionCounterInLocations = None
    action_demand_history = list()
    roundPlacementSolution = round_sig(final_placement_solution.allocationObjective, 2)
    final_objective = roundPlacementSolution + 1
    total_start_time = time.time()
    res_writer = csv.writer(res_file, dialect="excel", delimiter=";")
    while iterations < max_iterations:  # and final_objective > roundPlacementSolution:
        start_time = time.time()
        iterations += 1
        print("{} Calculation for {} time slots".format(iterations, action_time_slots))
        orchestration_allocation = minizinc.Instance(coin_bc, orchestration_model)
        orchestration_allocation["NUM_TIME_SLOTS"] = num_time_slots
        orchestration_allocation["MAX_SLA"] = max_sla
        orchestration_allocation["ITERATION_NO"] = iterations
        orchestration_allocation["MAX_REMAINING_SLOTS"] = max_remaining_slots
        orchestration_allocation["MINIMIZE_ALLOCATION"] = optimize_resources
        orchestration_allocation["ONE_SERVICE_OPT"] = service_alternating
        orchestration_allocation["JOINT_SCHEDULING"] = allocate_actions
        orchestration_allocation["targetDemandAllocations"] = targetDemandAllocations
        orchestration_allocation["functionPlacementTarget"] = functionPlacementTarget
        orchestration_allocation["targetSlaSatisfaction"] = targetSlaSatisfaction
        orchestration_allocation["targetDemandGap"] = targetDemandGap
        orchestration_allocation["targetDemandAllocationCost"] = targetDemandAllocationCost
        orchestration_allocation["maxFunctionRequirements"] = function_requirements
        orchestration_allocation["initialFunctionPlacement"] = initial_function_placement
        service_demands = gen_traffic_demand_information(iterations, traffic_min_demand, traffic_demand_level,
                                                         change_traffic_demands, traffic_trend_threshold,
                                                         traffic_change_percent, seed_base)
        initial_demand_allocation = gen_initial_demand_allocation(service_demands, initial_function_placement,
                                                                  function_requirements, initial_demand_allocation)
        orchestration_allocation["initialDemandAllocation"] = initial_demand_allocation
        orchestration_allocation["serviceDemands"] = service_demands
        prev_action_allocation = None
        if iterations > 1:
            prev_action_allocation = action_demand_history[-1]
        action_allocation = get_initial_action_information(iterations, joint_actions_allocation,
                                                           initial_function_placement, lastActionCounterInLocations,
                                                           action_time_slots, action_demand_level,
                                                           change_action_demands, action_trend_threshold,
                                                           action_change_percent, prev_action_allocation,
                                                           action_arrival_time, max_iterations, seed_base)
        action_demand_history.append(action_allocation)
        function_reservations_for_actions = get_cl_action_feedback(iterations, action_feedback_delay,
                                                                   action_demand_history, allocate_actions)
        orchestration_allocation["functionReservationForActions"] = function_reservations_for_actions
        orchestration_allocation["initialActionCounter"] = action_allocation["initialActionCounter"]
        orchestration_allocation["actionDemands"] = action_allocation["actionDemands"]
        orchestration_allocation["extraTimeSlots"] = remaining_time_slots
        # Solve the allocation problem
        final_allocation_solution = None
        async for result in orchestration_allocation.solutions(all_solutions=False, intermediate_solutions=True,
                                                               processes=1): #, ignore_errors=True):
            if result.solution is None:
                print("No Solution")
                break
            final_allocation_solution = result.solution
            print("Intermediate Solution: {:.4f}/{:.4f} - {:.0f} seconds".format(result.solution.allocationObjective, result.solution.lastTimeSlotObjective, time.time() - start_time))

        print("Completed in: {:.0f} seconds".format(time.time() - start_time))
        print("Latest Solution: {:.4f}".format(final_allocation_solution.allocationObjective))
        final_objective = round_sig(final_allocation_solution.lastTimeSlotObjective, 2)
        print(final_allocation_solution)
        initial_demand_allocation = get_last_slot(final_allocation_solution.demandAllocations, 4)
        initial_function_placement = get_last_slot(final_allocation_solution.functionPlacement, 4)
        remaining_time_slots = final_allocation_solution.remainingSlots
        avgSla[0] += final_allocation_solution.avgSlaSatisfaction[0]
        avgSla[1] += final_allocation_solution.avgSlaSatisfaction[1]
        lastActionCounterInLocations = final_allocation_solution.totalActionCounterInLocations
        for i in range(2):
            for j in range(2):
                totalPlacement[i][j] = totalPlacement[i][j] + final_allocation_solution.totalFunctionPlacement[i][j]
                for k in range(2):
                    totalActionCounter[i][j][k] = totalActionCounter[i][j][k] + final_allocation_solution.totalActionCounter[i][j][k]

        res_writer.writerow(localize_floats(
            [iterations, round_sig(final_allocation_solution.allocationObjective, 2),
             round_sig(np.average(remove_first_slot(final_allocation_solution.slaSatisfaction[0], 2)) / max_sla, 3),
             round_sig(np.min(remove_first_slot(final_allocation_solution.slaSatisfaction[0], 2)) / max_sla, 3),
             round_sig(np.max(remove_first_slot(final_allocation_solution.slaSatisfaction[0], 2)) / max_sla, 3),
             round_sig(np.var(remove_first_slot(final_allocation_solution.slaSatisfaction[0], 2)) / max_sla, 3),
             round_sig(np.std(remove_first_slot(final_allocation_solution.slaSatisfaction[0], 2)) / max_sla, 3),
             final_allocation_solution.totalFunctionPlacement[0][0],
             final_allocation_solution.totalFunctionPlacement[0][1],
             final_allocation_solution.totalActionCounter[0][0][0],
             final_allocation_solution.totalActionCounter[0][1][0],
             final_allocation_solution.totalActionCounter[0][0][1],
             final_allocation_solution.totalActionCounter[0][1][1],
             round_sig(np.average(remove_first_slot(final_allocation_solution.slaSatisfaction[1], 2)) / max_sla, 3),  # AVG SLA
             round_sig(np.min(remove_first_slot(final_allocation_solution.slaSatisfaction[1], 2)) / max_sla, 3),  # MIN SLA
             round_sig(np.max(remove_first_slot(final_allocation_solution.slaSatisfaction[1], 2)) / max_sla, 3),  # MAX SLA
             round_sig(np.var(remove_first_slot(final_allocation_solution.slaSatisfaction[1], 2)) / max_sla, 3),  # VAR SLA
             round_sig(np.std(remove_first_slot(final_allocation_solution.slaSatisfaction[1], 2)) / max_sla, 3),  # STD SLA
             final_allocation_solution.totalFunctionPlacement[1][0],  # LB FP
             final_allocation_solution.totalFunctionPlacement[1][1],  # DNS FP
             final_allocation_solution.totalActionCounter[1][0][0],  # LB ACT 1
             final_allocation_solution.totalActionCounter[1][1][0],  # DNS ACT 1
             final_allocation_solution.totalActionCounter[1][0][1],  # LB ACT 2
             final_allocation_solution.totalActionCounter[1][1][1]  # DNS ACT 2
             ]))
        res_file.flush()

    print("Final Solution: {:.4f} after {} time slots".format(final_objective, (num_time_slots - 1) * iterations))
    print("Final Statistics:")
    print("SLA S1: {:.3f} S2: {:.3f}".format(avgSla[0] / iterations, avgSla[1] / iterations))
    print("FNP S1: [{},{}] S2: [{},{}]".format(totalPlacement[0][0], totalPlacement[0][1], totalPlacement[1][0],
                                               totalPlacement[1][1]))
    print("ACT S1: [{},{}] [{},{}] S2: [{},{}] [{},{}]".format(totalActionCounter[0][0][0], totalActionCounter[0][0][1],
                                                               totalActionCounter[0][1][0], totalActionCounter[0][1][1],
                                                               totalActionCounter[1][0][0], totalActionCounter[1][0][1],
                                                               totalActionCounter[1][1][0], totalActionCounter[1][1][1]))


def set_res_file_header(res_file):
    res_writer = csv.writer(res_file, delimiter=";")
    res_writer.writerow(['Iter', 'Result',
                         'S1_AVG_SLA',
                         'S1_MIN_SLA',
                         'S1_MAX_SLA',
                         'S1_VAR_SLA',
                         'S1_STD_SLA',
                         'S1_LB_FP',
                         'S1_DNS_FP',
                         'S1_LB_ACT_1',
                         'S1_DNS_ACT_1',
                         'S1_LB_ACT_2',
                         'S1_DNS_ACT_2',
                         'S2_AVG_SLA',
                         'S2_MIN_SLA',
                         'S2_MAX_SLA',
                         'S2_VAR_SLA',
                         'S2_STD_SLA',
                         'S2_LB_FP',
                         'S2_DNS_FP',
                         'S2_LB_ACT_1',
                         'S2_DNS_ACT_1',
                         'S2_LB_ACT_2',
                         'S2_DNS_ACT_2'
                         ])
    res_file.flush()


def run_simulation(plan_file):
    main_config = load_config('config.cfg', True)

    plan_config = load_config(plan_file, False)
    res_folder_name = Path("results/{}".format(plan_config['Simulation'].get('results-folder')))
    completed = plan_config['Simulation'].getboolean('completed')
    app_res_file = plan_config['Simulation'].getboolean('append-results')

    if completed:
        print("Simulation was already completed")
        return
    # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

    # pd.options.plotting.backend = "plotly"

    # Plot
    # fig = df.plot(x='Date', y=['AAPL.High', 'AAPL.Low'])

    # fig.show()

    values = {}
    options_size = 0
    for section in plan_config.sections():
        if "Simulation" == section or "Plot" == section:
            continue
        for key in plan_config[section]:
            try:
                main_config[section]
            except KeyError:
                print("Section '{}' not recognized in main config file".format(section))
                exit(1)
            try:
                main_config[section][key]
            except KeyError:
                print("Unrecognized '{}' parameter in section '{}' of main config file".format(key, section))
                exit(1)
            try:
                value = main_config.get(section, key)
                if value in configparser.RawConfigParser.BOOLEAN_STATES and value != '1' and value != '0':
                    value = main_config.getboolean(section, key)
                else:
                    raise ValueError()
            except ValueError:
                try:
                    value = main_config.getint(section, key)

                except ValueError:
                    try:
                        value = main_config.getfloat(section, key)
                    except ValueError:
                        value = main_config.get(section, key)
            if section not in values:
                values[section] = {}
            if not isinstance(value, bool) and isinstance(value, int) or isinstance(value, float):
                tmp_values = ast.literal_eval(plan_config[section].get(key))
                if 0 < options_size != len(tmp_values):
                    print("Mismatch of option size in the simulation plan")
                    exit(1)
                else:
                    options_size = len(tmp_values)
                values[section][key] = tmp_values
            else:
                values[section][key] = plan_config[section].get(key)

    if not Path.exists(Path("results")):
        Path.mkdir(Path("results"))

    if Path.exists(res_folder_name) and Path.is_dir(res_folder_name):
        rmtree(path=res_folder_name, ignore_errors=True)

    if not Path.exists(res_folder_name) or not Path.is_dir(res_folder_name):
        Path.mkdir(res_folder_name)

    sys.stdout = Logger("{}/output.log".format(res_folder_name))
    simulation_start_time = time.time()
    print("START: {}".format(datetime.datetime.now()))

    copyfile(plan_file, '{}/plan.cfg'.format(res_folder_name))
    copyfile('config.cfg', '{}/config.cfg'.format(res_folder_name))

    if options_size == 0:
        print("Singular simulation test")
        print(main_config)
        res_file_name = "{}/{}".format(res_folder_name, gen_result_file_name(main_config, 1))
        with open(res_file_name, 'a' if app_res_file else 'w', newline='') as res_file:
            set_res_file_header(res_file)
            asyncio.run(run_allocation(main_config, res_file))
    else:
        print("Simulation Values: {}".format(values))
        for i in range(options_size):
            for section in values:
                for key in values[section]:
                    if isinstance(values[section][key], list):
                        main_config.set(section, key, str(values[section][key][i]))
                    else:
                        main_config.set(section, key, str(values[section][key]).lower())
            res_file_name = "{}/{}".format(res_folder_name, gen_result_file_name(main_config, i + 1))
            print_config(main_config)
            with open(res_file_name, 'a' if app_res_file else 'w', newline='') as res_file:
                set_res_file_header(res_file)
                asyncio.run(run_allocation(main_config, res_file))

    plan_config['Simulation']['completed'] = 'yes'
    with open(plan_file, 'w') as configfile:
        plan_config.write(configfile)

    simulation_stop_time = time.time()
    print("STOP: {}".format(datetime.datetime.now()))
    time_diff = simulation_stop_time - simulation_start_time
    print("TIME: {:.0f} seconds, {:.0f} minutes".format(math.ceil(time_diff), math.ceil(time_diff/60.0)))


def main():
    plan_file = "plan.cfg"
    if len(sys.argv) > 1:
        plan_file = str(sys.argv[1])
    run_simulation(plan_file)


if __name__ == "__main__":
    main()

