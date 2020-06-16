from pathlib import Path

import minizinc
import asyncio
import random
import decimal
import time
import math

decimal_ctx = decimal.Context()
decimal_ctx.prec = 20


def get_last_slot(tab, dimensions, level=1):
    if level == dimensions:
        return tab[-1]
    else:
        for i in range(len(tab)):
            tab[i] = get_last_slot(tab[i], dimensions, level + 1)
        return tab


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
                prev_value = value
                counter += 1
                value = round(number, counter)
                str_val = non_exp_repr(value)
                sig_number = len(str_val) - correction - str_val.count('0')
                # print("{} - {}".format(value, sig_number))
        return value


def get_initial_action_information(iteration, joint_actions_allocation, initial_function_placement,
                                   total_action_counter, action_slots_number, action_demand_level,
                                   change_action_demands):
    action_demand = [[[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                              [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]]
    initial_action_counter = [[[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                              [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]]

    result = {
        "actionDemands": action_demand,
        "initialActionCounter": initial_action_counter
    }

    max_break_counter = 100

    if joint_actions_allocation:
        print("Action Demands")
        new_demand = [[0, 0], [0, 0]]
        random.seed(iteration)
        for s in range(2):
            for f in range(2):
                required_demand = action_demand_level * (f + 1) * action_slots_number  # twice the amount for DNS
                if iteration > 1:
                    diff = required_demand - total_action_counter[s][f]
                    required_demand += diff
                print("ACT Demand[{},{}]) = {}".format(s, f, required_demand))
                counter = 0
                while new_demand[s][f] < required_demand:
                    counter += 1
                    if counter > max_break_counter:
                        raise Exception("Cannot allocate requested action demand {}".format(required_demand))
                    l = random.randint(0, 4)
                    a = 1
                    if action_demand[s][l][f][a] < 100000:  # (initial_function_placement[s][l][f] * action_slots_number):
                        action_demand[s][l][f][a] += 1
                        new_demand[s][f] += 1
                        counter = 0

        print("Total Demand: [[{},{}], [{},{}]]".format(new_demand[0][0], new_demand[0][1], new_demand[1][0], new_demand[1][1]))

    else:
        print("Action Schedule")
        new_demand = [[0, 0], [0, 0]]
        random.seed(iteration)
        for s in range(2):
            for f in range(2):
                required_demand = action_demand_level * (f + 1)  # twice the amount for DNS
                if iteration > 1:
                    diff = required_demand - math.floor(total_action_counter[s][f] / action_slots_number)
                    required_demand += diff
                print("ACT Demand[{},{}]) = {}".format(s, f, required_demand))
                counter = 0
                while new_demand[s][f] < required_demand:
                    counter += 1
                    if counter > max_break_counter:
                        raise Exception("Cannot allocate requested action demand {}".format(required_demand))
                    l = random.randint(0, 4)
                    a = 1
                    if initial_action_counter[s][l][f][a] < initial_function_placement[s][l][f]:
                        initial_action_counter[s][l][f][a] += 1
                        new_demand[s][f] += 1
                        counter = 0

        print("Total Demand: [[{},{}], [{},{}]]".format(new_demand[0][0] * action_slots_number,
                                                        new_demand[0][1] * action_slots_number,
                                                        new_demand[1][0] * action_slots_number,
                                                        new_demand[1][1] * action_slots_number))

    return result


def gen_traffic_demand_information(iteration, initial_function_placement, traffic_demand_level, change_traffic_demands):
    traffic_demands = [[40, 15, 20, 5, 25], [15, 20, 30, 15, 20]]

    return traffic_demands


async def run_allocation(slot_multiplier, max_iterations, demand_level, joint_actions_allocation,
                         change_traffic_demands, change_action_demands):
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
    target_placement["ITERATION_NO"] = 1
    initial_demand_allocation = [[[16, 0, 0, 0, 0], [0, 15, 0, 0, 0], [0, 1, 16, 2, 0], [0, 0, 0, 5, 0],[0, 0, 0, 4, 16]],
                                 [[15, 0, 0, 0, 0], [1, 16, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 0, 15, 0], [0, 0, 0, 1, 16]]]
    initial_function_placement = [[[4, 8], [4, 8], [4, 8], [4, 8], [4, 8]], [[4, 8], [4, 8], [4, 8], [4, 8], [4, 8]]]
    remaining_time_slots = [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]
    service_demands = gen_traffic_demand_information(1, initial_function_placement, demand_level,
                                                     change_traffic_demands)
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
    final_objective = 1
    iterations = 0
    avgSla = [0, 0]
    totalPlacement = [[0, 0], [0, 0]]
    totalActionCounter = [[0, 0], [0, 0]]
    lastActionCounter = [[0, 0], [0, 0]]
    roundPlacementSolution = round_sig(final_placement_solution.allocationObjective, 2)
    total_start_time = time.time()
    while iterations <= max_iterations and final_objective > roundPlacementSolution:
        start_time = time.time()
        iterations += 1
        print("{} Calculation for {} time slots".format(iterations, action_time_slots))
        orchestration_allocation = minizinc.Instance(coin_bc, orchestration_model)
        orchestration_allocation["NUM_TIME_SLOTS"] = num_time_slots
        orchestration_allocation["ITERATION_NO"] = iterations
        orchestration_allocation["targetDemandAllocations"] = targetDemandAllocations
        orchestration_allocation["functionPlacementTarget"] = functionPlacementTarget
        orchestration_allocation["targetSlaSatisfaction"] = targetSlaSatisfaction
        orchestration_allocation["targetDemandGap"] = targetDemandGap
        orchestration_allocation["targetDemandAllocationCost"] = targetDemandAllocationCost
        orchestration_allocation["initialDemandAllocation"] = initial_demand_allocation
        orchestration_allocation["initialFunctionPlacement"] = initial_function_placement
        service_demands = gen_traffic_demand_information(iterations, initial_function_placement, demand_level,
                                                         change_traffic_demands)
        orchestration_allocation["serviceDemands"] = service_demands
        action_allocation = get_initial_action_information(iterations, joint_actions_allocation,
                                                           initial_function_placement, lastActionCounter,
                                                           action_time_slots, demand_level, change_action_demands)
        orchestration_allocation["initialActionCounter"] = action_allocation["initialActionCounter"]
        orchestration_allocation["actionDemands"] = action_allocation["actionDemands"]
        orchestration_allocation["extraTimeSlots"] = remaining_time_slots
        # Solve the allocation problem
        final_allocation_solution = None
        async for result in orchestration_allocation.solutions(all_solutions=False, intermediate_solutions=True,
                                                               processes=1): #, ignore_errors=True):
            if result.solution is None:
                print("No Solution")
                continue
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
        for i in range(2):
            for j in range(2):
                totalPlacement[i][j] = totalPlacement[i][j] + final_allocation_solution.totalFunctionPlacement[i][j]
                totalActionCounter[i][j] = totalActionCounter[i][j] + final_allocation_solution.totalActionCounter[i][j]
                lastActionCounter[i][j] = final_allocation_solution.totalActionCounter[i][j]

    print("Final Solution: {:.4f} after {} time slots".format(final_objective, (num_time_slots - 1) * iterations))
    print("SLA S1: {:.3f} S2: {:.3f}".format(avgSla[0] / iterations, avgSla[1] / iterations))
    print("FNP S1: [{},{}] S2: [{},{}]".format(totalPlacement[0][0], totalPlacement[0][1], totalPlacement[1][0],
                                               totalPlacement[1][1]))
    print("ACT S1: [{},{}] S2: [{},{}]".format(totalActionCounter[0][0], totalActionCounter[0][1],
                                               totalActionCounter[1][0],
                                               totalActionCounter[1][1]))


# print(get_last_slot([[[2, 3], [1, 6], [0, 4]], [[-2, 5], [0, -3], [2, 10]]], 3))
# print(round_sig(-3454324234, 20))

asyncio.run(run_allocation(3, 4, 15, False, False, False))
