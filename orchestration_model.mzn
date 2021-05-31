include "common_declarations.mzn";

% how much capacity for demands we would have in locations - for THR Capability only
% how we need to change placement of functions to satisfy demands within location limits for each time slot
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE] of 0..100: functionPlacementTarget;% = array3d(SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, [7, 17, 3, 7, 4, 10, 1, 3, 5, 12, 3, 8, 4, 10, 6, 15, 3, 8, 4, 10]);

% total optimal THR demand in each location
array[SERVICE_INSTANCES, LOCATIONS, LOCATIONS] of 0..max([locationCapacity[l, THR] | l in LOCATIONS]): targetDemandAllocations;% = array3d(SERVICE_INSTANCES, LOCATIONS, LOCATIONS, [3400, 50, 200, 11, 147, 0, 1350, 0, 0, 0, 0, 0, 1800, 0, 0, 0, 0, 0, 489, 3, 0, 0, 0, 0, 2250, 1500, 0, 0, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 3000, 0, 0, 0, 0, 0, 1500, 0, 0, 0, 0, 0, 2000]);

% avarage percentage of traffic satisfied proportional to the demand
array[SERVICE_INSTANCES, LOCATIONS] of 0.0..MAX_SLA: targetSlaSatisfaction;% = array2d(SERVICE_INSTANCES, LOCATIONS, [9000.0, 9000.0, 9000.0, 9813.75, 9000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0]);

%how much of the demand is not satisfied - we should minimize it
array[SERVICE_INSTANCES, LOCATIONS] of 0..max([serviceDemands[i, l] | i in SERVICE_INSTANCES, l in LOCATIONS]): targetDemandGap;% = array2d(SERVICE_INSTANCES, LOCATIONS, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

%what is the cost of the allocation of demand - on distance basis
array[SERVICE_INSTANCES, LOCATIONS] of 0.0..MAX_DISTANCE: targetDemandAllocationCost;% = array2d(SERVICE_INSTANCES, LOCATIONS, [40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

% the amount of functions reserved for
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE] of 0..100: functionReservationForActions;

% initial allocation of actions
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, CUSTOM_ACTION_TYPE, TIME_SLOTS] of 0..100: initialActionCounter;

% actions demands
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, CUSTOM_ACTION_TYPE] of 0..100: actionDemands;

% for each function we cannot have more than initial amount + number of action time slots
int: MAX_FN_PLACEMENT = max([initialFunctionPlacement[i, l, f] | i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE]) + NUM_TIME_SLOTS - 1;

int: TOTAL_ACTION_DEMAND = sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE)(actionDemands[i, l, f, a]);

int: TOTAL_INITIAL_ACTION = sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE, t in ACTION_TIME_SLOTS)(initialActionCounter[i, l, f, a, t]);

%%%%%%%%%%%%%
% Variables %
%%%%%%%%%%%%%

% how much capacity for demands we have in locations - for THR Capability only
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, TIME_SLOTS] of var 0..max([locationCapacity[l, THR] | l in LOCATIONS]): serviceCapacity;

% placement of functions to satisfy demands within location limits for each time slot
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, TIME_SLOTS] of var 0..MAX_FN_PLACEMENT: functionPlacement;

%placement of scheduled actions
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, ACTION_TYPE, TIME_SLOTS] of var 0..MAX_FN_PLACEMENT: actionPlacement;

% how many functions must be reduced for the execution actions on them
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, TIME_SLOTS] of var 0..MAX_FN_PLACEMENT: functionPlacementReduction;

% for how many functions we lock resources
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, TIME_SLOTS] of var 0..1: functionCapacityReservation;   %verify if correct assumption, used only for scaleout what limits use to 1 only in the slot

% placement of functions to satisfy demands within location limits for each time slot
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, TIME_SLOTS] of var 0..1: conflictingActionCounter;

% placement of functions to satisfy demands within location limits for each time slot
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, ACTION_TYPE, TIME_SLOTS] of var 0..100: actionCounter;

% allocation of actions to satisfy action demands
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, CUSTOM_ACTION_TYPE, TIME_SLOTS] of var 0..min([MAX_FN_PLACEMENT, max([actionDemands[i, l, f, a] | i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE])]): actionCounterAllocation;

%how much of the action demand is not satisfied - we should minimize it
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, CUSTOM_ACTION_TYPE] of var 0..100: actionDemandGap;

%amount of time slots we can move to the next iteration for execution of the action
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE] of var 0..NUM_TIME_SLOTS: remainingSlots;

% total actual THR demand in each location
array[SERVICE_INSTANCES, LOCATIONS, LOCATIONS, TIME_SLOTS] of var 0..max([locationCapacity[l, THR] | l in LOCATIONS]): demandAllocations;

% total actual THR demand reduction in each location
array[SERVICE_INSTANCES, LOCATIONS, LOCATIONS, TIME_SLOTS] of var 0..max([locationCapacity[l, THR] | l in LOCATIONS]): demandAllocationsReduction;

% total optimal THR demand in each location
array[SERVICE_INSTANCES, LOCATIONS, TIME_SLOTS] of var 0..max([locationCapacity[l, THR] | l in LOCATIONS]): balancerDemandAllocations;

% total actual THR demand reduction in each location
array[SERVICE_INSTANCES, LOCATIONS, TIME_SLOTS] of var 0..max([locationCapacity[l, THR] | l in LOCATIONS]): balancerDemandAllocationsReduction;

% total incoming THR in each location
array[LOCATIONS, TIME_SLOTS] of var 0..max([locationCapacity[l, THR] | l in LOCATIONS]): throughputInLocations;

%how much of the demand is not satisfied - we should minimize it
array[SERVICE_INSTANCES, LOCATIONS, TIME_SLOTS] of var 0..max([serviceDemands[i, l] | i in SERVICE_INSTANCES, l in LOCATIONS]): demandGap;

%what is the cost of the allocation of demand - on distance basis
array[SERVICE_INSTANCES, LOCATIONS, TIME_SLOTS] of var 0.0..MAX_DISTANCE: demandAllocationCost;

% artificial SLA factor based on demand satisfaction and demand allocation placement
array[SERVICE_INSTANCES, LOCATIONS, TIME_SLOTS] of var 0.0..MAX_SLA: slaSatisfaction;

%SLA averaged over all slots and location for service instance - for cumulative comparison
array[SERVICE_INSTANCES] of var 0.0..1.0: avgSlaSatisfaction;

%how many instances we have consumed over all the time slots
array[SERVICE_INSTANCES, FUNCTION_TYPE] of var int: totalFunctionPlacement;

array[SERVICE_INSTANCES, FUNCTION_TYPE, CUSTOM_ACTION_TYPE] of var int: totalActionCounter;

array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE, CUSTOM_ACTION_TYPE] of var int: totalActionCounterInLocations;

%%%%%%%%%%%%%%%%%%
%   CONSTRAINTS
%%%%%%%%%%%%%%%%%%

constraint assert(forall(l1 in LOCATIONS, l2 in LOCATIONS)(locationDistance[l1, l2] >= 0), "Invalid location costs");
constraint assert(NUM_SERVICE_INSTANCES > 0, "Invalid amount of services");
constraint assert(MAX_SLA >= 1, "Invalid max SLA value declared");

% initial amount of vLB for service in each location
constraint assert(forall(i in SERVICE_INSTANCES, l in LOCATIONS)(
  initialFunctionPlacement[i, l, vLB_1] >= 0 %vLB_1
), "Invalid initial amount of vLB in locations");

% initial amount of vDNS for service in each location
constraint assert(forall(i in SERVICE_INSTANCES, l in LOCATIONS)(
  initialFunctionPlacement[i, l, vDNS_1] >= 0 %vDNS_1
), "Invalid initial amount of vDNS in locations");

% resources allocated to functions cannot exceed the available resources for each capability in location
constraint forall(l in LOCATIONS, t in TIME_SLOTS)(
  sum(i in SERVICE_INSTANCES, f in FUNCTION_TYPE)((functionPlacement[i, l, f, t] + functionCapacityReservation[i, l, f, t]) * maxFunctionRequirements[f, CPU]) <= locationCapacity[l, CPU]
);

constraint forall(l in LOCATIONS, t in TIME_SLOTS)(
  sum(i in SERVICE_INSTANCES, f in FUNCTION_TYPE)((functionPlacement[i, l, f, t] + functionCapacityReservation[i, l, f, t]) * maxFunctionRequirements[f, RAM]) <= locationCapacity[l, RAM]
);

constraint forall(l in LOCATIONS, t in TIME_SLOTS)(
  throughputInLocations[l, t] = sum(i in SERVICE_INSTANCES)(balancerDemandAllocations[i, l, t])
  + sum(i in SERVICE_INSTANCES, l2 in LOCATIONS where l2 != l)(demandAllocations[i, l2, l, t])
);

constraint forall(l in LOCATIONS, t in TIME_SLOTS)(
  throughputInLocations[l, t] <= locationCapacity[l, THR]
);

constraint forall(i in SERVICE_INSTANCES, f in FUNCTION_TYPE)(
    totalFunctionPlacement[i, f] = sum(l in LOCATIONS, t in ACTION_TIME_SLOTS)(functionPlacement[i, l, f, t])
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in TIME_SLOTS)(
  functionPlacement[i, l, f, 1] = initialFunctionPlacement[i, l, f]
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in TIME_SLOTS)(
    if JOINT_SCHEDULING then
        serviceCapacity[i, l, f, t] = (functionPlacement[i, l, f, t] - functionPlacementReduction[i, l, f, t]) * maxFunctionRequirements[f, THR]
    else
        serviceCapacity[i, l, f, t] = max([0, functionPlacement[i, l, f, t] - functionReservationForActions[i, l, f]]) * maxFunctionRequirements[f, THR]
    endif
);

% initial allocation cannot be greater than the capacity for services in locations
%constraint forall(i in SERVICE_INSTANCES, f in FUNCTION_TYPE, l2 in LOCATIONS)(
%  sum(l1 in LOCATIONS)(initialDemandAllocation[i, l1, l2]) <= serviceCapacity[i, l2, f, 1]
%);

% initial allocation cannot be greater than the demand for services in locations
%constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS)(
%  sum(l2 in LOCATIONS)(initialDemandAllocation[i, l1, l2]) <= serviceDemands[i, l1]
%);

%constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS, l2 in LOCATIONS)(
%  demandAllocations[i, l1, l2, 1] = initialDemandAllocation[i, l1, l2]
%);

%constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS)(
%  balancerDemandAllocations[i, l1, 1] = sum(l2 in LOCATIONS)(initialDemandAllocation[i, l1, l2])
%);

% actual satisfaction of SLA - very strange behavior for the first slot
constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS)(
  slaSatisfaction[i, l1, 1] = sum(l2 in LOCATIONS)(initialDemandAllocation[i, l1, l2]*(1-locationDistance[l1, l2]/MAX_DISTANCE)/serviceDemands[i, l1]) * MAX_SLA
);

constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS, t in ACTION_TIME_SLOTS)(
    if JOINT_SCHEDULING then
        slaSatisfaction[i, l1, t] = sum(l2 in LOCATIONS)(demandAllocations[i, l1, l2, t]*(1-locationDistance[l1, l2]/MAX_DISTANCE)/serviceDemands[i, l1]) * MAX_SLA
    else
        slaSatisfaction[i, l1, t] = max([0.0, sum(l2 in LOCATIONS)((demandAllocations[i, l1, l2, t] - demandAllocationsReduction[i, l1, l2, t])*(1-locationDistance[l1, l2]/MAX_DISTANCE)/serviceDemands[i, l1]) * MAX_SLA])
    endif
);

constraint forall(i in SERVICE_INSTANCES)(
  avgSlaSatisfaction[i] = sum(l1 in LOCATIONS, t in ACTION_TIME_SLOTS)(slaSatisfaction[i, l1, t])/((NUM_TIME_SLOTS - 1) * NUM_LOCATION * MAX_SLA)
);

constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS, t in TIME_SLOTS)(
  demandGap[i, l1, t] = serviceDemands[i, l1] - min(balancerDemandAllocations[i, l1, t], sum([demandAllocations[i, l1, l2, t] | l2 in LOCATIONS]))
);

constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS, t in TIME_SLOTS)(
  demandAllocationCost[i, l1, t] = sum([demandAllocations[i, l1, l2, t]*locationDistance[l1, l2]/serviceDemands[i, l1] | l2 in LOCATIONS])
);

constraint forall(i in SERVICE_INSTANCES, l2 in LOCATIONS, t in ACTION_TIME_SLOTS)(
    sum(l1 in LOCATIONS)(demandAllocations[i, l1, l2, t]) <= serviceCapacity[i, l2, vDNS_1, t]
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, t in ACTION_TIME_SLOTS)(
    balancerDemandAllocations[i, l, t] <= serviceCapacity[i, l, vLB_1, t]
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, t in ACTION_TIME_SLOTS)(
    balancerDemandAllocations[i, l, t] == sum(l2 in LOCATIONS)(demandAllocations[i, l, l2, t])
);

% the allocation cannot exceed the demands
constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS, t in TIME_SLOTS) (
    sum(l2 in LOCATIONS)(demandAllocations[i, l1, l2, t]) <= serviceDemands[i, l1]
);


% ACTION CONSTRAINTS

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, a in ACTION_TYPE) (
  actionCounter[i, l, f, a, 1] = 0
);

constraint forall(l in LOCATIONS, f in FUNCTION_TYPE, a in 1..CUSTOM_ACTION_OFFSET, t in ACTION_TIME_SLOTS)(
    if ONE_SERVICE_OPT /\ ITERATION_NO mod 2 == 1 then
        actionCounter[1, l, f, a, t] = 0
    elseif ONE_SERVICE_OPT /\ ITERATION_NO mod 2 == 0 then
        actionCounter[2, l, f, a, t] = 0
    else
        actionCounter[1, l, f, a, t] >= 0 /\ actionCounter[2, l, f, a, t] >= 0
    endif
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS)(
  let {
    var int: placementChange = functionPlacement[i, l, f, t] - functionPlacement[i, l, f, t - 1];
  } in if placementChange > 0 then
        actionCounter[i, l, f, SCALE_OUT, t] = abs(placementChange) /\ actionCounter[i, l, f, SCALE_IN, t] = 0
    else
        actionCounter[i, l, f, SCALE_OUT, t] = 0 /\ actionCounter[i, l, f, SCALE_IN, t] = abs(placementChange)
    endif
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, t in ACTION_TIME_SLOTS) (
  let {
    var int: demandChange = sum(l2 in LOCATIONS)(abs(demandAllocations[i, l, l2, t] - demandAllocations[i, l, l2, t - 1]));
  } in if demandChange > 0 then
    actionCounter[i, l, vLB_1, TD, t] = 1
  else
    actionCounter[i, l, vLB_1, TD, t] = 0
  endif
);

%this function does not participate in the distribution action
constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, t in ACTION_TIME_SLOTS) (
  actionCounter[i, l, vDNS_1, TD, t] = 0
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS) (
    actionCounter[i, l, f, ACT_1, t] = max(0, min(functionPlacement[i, l, f, t - 1] - 1, initialActionCounter[i, l, f, ACT_1 - CUSTOM_ACTION_OFFSET, t]))
                                     + actionCounterAllocation[i, l, f, ACT_1 - CUSTOM_ACTION_OFFSET, t]
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS) (
    actionCounter[i, l, f, ACT_2, t] = max(0, min(functionPlacement[i, l, f, t - 1] - 1 - actionCounter[i, l, f, ACT_1, t], initialActionCounter[i, l, f, ACT_2 - CUSTOM_ACTION_OFFSET, t]))
                                     + actionCounterAllocation[i, l, f, ACT_2 - CUSTOM_ACTION_OFFSET, t]
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS) (
    sum(a in CUSTOM_ACTION_TYPE)(actionCounterAllocation[i, l, f, a, t]) <= functionPlacement[i, l, f, t]
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE, t in ACTION_TIME_SLOTS) (
    if TOTAL_ACTION_DEMAND == 0 then
        actionCounterAllocation[i, l, f, a, t] = 0
    endif
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE) (
    sum(t in ACTION_TIME_SLOTS)(actionCounterAllocation[i, l, f, a, t]) <= actionDemands[i, l, f, a]
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE) (
    if TOTAL_ACTION_DEMAND > 0 then
        actionDemandGap[i, l, f, a] = actionDemands[i, l, f, a] - sum(t in ACTION_TIME_SLOTS)(actionCounterAllocation[i, l, f, a, t])
    else
        actionDemandGap[i, l, f, a] = sum(t in ACTION_TIME_SLOTS)(initialActionCounter[i, l, f, a, t] - actionCounter[i, l, f, a + CUSTOM_ACTION_OFFSET, t])
    endif
);

%the limit of scaled instances at once
constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in TIME_SLOTS) (
  actionCounter[i, l, f, SCALE_OUT, t] <= MAX_SCALED_INST_PER_FN /\ actionCounter[i, l, f, SCALE_IN, t] <= MAX_SCALED_INST_PER_FN
);

%sum of all conflicting action executed at the same time - it must be 0 or 1 - this is the limit of variable
constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in TIME_SLOTS) (
  conflictingActionCounter[i, l, f, t] = sum(a in ACTION_TYPE where actionCounter[i, l, f, a, t] > 0)(actionProperties[a, ACT_CONFLICT])
);

% time spent to do all the conflicting actions for function in location cannot be greater than the slots available for that
constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE) (
    sum(t in TIME_SLOTS, a in ACTION_TYPE where actionProperties[a, ACT_CONFLICT] > 0)(actionCounter[i, l, f, a, t]*actionDuration[locationType[l], f, a]) <= (NUM_TIME_SLOTS - 1 + extraTimeSlots[i, l, f])
);

% action is accounted in the last slot of its execution, so we need to have time slot space to execute it
constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, a in ACTION_TYPE where actionProperties[a, ACT_CONFLICT] > 0, t in ACTION_TIME_SLOTS) (
    actionCounter[i, l, f, a, t] == 0 \/ actionCounter[i, l, f, a, t]*actionDuration[locationType[l], f, a] < (t + extraTimeSlots[i, l, f])
);

% when some conflicting action is being executed any other conflicting action cannot be accounted at the same time
constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, a in ACTION_TYPE, t in ACTION_TIME_SLOTS) (
    if actionProperties[a, ACT_CONFLICT] > 0 then
        sum(a1 in ACTION_TYPE where actionProperties[a1, ACT_CONFLICT] > 0)(actionPlacement[i, l, f, a1, t]) <= 1
    endif
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, a in ACTION_TYPE, t in TIME_SLOTS) (
    if t == 1 \/ actionDuration[locationType[l], f, a] == 1 then
        actionPlacement[i, l, f, a, NUM_TIME_SLOTS - t + 1] = actionCounter[i, l, f, a, NUM_TIME_SLOTS - t + 1]
    elseif t < NUM_TIME_SLOTS then
        actionPlacement[i, l, f, a, NUM_TIME_SLOTS - t + 1] = actionCounter[i, l, f, a, NUM_TIME_SLOTS - t + 1] + actionCounter[i, l, f, a, min(NUM_TIME_SLOTS, NUM_TIME_SLOTS - t + actionDuration[locationType[l], f, a])]
    else
        actionPlacement[i, l, f, a, 1] = 0
    endif
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS) (
    sum(a in ACTION_TYPE)(actionPlacement[i, l, f, a, t]) <= functionPlacement[i, l, f, t]
);

% when action blocks function it cannot be used for allocation of demands - time of account included only for non ScaleIn operation (when function instance is removed)
constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS) (
    %functionPlacementReduction[i, l, f, t] = sum(a in ACTION_TYPE where actionProperties[a, ACT_FN_LOCK] > 0 /\ a != SCALE_IN)(actionPlacement[i, l, f, a, t]) + sum(a in ACTION_TYPE where actionProperties[a, ACT_FN_LOCK] > 0 /\ a == SCALE_IN)(actionPlacement[i, l, f, a, t])%-actionCounter[i, l, f, a, t])
    functionPlacementReduction[i, l, f, t] = sum(a in ACTION_TYPE where actionProperties[a, ACT_FN_LOCK] > 0)(actionPlacement[i, l, f, a, t])
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS) (
    functionPlacementReduction[i, l, f, t] < functionPlacement[i, l, f, t]
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, t in ACTION_TIME_SLOTS) (
    sum(l2 in LOCATIONS)(demandAllocationsReduction[i, l2, l, t]) == max([0, sum(l2 in LOCATIONS)(demandAllocations[i, l2, l, t]) - (functionPlacement[i, l, vDNS_1, t] - functionPlacementReduction[i, l, vDNS_1, t] + functionReservationForActions[i, l, vDNS_1])*maxFunctionRequirements[vDNS_1, THR]])
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, t in ACTION_TIME_SLOTS) (
    balancerDemandAllocationsReduction[i, l, t] = max([0, balancerDemandAllocations[i, l, t] - (functionPlacement[i, l, vLB_1, t] - functionPlacementReduction[i, l, vLB_1, t] + functionReservationForActions[i, l, vLB_1])*maxFunctionRequirements[vLB_1, THR], ])
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, l2 in LOCATIONS, t in ACTION_TIME_SLOTS) (
    demandAllocationsReduction[i, l, l2, t] <= demandAllocations[i, l, l2, t]
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS) (
    %functionCapacityReservation[i, l, f, t] = sum(a in ACTION_TYPE where actionProperties[a, ACT_RES_LOCK] > 0 /\ a != SCALE_OUT)(actionPlacement[i, l, f, a, t]) + sum(a in ACTION_TYPE where actionProperties[a, ACT_RES_LOCK] > 0 /\ a == SCALE_OUT)(actionPlacement[i, l, f, a, t]-actionCounter[i, l, f, a, t])
    functionCapacityReservation[i, l, f, t] = sum(a in ACTION_TYPE where actionProperties[a, ACT_RES_LOCK] > 0)(actionPlacement[i, l, f, a, t])
);

%applies only to conflicting actions as those non conflicting can be executed any time
constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE) (
    remainingSlots[i, l, f] = min(MAX_REMAINING_SLOTS, NUM_TIME_SLOTS - max(t in ACTION_TIME_SLOTS where conflictingActionCounter[i, l, f, t] > 0)(t))
);

constraint forall(i in SERVICE_INSTANCES, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE)(
    totalActionCounter[i, f, a] = sum(l in LOCATIONS, t in ACTION_TIME_SLOTS)(actionCounter[i, l, f, a + CUSTOM_ACTION_OFFSET, t])
);

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE)(
    totalActionCounterInLocations[i, l, f, a] = sum(t in ACTION_TIME_SLOTS)(actionCounter[i, l, f, a + CUSTOM_ACTION_OFFSET, t])
);

%%%%%%%%%%%%%%%%%%
%   OBJECTIVE
%%%%%%%%%%%%%%%%%%

array[SERVICE_INSTANCES, ACTION_TIME_SLOTS] of var float: obj1_parts;
array[SERVICE_INSTANCES, ACTION_TIME_SLOTS] of var float: obj2_parts;

constraint forall(i in SERVICE_INSTANCES, t in ACTION_TIME_SLOTS)(
    obj1_parts[i, t] = sum(l in LOCATIONS)(demandGap[i, l, t])/(sum(i in SERVICE_INSTANCES, l in LOCATIONS)(serviceDemands[i, l])*NUM_TIME_SLOTS)*200000
);

constraint forall(i in SERVICE_INSTANCES, t in ACTION_TIME_SLOTS)(
    obj2_parts[i, t] = sum(l in LOCATIONS)(demandAllocationCost[i, l, t])/(NUM_LOCATION*NUM_SERVICE_INSTANCES*MAX_DISTANCE*NUM_TIME_SLOTS)*100
);

var float: obj1_1 = sum(i in 1..1, t in ACTION_TIME_SLOTS)(obj1_parts[i, t]);
var float: obj2_1 = sum(i in 1..1, t in ACTION_TIME_SLOTS)(obj2_parts[i, t]);
% var float: obj3_1 = sum(i in 1..1, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS)((functionPlacement[i, l, f, t] - functionPlacementReduction[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100;
% var float: obj3_1 = sum(i in 1..1, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS)((functionPlacement[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100;

var float: obj3_1;
constraint(
    if JOINT_SCHEDULING > 0 then
        %obj3_1 = sum(i in 1..1, t in ACTION_TIME_SLOTS)(max([obj1_parts[i, t] + obj2_parts[i, t], sum(l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100]))
        obj3_1 = sum(i in 1..1, t in ACTION_TIME_SLOTS)(max([0, sum(l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100]))
    else
        %obj3_1 = sum(i in 1..1, t in ACTION_TIME_SLOTS)(max([obj1_parts[i, t] + obj2_parts[i, t], sum(l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100]))
        obj3_1 = sum(i in 1..1, t in ACTION_TIME_SLOTS)(max([0, sum(l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100]))
    endif
);

%var float: obj3_1 = sum(i in 1..1, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS)(actionCounter[i, l, f, SCALE_OUT, t])/(NUM_LOCATION*max(FUNCTION_TYPE)*NUM_TIME_SLOTS*100);
var float: obj4_1;
constraint(
    if TOTAL_ACTION_DEMAND > 0 then
        obj4_1 = sum(i in 1..1, l in LOCATIONS, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE)(actionDemandGap[i, l, f, a])/TOTAL_ACTION_DEMAND*100
    else
        obj4_1 = 0 % sum(i in 1..1, l in LOCATIONS, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE)(actionDemandGap[i, l, f, a])/TOTAL_INITIAL_ACTION*100
    endif
);

var float: obj1_2 = sum(i in 2..2, t in ACTION_TIME_SLOTS)(obj1_parts[i, t]);
var float: obj2_2 = sum(i in 2..2, t in ACTION_TIME_SLOTS)(obj2_parts[i, t]);
% var float: obj3_2 = sum(i in 2..2, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS)((functionPlacement[i, l, f, t] - functionPlacementReduction[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100;
% var float: obj3_2 = sum(i in 2..2, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS)((functionPlacement[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100;
var float: obj3_2;
constraint(
    if JOINT_SCHEDULING > 0 then
        %obj3_2 = sum(i in 2..2, t in ACTION_TIME_SLOTS)(max([obj1_parts[i, t] + obj2_parts[i, t], sum(l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100]))
        obj3_2 = sum(i in 2..2, t in ACTION_TIME_SLOTS)(max([0, sum(l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100]))
    else
        %obj3_2 = sum(i in 2..2, t in ACTION_TIME_SLOTS)(max([obj1_parts[i, t] + obj2_parts[i, t], sum(l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100]))
        obj3_2 = sum(i in 2..2, t in ACTION_TIME_SLOTS)(max([0, sum(l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, t])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100]))
    endif
);
% var float: obj3_2 = sum(i in 2..2, l in LOCATIONS, f in FUNCTION_TYPE, t in ACTION_TIME_SLOTS)(actionCounter[i, l, f, SCALE_OUT, t])/(NUM_LOCATION*max(FUNCTION_TYPE)*NUM_TIME_SLOTS*100);
var float: obj4_2;
constraint(
    if TOTAL_ACTION_DEMAND > 0 then
        obj4_2 = sum(i in 2..2, l in LOCATIONS, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE)(actionDemandGap[i, l, f, a])/TOTAL_ACTION_DEMAND*100
    else
        obj4_2 = 0 % sum(i in 2..2, l in LOCATIONS, f in FUNCTION_TYPE, a in CUSTOM_ACTION_TYPE)(actionDemandGap[i, l, f, a])/TOTAL_INITIAL_ACTION*100
    endif
);

var float: lastObj1 = sum(i in SERVICE_INSTANCES, l in LOCATIONS)(demandGap[i, l, NUM_TIME_SLOTS])/(sum(i in SERVICE_INSTANCES, l in LOCATIONS)(serviceDemands[i, l]))*200000;
var float: lastObj2 = sum(i in SERVICE_INSTANCES, l in LOCATIONS)(demandAllocationCost[i, l, NUM_TIME_SLOTS])/(NUM_LOCATION*NUM_SERVICE_INSTANCES*MAX_DISTANCE)*100;
% var float: lastObj3 = sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)((functionPlacement[i, l, f, NUM_TIME_SLOTS] - functionPlacementReduction[i, l, f, NUM_TIME_SLOTS])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU]))/100;
% var float: lastObj3 = sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)((functionPlacement[i, l, f, NUM_TIME_SLOTS])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU]))/100;
var float: lastObj3;
constraint(
    if JOINT_SCHEDULING > 0 then
        %lastObj3 = max([lastObj1 + lastObj2, sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, NUM_TIME_SLOTS])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100])
        lastObj3 = max([0, sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, NUM_TIME_SLOTS])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100])
    else
        %lastObj3 = max([lastObj1 + lastObj2, sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, NUM_TIME_SLOTS])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100])
        lastObj3 = max([0, sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(abs(functionPlacement[i, l, f, NUM_TIME_SLOTS])*maxFunctionRequirements[f, CPU])/(sum(l in LOCATIONS)(locationCapacity[l, CPU])*NUM_TIME_SLOTS)/100])
    endif
);

% var float: lastObj3 = sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(actionCounter[i, l, f, SCALE_OUT, NUM_TIME_SLOTS])/(NUM_SERVICE_INSTANCES*NUM_LOCATION*max(FUNCTION_TYPE)*100);

var float: allocationObjective;
var float: totalObjective;
var float: lastTimeSlotObjective;

constraint(
    lastTimeSlotObjective = lastObj1 + lastObj2 + obj4_1 + obj4_2
);

constraint(
    allocationObjective = 1 + obj1_1 + obj2_1 + obj4_1 + obj1_2 + obj2_2 + obj4_2
);

constraint(
    if MINIMIZE_ALLOCATION then
        totalObjective = 10000*(allocationObjective + obj3_1 + obj3_2)
    else
        totalObjective = 10000*allocationObjective
    endif
);

solve minimize totalObjective;

%%%%%%%%%%%%%%%%%%
%   OUTPUTS
%%%%%%%%%%%%%%%%%%
% ["\(sum(i in SERVICE_INSTANCES, l in LOCATIONS)(serviceDemands[i, l]))\n"] ++
output ["TOTAL ACT DEMAND: \(TOTAL_ACTION_DEMAND)\n"] ++ ["Objective: \(allocationObjective)/\(lastTimeSlotObjective)"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ show_int(6, serviceCapacity[i, l, vLB_1, t]) ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "|" else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ show_int(6, serviceCapacity[i, l, vDNS_1, t]) ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "|" else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nvLB Allocation in Locations\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ show_int(6, balancerDemandAllocations[i, l, t]) ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "|" else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nvDNS Allocation in Locations\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ "[" ++ show_int(2, sum(l2 in LOCATIONS where l2 == l)(demandAllocations[i, l2, l, t])) ++ "," ++ show_int(2, sum(l2 in LOCATIONS where l2 != l)(demandAllocations[i, l, l2, t])) ++ "] " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "| " else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nvDNS Allocation Reduction in Locations\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ "[" ++ show_int(2, sum(l2 in LOCATIONS where l2 == l)(demandAllocationsReduction[i, l2, l, t])) ++ "," ++ show_int(2, sum(l2 in LOCATIONS where l2 != l)(demandAllocationsReduction[i, l, l2, t])) ++ "] " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "| " else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nTraffic Demand Allocation Gap\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ show_int(6, demandGap[i, l, t]) ++ " " ++ 
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "|" else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++           
       ["\nTraffic Demand Allocation Cost\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ show_float(6, 0, demandAllocationCost[i, l, t]) ++ " " ++ 
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "|" else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nAction Demand Allocation Gap for vLB\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ "[\(actionDemandGap[i, l, vLB_1, 1]),\(actionDemandGap[i, l, vLB_1, 2])]" ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "| " else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nAction Counter for vDNS\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ "[\(actionCounter[i, l, vDNS_1, 4, t]),\(actionCounter[i, l, vDNS_1, 5, t])]" ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "| " else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nAction Demand Allocation Gap for vDNS\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ "[\(actionDemandGap[i, l, vDNS_1, 1]),\(actionDemandGap[i, l, vDNS_1, 2])]" ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "| " else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nvCompleted LB Scale vs vDNS Scale vs TD\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ "[\(actionCounter[i, l, vLB_1, SCALE_OUT, t]),\(actionCounter[i, l, vLB_1, SCALE_IN, t]),\(actionCounter[i, l, vLB_1, TD, t])]" ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "| " else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nLB action placement"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ "[\(actionPlacement[i, l, vLB_1, SCALE_OUT, t]),\(actionPlacement[i, l, vLB_1, SCALE_IN, t]),\(actionPlacement[i, l, vLB_1, TD, t])]" ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "| " else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nFunction placement\n\(LOCATIONS_ENUM)"] ++       
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ "[\(functionPlacement[i, l, vLB_1, t]),\(functionPlacement[i, l, vDNS_1, t])]" ++ " " ++ 
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "| " else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++ 
       [if l == 1 /\ t == 1 then "\n>>" else "" endif ++ "[\(functionPlacementTarget[i, l, vLB_1]),\(functionPlacementTarget[i, l, vDNS_1])]" ++ " " ++ 
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "| " else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nRemaining slots\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 then "\n>>" else "" endif ++ "[\(remainingSlots[i, l, vLB_1]),\(remainingSlots[i, l, vDNS_1])]" ++ " " | i in SERVICE_INSTANCES, l in LOCATIONS] ++
       ["\nSLA Satisfaction Level\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ show_float(5, 3, slaSatisfaction[i, l, t]/MAX_SLA) ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "|" else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       [if l == 1 /\ t == 1 then "\n>>" else "" endif ++ show_float(5, 3, targetSlaSatisfaction[i, l]/MAX_SLA) ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "|" else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nFunction placement reduction"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ "[\(functionPlacementReduction[i, l, vLB_1, t]),\(functionPlacementReduction[i, l, vDNS_1, t])]" ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "| " else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nFunction capacity reservation"] ++
       [if l == 1 /\ t == 1 then "\n> " else "" endif ++ "[\(functionCapacityReservation[i, l, vLB_1, t]),\(functionCapacityReservation[i, l, vDNS_1, t])]" ++ " " ++
         if l == max(LOCATIONS) /\ t < max(TIME_SLOTS) then "| " else "" endif | i in SERVICE_INSTANCES, t in TIME_SLOTS, l in LOCATIONS] ++
       ["\nIteration Statistics"] ++
       ["\nAVG SLA: [" ++ show_float(4, 3, avgSlaSatisfaction[1]) ++ ", " ++ show_float(4, 3, avgSlaSatisfaction[2]) ++ "]"] ++
       ["\nFNP: ["] ++ ["\(sum(f in FUNCTION_TYPE)(totalFunctionPlacement[1, f])), \(sum(f in FUNCTION_TYPE)(totalFunctionPlacement[2, f]))]"] ++
       ["\nACT 1: ["] ++ ["\(sum(f in FUNCTION_TYPE)(totalActionCounter[1, f, 1])), \(sum(f in FUNCTION_TYPE)(totalActionCounter[2, f, 1]))]"] ++
       ["\nACT 2: ["] ++ ["\(sum(f in FUNCTION_TYPE)(totalActionCounter[1, f, 2])), \(sum(f in FUNCTION_TYPE)(totalActionCounter[2, f, 2]))]"] ++
       ["\nGAP 1: ["] ++ ["\(sum(f in FUNCTION_TYPE, l in LOCATIONS)(actionDemandGap[1, l, f, 1])), \(sum(f in FUNCTION_TYPE, l in LOCATIONS)(actionDemandGap[2, l, f, 1]))]"] ++
       ["\nGAP 2: ["] ++ ["\(sum(f in FUNCTION_TYPE, l in LOCATIONS)(actionDemandGap[1, l, f, 2])), \(sum(f in FUNCTION_TYPE, l in LOCATIONS)(actionDemandGap[2, l, f, 2]))]"] ++
       ["\nTHR: \(sum(l in LOCATIONS)(throughputInLocations[l, NUM_TIME_SLOTS]))/\(sum(l in LOCATIONS)(locationCapacity[l, THR]))"] ++
       ["\nCPU: \(sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(functionPlacement[i, l, f, NUM_TIME_SLOTS]*maxFunctionRequirements[f, CPU]))/\(sum(l in LOCATIONS)(locationCapacity[l, CPU]))"] ++
       ["\nRAM: \(sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(functionPlacement[i, l, f, NUM_TIME_SLOTS]*maxFunctionRequirements[f, RAM]))/\(sum(l in LOCATIONS)(locationCapacity[l, RAM]))"];
