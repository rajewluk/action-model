include "common_declarations.mzn";

%%%%%%%%%%%%%
% Variables %
%%%%%%%%%%%%%

% how much capacity for demands we would have in locations - for THR Capability only
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE] of var 0..max([locationCapacity[l, THR] | l in LOCATIONS]): targetServiceCapacity;

% how we need to change placement of functions to satisfy demands within location limits for each time slot
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE] of var 0..100: functionPlacementTarget; 

% total optimal THR demand in each location
array[SERVICE_INSTANCES, LOCATIONS, LOCATIONS] of var 0..max([locationCapacity[l, THR] | l in LOCATIONS]): targetDemandAllocations;

% total optimal THR demand in each location
array[SERVICE_INSTANCES, LOCATIONS] of var 0..max([locationCapacity[l, THR] | l in LOCATIONS]): targetBalancerDemandAllocations;

% total incoming THR in each location
array[LOCATIONS] of var 0..max([locationCapacity[l, THR] | l in LOCATIONS]): targetThroughputInLocations;

% avarage percentage of traffic satisfied proportional to the demand
array[SERVICE_INSTANCES, LOCATIONS] of var 0.0..MAX_SLA: targetSlaSatisfaction; 

%how much of the demand is not satisfied - we should minimize it
array[SERVICE_INSTANCES, LOCATIONS] of var 0..max([serviceDemands[i, l] | i in SERVICE_INSTANCES, l in LOCATIONS]): targetDemandGap;

%what is the cost of the allocation of demand - on distance basis
array[SERVICE_INSTANCES, LOCATIONS] of var 0.0..MAX_DISTANCE: targetDemandAllocationCost;

var int: maxActionDuration;

%%%%%%%%%%%%%%%%%%
%   CONSTRAINTS
%%%%%%%%%%%%%%%%%%
constraint (
    maxActionDuration = MAX_ACTION_DURATION
);
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

constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(
  functionPlacementTarget[i, l, f] >= 0
);

% resources allocated to functions cannot exceed the available resources for each capability in location
constraint forall(l in LOCATIONS)(
  sum(i in SERVICE_INSTANCES, f in FUNCTION_TYPE)(functionPlacementTarget[i, l, f] * maxFunctionRequirements[f, CPU]) <= locationCapacity[l, CPU]
);

constraint forall(l in LOCATIONS)(
  sum(i in SERVICE_INSTANCES, f in FUNCTION_TYPE)(functionPlacementTarget[i, l, f] * maxFunctionRequirements[f, RAM]) <= locationCapacity[l, RAM]
);

constraint forall(l in LOCATIONS)(
  targetThroughputInLocations[l] = sum(i in SERVICE_INSTANCES)(targetBalancerDemandAllocations[i, l])
  + sum(i in SERVICE_INSTANCES, l2 in LOCATIONS where l2 != l)(targetDemandAllocations[i, l2, l])
);

constraint forall(l in LOCATIONS)(
  targetThroughputInLocations[l] <= locationCapacity[l, THR]
);

% calculation of targetServiceCapacity
constraint forall(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(
    targetServiceCapacity[i, l, f] = functionPlacementTarget[i, l, f] * maxFunctionRequirements[f, THR]
);

% target satisfaction of SLA 
constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS)(
  targetSlaSatisfaction[i, l1] = sum(l2 in LOCATIONS)(targetDemandAllocations[i, l1, l2]*(1-locationDistance[l1, l2]/MAX_DISTANCE)/serviceDemands[i, l1]) * MAX_SLA
);

constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS)(
  targetDemandGap[i, l1] = serviceDemands[i, l1] - min(targetBalancerDemandAllocations[i, l1], sum([targetDemandAllocations[i, l1, l2] | l2 in LOCATIONS]))
);

constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS)(
  targetDemandAllocationCost[i, l1] = sum([targetDemandAllocations[i, l1, l2]*locationDistance[l1, l2]/serviceDemands[i, l1] | l2 in LOCATIONS])
);

constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS, l2 in LOCATIONS)(targetDemandAllocations[i, l1, l2] >= 0);

% we cannot exceed the capacity of each location
constraint forall(i in SERVICE_INSTANCES, l2 in LOCATIONS)(
    sum(l1 in LOCATIONS)(targetDemandAllocations[i, l1, l2]) <= targetServiceCapacity[i, l2, vDNS_1]
);

% we cannot exceed the capacity of each location
constraint forall(i in SERVICE_INSTANCES, l2 in LOCATIONS)(
    targetBalancerDemandAllocations[i, l2] <= targetServiceCapacity[i, l2, vLB_1]
);

% the allocation cannot exceed the demands
constraint forall(i in SERVICE_INSTANCES, l1 in LOCATIONS)(
    sum(l2 in LOCATIONS)(targetDemandAllocations[i, l1, l2]) <= serviceDemands[i, l1]
);

%%%%%%%%%%%%%%%%%%
%   OBJECTIVE
%%%%%%%%%%%%%%%%%%

var float: obj1 = sum(i in SERVICE_INSTANCES, l in LOCATIONS)(targetDemandGap[i, l])/sum(i in SERVICE_INSTANCES, l in LOCATIONS)(serviceDemands[i, l])*200000;
var float: obj2 = sum(i in SERVICE_INSTANCES, l in LOCATIONS)(targetDemandAllocationCost[i, l])/(NUM_LOCATION*NUM_SERVICE_INSTANCES*MAX_DISTANCE)*100;
var float: obj3 = sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(functionPlacementTarget[i, l, f]*maxFunctionRequirements[f, CPU])/sum(l in LOCATIONS)(locationCapacity[l, CPU]);

var float: allocationObjective;

constraint(
    allocationObjective = obj1 + obj2
);

var float: totalObjective;

constraint(
    if MINIMIZE_ALLOCATION then
        totalObjective = allocationObjective + obj3
    else
        totalObjective = allocationObjective
    endif
);

solve minimize totalObjective;

%%%%%%%%%%%%%%%%%%
%   OUTPUTS
%%%%%%%%%%%%%%%%%%
% ["\(sum(i in SERVICE_INSTANCES, l in LOCATIONS)(serviceDemands[i, l]))\n"] ++
output ["Objective: " ++ show_float(0, 4, allocationObjective)] ++
       ["\nvLB Allocation in Locations\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 then "\n>>" else "" endif ++ show_int(6, targetBalancerDemandAllocations[i, l]) ++ " " | i in SERVICE_INSTANCES, l in LOCATIONS] ++
       ["\nvDNS Allocation in Locations\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 then "\n>>" else "" endif ++ "[" ++ show_int(3, sum(l2 in LOCATIONS where l2 == l)(targetDemandAllocations[i, l, l2])) ++ "," ++ show_int(3, sum(l2 in LOCATIONS where l2 != l)(targetDemandAllocations[i, l, l2])) ++ "] " | i in SERVICE_INSTANCES, l in LOCATIONS] ++
       ["\nDemand Allocation Gap in Locations\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 then "\n>>" else "" endif ++ show_int(6, targetDemandGap[i, l]) ++ " " | i in SERVICE_INSTANCES, l in LOCATIONS] ++
       ["\nDemand Allocation Cost in Locations\n\(LOCATIONS_ENUM)"] ++
       [if l == 1 then "\n>>" else "" endif ++ show_float(6, 0, targetDemandAllocationCost[i, l]) ++ " " | i in SERVICE_INSTANCES, l in LOCATIONS] ++
       ["\nSLA Satisfaction Level\n\(LOCATIONS_ENUM)"] ++      
       [if l == 1 then "\n>>" else "" endif ++ show_float(6, 3, targetSlaSatisfaction[i, l]/MAX_SLA) ++ " " | i in SERVICE_INSTANCES, l in LOCATIONS] ++   
       ["\nFunction Placement\n\(LOCATIONS_ENUM)"] ++                  
       [if l == 1 then "\n>>" else "" endif ++ "[\(functionPlacementTarget[i, l, vLB_1]),\(functionPlacementTarget[i, l, vDNS_1])]" ++ " " | i in SERVICE_INSTANCES, l in LOCATIONS]++
       ["\nTotal Resource Consumption"] ++
       ["\nTHR: \(sum(l in LOCATIONS)(targetThroughputInLocations[l]))/\(sum(l in LOCATIONS)(locationCapacity[l, THR]))"] ++
       ["\nCPU: \(sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(functionPlacementTarget[i, l, f]*maxFunctionRequirements[f, CPU]))/\(sum(l in LOCATIONS)(locationCapacity[l, CPU]))"] ++
       ["\nRAM: \(sum(i in SERVICE_INSTANCES, l in LOCATIONS, f in FUNCTION_TYPE)(functionPlacementTarget[i, l, f]*maxFunctionRequirements[f, RAM]))/\(sum(l in LOCATIONS)(locationCapacity[l, RAM]))"];
