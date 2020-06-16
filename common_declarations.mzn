bool: MINIMIZE_ALLOCATION;
bool: ONE_SERVICE_OPT;
int: NUM_LOCATION = 5;
set of int: LOCATIONS = 1..NUM_LOCATION;
enum LOCATIONS_ENUM;
enum LOCATION_TYPES_ENUM;
set of int: LOCATION_TYPES = 1..2;
enum SERVICE_TYPES;
set of int: CAPABILITY = 1..3;
enum CAPABILITY_ENUM = {CPU, RAM, THR}; %CPU [cores], RAM [GB], THR [Mb/s] - for OS function THR is enough because it is achievable for allocated CPU and RAM. For K8s consumed RAM and CPU on node level depends on THR
set of int: FUNCTION_TYPE = 1..2;
enum FUNCTION_TYPE_ENUM = {vLB_1, vDNS_1};
set of int: ACTION_TYPE = 1..5;
set of int: CUSTOM_ACTION_TYPE = 1..2; % + CUSTOM_ACTION_OFFSET
enum ACTION_TYPE_ENUM = {SCALE_OUT, SCALE_IN, TD, ACT_1, ACT_2};
set of int: ACTION_PROPERTIES = 1..4;
enum ACTION_PROPERTIES_ENUM = {ACT_CONFLICT, ACT_RES_LOCK, ACT_FN_LOCK, ACT_DEADLINE};

int: NUM_SERVICE_INSTANCES;
float: MAX_SLA;
int: MAX_DISTANCE; % calculated
int: NUM_TIME_SLOTS;
int: ITERATION_NO;
int: MAX_SCALED_INST_PER_FN;
int: MAX_ACTION_DURATION; % calculated
int: MAX_REMAINING_SLOTS;
int: MAX_ACTION_PENALTY;
int: CUSTOM_ACTION_OFFSET = max(ACTION_TYPE) - max(CUSTOM_ACTION_TYPE);

array[LOCATIONS, LOCATIONS] of 0..1000: locationDistance;
array[LOCATIONS] of LOCATION_TYPES: locationType;
set of int: SERVICE_INSTANCES = 1..NUM_SERVICE_INSTANCES;
array[SERVICE_INSTANCES] of int: serviceInstanceType;
array[SERVICE_INSTANCES] of int: serviceInstanceSLALevel;
array[SERVICE_INSTANCES, LOCATIONS] of int: serviceDemands;
array[LOCATION_TYPES, FUNCTION_TYPE, ACTION_TYPE] of 0..4: actionDuration;
array[ACTION_TYPE, ACTION_PROPERTIES] of 0..MAX_ACTION_PENALTY: actionProperties;
set of int: TIME_SLOTS = 1..NUM_TIME_SLOTS;
set of int: ACTION_TIME_SLOTS = 2..NUM_TIME_SLOTS;

array[FUNCTION_TYPE, CAPABILITY] of 0..20000: maxFunctionRequirements; %w OpenStack CPU i RAM fixed, THR changing, in K8s all fluctuates. Utilization of RAM dependent on THR
array[LOCATIONS, CAPABILITY] of 0..20000: locationCapacity; % simplified - cumulative per all nodes in location
% initial placement of instances
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE] of 0..100: initialFunctionPlacement;
% initial placement of instances
array[SERVICE_INSTANCES, LOCATIONS, LOCATIONS] of 0..max([locationCapacity[l, to_enum(CAPABILITY, 3)] | l in LOCATIONS]): initialDemandAllocation;
% remaining time slots from previous iteration
array[SERVICE_INSTANCES, LOCATIONS, FUNCTION_TYPE] of 0..NUM_TIME_SLOTS: extraTimeSlots;