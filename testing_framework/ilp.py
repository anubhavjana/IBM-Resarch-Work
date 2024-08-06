import pulp
import json
import time
import math

from latency_segments import latency_segments


instances = {
    "Llama-8B A100 80GB (TGIS)-mode-1.1": 10, # this means mode 1 will be running in a single mode, mode 2 will be rinning in 1 mode

    "Llama-8B A100 80GB (TGIS)-mode-2.1": 12,
    "Llama-8B A100 80GB (TGIS)-mode-2.2": 12,
    "Llama-8B A100 80GB (TGIS)-mode-2.3": 12,
    
    "Llama-8B A100 80GB (TGIS)-mode-3.1": 15,
    "Llama-8B A100 80GB (TGIS)-mode-4.1": 20,
   
}


# replace with actual instance ips 
instance_ips = {
   
    "Llama-8B A100 80GB (TGIS)-mode-1.1": "10.128.5.11",

    "Llama-8B A100 80GB (TGIS)-mode-2.1": "10.128.4.15",
    "Llama-8B A100 80GB (TGIS)-mode-2.2": "10.128.4.16",
    "Llama-8B A100 80GB (TGIS)-mode-2.3": "10.128.4.17",
   
    "Llama-8B A100 80GB (TGIS)-mode-3.1": "10.128.5.14",
   
    "Llama-8B A100 80GB (TGIS)-mode-4.1": "10.128.5.15"
   
}

# instances = {
#     "Llama-8B A100 80GB (TGIS)-mode-1.1": 10, # this means mode 1 will be running in a single mode, mode 2 will be rinning in 1 mode
#     "Llama-8B A100 80GB (TGIS)-mode-1.2": 10,
#     "Llama-8B A100 80GB (TGIS)-mode-1.3": 10,
#     "Llama-8B A100 80GB (TGIS)-mode-1.4": 10,
#     "Llama-8B A100 80GB (TGIS)-mode-2.1": 12,
#     "Llama-8B A100 80GB (TGIS)-mode-2.2": 12,
#     "Llama-8B A100 80GB (TGIS)-mode-2.3": 12,
#     "Llama-8B A100 80GB (TGIS)-mode-2.4": 12,
#     "Llama-8B A100 80GB (TGIS)-mode-3.1": 15,
#     "Llama-8B A100 80GB (TGIS)-mode-3.2": 15,
#     "Llama-8B A100 80GB (TGIS)-mode-3.3": 15,
#     "Llama-8B A100 80GB (TGIS)-mode-3.4": 15,
#     "Llama-8B A100 80GB (TGIS)-mode-4.1": 20,
#     "Llama-8B A100 80GB (TGIS)-mode-4.2": 20,
#     "Llama-8B A100 80GB (TGIS)-mode-4.3": 20,
#     "Llama-8B A100 80GB (TGIS)-mode-4.4": 20
    
# }


N = len(instances)  # Number of GPU configurations
R = 1  # Number of modes per GPU
M = 2  # Number of user classes
K = 4  # Number of replicas per instance per mode


# Define user names
users = ['Alan','Noel']

aggregate_rate_instance = list(instances.values())


# This function returns latency constraints based on predefined piecewise linear equations
def f_latency(r, i, j):
    instance_name = list(instances.keys())[i] # get instance name from "i"
    instance_segments = latency_segments.get(instance_name, {})
    # mode_rr = aggregate_rate_instance[i]
    constraints = []
    intercepts = []
    slopes = []
    segments = instance_segments.get(j, []) # tuple of slope,intercept
    for (m, c) in segments:
        constraints.append(m * r + c)
        intercepts.append(c)
        slopes.append(m)
    # return 10 * r * (i+1) * (j+1)
    return constraints,intercepts,slopes

# Function to solve the optimization problem
def solve_optimization(Q, L):
    # Create the problem
    prob = pulp.LpProblem("Instance_Allocation_Problem", pulp.LpMinimize)

    # Variables
    # Binary Variables:
    # G[i, j]: Indicates if GPU i is operated in mode j. G[i, j] ∈ {0, 1}
    G = pulp.LpVariable.dicts("G", (range(N), range(R)), cat='Binary')

    # Decision Variable:
    # X[u, i, j]: Rate allocation to user Uu on GPU i in mode j. X[u, i, j] ≥ 0
    X = pulp.LpVariable.dicts("X", (range(M), range(N), range(R)), lowBound=0, cat='Continuous')

    # Auxiliary Variable for Total Allocated Rate
    T = pulp.LpVariable.dicts("T", range(N), lowBound=0, cat='Continuous')

    # Objective Function
    prob += pulp.lpSum(G[i][j] for i in range(N) for j in range(R))

    # Constraints
    # GPU Mode Activation:
    # The sum of modes activated for each GPU i must not exceed 1: ∑j=1R G[i, j] ≤ 1 ∀i
    for i in range(N):
        prob += pulp.lpSum(G[i][j] for j in range(R)) <= 1, f"Mode_Activation_{i}"

    
    # for i in range(N):
    #     for j in range(R):
    #         prob += G[i][j] >= 0, f"Non_Negative_G_{i}_{j}"
    #         prob += G[i][j] <= 1, f"Upper_Bound_G_{i}_{j}"

    

    # Rate Allocation:
    # The total rate allocated to each user Uu across "i" GPUs operatung at "j" mode must be equal to their input request rate Qu: ∑i=1N ∑j=1R X[u, i, j] = Qu ∀u=1,2,…,M
    for u in range(M):
        prob += pulp.lpSum(X[u][i][j] for i in range(N) for j in range(R)) == Q[u], f"Rate_Allocation_{users[u]}"

    
    for u in range(M):
        for i in range(N):
            for j in range(R):
                prob += X[u][i][j] >= 0 , f'Rate_Allocation_Positive_{users[u]}_{i}_{j}'
    # for u in range(M):
    #     prob += pulp.lpSum(X[u][i][j] for i in range(N) for j in range(R)) >= 0, f"Rate_Allocation__positive_{users[u]}"

        

    # Total Allocated Rate Constraint:
    # The total rate allocated to GPU i in all modes must not exceed the maximum RPM of the active mode

    
    for i in range(N):
        prob += pulp.lpSum(X[u][i][j] for u in range(M) for j in range(R)) <= pulp.lpSum(G[i][j] * aggregate_rate_instance[i] for j in range(R)), f"Aggregate_Rate_{i}"



    # Latency Bound:
    # The latency for each user Uu must be within their latency bound Lu: f(i,j)(∑k=1u X[k,i,j]) ≤ Lu ∀i=1,…,N, ∀j=1,…,R, ∀u=1,…,M
    for i in range(N):
        for j in range(R):
            for u in range(M):
                r = pulp.lpSum(X[k][i][j] for k in range(u+1))
                latency_constraints, intercepts, slopes = f_latency(r, i, j)

                # if RHS becomes negative, make it 0 ==> e.g. 6.46074472287 X_0_0_0 <= 0
                for idx, constraint in enumerate(latency_constraints):
                    
                    if intercepts[idx] >= L[u]:
                        # If intercept >= latency bound, the constraint becomes m*r <= 0
                        prob += slopes[idx]* r <= 0 , f"Latency_Constraint_{users[u]}_{i}_{j}_{idx}"
                    else:
                        prob += slopes[idx]* r + intercepts[idx] <= L[u], f"Latency_Constraint_{users[u]}_{i}_{j}_{idx}"
                        

    prob.solve()

    # print(prob)

    # Output results
    print("Status:", pulp.LpStatus[prob.status])
    print("Objective value:", pulp.value(prob.objective))


    # print(f'Printing X[u][i][j]...')
    # for u in range(M):
    #     for i in range(N):
    #         for j in range(R):
    #             print(f'X[{u}][{i}][{j}] = {X[u][i][j].varValue}')


    # for i in range(N):
    #     for j in range(R):
    #         var_value = G[i][j].varValue
    #         # if var_value is not None and var_value not in [0, 1]:
    #         #     print(f"Warning: G[{i}][{j}] = {var_value} (not binary)")
    #         print(f'G[{i}][{j}] = {var_value}')

    # Print selected instances
    print("Selected Inference instances:")
    for i in range(N):
        for j in range(R):
            if G[i][j].varValue == 1:
                instance_name = list(instances.keys())[i]
                instance_ip = instance_ips[instance_name]
                print(f"GPU Instance: {instance_name}, IP: {instance_ip}, Mode: {j+1}")

    
    

    # Print rate allocation
    print("\nRate allocation per user per GPU mode:")
    for u in range(M):
        for i in range(N):
            for j in range(R):
                if X[u][i][j].varValue > 0 and G[i][j] == 1 :
                    instance_name = list(instances.keys())[i]
                    instance_ip = instance_ips[instance_name]
                    print(f"User {users[u]} -> GPU Instance: {instance_name}, IP: {instance_ip}, Mode: {j+1}, Rate: {X[u][i][j].varValue}")
    print("------------------------------------------------------------------------------------------------------------------------------------\n")

    # Prepare JSON configuration
    user_config = {"Users": []}

    for u in range(M):
        user_info = {"UserId": users[u], "Services": []}
        for i in range(N):
            for j in range(R):
                if X[u][i][j].varValue > 0:
                    instance_name = list(instances.keys())[i]
                    instance_ip = instance_ips[instance_name]
                    rate_allocated = X[u][i][j].varValue
                    total_request_rate = Q[u]
                    traffic_percentage = (rate_allocated / total_request_rate) * 100
                    service_info = {
                        "Service": instance_ip,
                        "Port": 8033,  
                        "Traffic": math.ceil(traffic_percentage),
                        "Priority": u + 1
                    }
                    user_info["Services"].append(service_info)
        user_config["Users"].append(user_info)

    # Write to a user config file 
    with open('user_config.json', 'w') as json_file:
        json.dump(user_config, json_file, indent=4)

# Function to read configuration file
def read_config_file(config_file):
    with open(config_file, 'r') as file:
        config_data = json.load(file)
    return config_data

# Function to update request rates and latency bounds periodically
def update_parameters_and_solve(config_file):
    config_data = read_config_file(config_file)

    updates = config_data["updates"]
    for i, update in enumerate(updates):
        time_to_wait = update["time"]
        Q = update["request_rates"]
        L = update["latencies"]

        if i == 0:
            # Skip the first update as it's already processed
            continue

        print(f"Updating ILP input configurations after {time_to_wait} minutes...\n")
        time.sleep((time_to_wait * 60)+2)

        # Solve the optimization problem with updated parameters
        solve_optimization(Q, L)
        print(f"Updated ILP input configurations after waiitng {time_to_wait} minutes...\n")


# Read initial configuration
config_data = read_config_file('input_config.json')
initial_update = config_data["updates"][0]
initial_Q = initial_update["request_rates"]
initial_L = initial_update["latencies"]

# Initial solve with initial values from config
solve_optimization(initial_Q, initial_L)

# Update parameters periodically based on the configuration file
update_parameters_and_solve('input_config.json')
