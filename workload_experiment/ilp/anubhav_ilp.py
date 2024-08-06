import pulp
import numpy as np
import json
import time

# Define initial problem data
instances = {
    "Llama-8B A100 80GB (TGIS)": 15,
    "Llama-8B MIG 3g.40gb (TGIS)": 8,
    "Granite-3B A100 80GB (TGIS)": 60,
    "Bloom-760 A100 80GB (TGIS)": 140,
    "Bloom-760 MIG 3g.40gb (TGIS)": 70
    # "Granite-3B A100 80GB (vLLM)": 300,
    # "Llama-8B A100 80GB (vLLM)": 80
}

instance_ips = {
    "Llama-8B A100 80GB (TGIS)": "192.168.1.1",
    "Llama-8B MIG 3g.40gb (TGIS)": "192.168.1.2",
    "Granite-3B A100 80GB (TGIS)": "192.168.1.3",
    "Bloom-760 A100 80GB (TGIS)": "192.168.1.4",
    "Bloom-760 MIG 3g.40gb (TGIS)": "192.168.1.5",
    # "Granite-3B A100 80GB (vLLM)": "192.168.1.6",
    # "Llama-8B A100 80GB (vLLM)": "192.168.1.7"
}

N = len(instances)  # Number of GPU configurations
R = 1 # Number of modes per GPU -> rate at which the GPU is operated at --> should be less than the aggregated rates
M = 3  # Number of user classes

# Initial Request rates and latency bounds
initial_Q = [3, 5, 7]  # Request rates for each user class
initial_L = [100, 200, 300]  # Latency bounds for each user class

aggregate_rate_instance = list(instances.values())


# Define user names
users = ['Alan', 'Noel', 'Hari']

# This function represents how long it takes to process queries at a given rate (r) on GPU (i) in mode (j)
# f(i, j, r): Denotes the latency in GPU i mode j for rate r
def f_latency(i, j, r):
    return 10 * r + ((i + 1) * (j + 1))  # dummy latency function of r, i, j.

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

    # Rate Allocation:
    # The total rate allocated to each user Uu across "i" GPUs operatung at "j" mode must be equal to their input request rate Qu: ∑i=1N ∑j=1R X[u, i, j] = Qu ∀u=1,2,…,M
    for u in range(M):
        prob += pulp.lpSum(X[u][i][j] for i in range(N) for j in range(R)) == Q[u], f"Rate_Allocation_{users[u]}"

    # Total Allocated Rate Constraint:
    # The total rate allocated to GPU i in all modes must not exceed the maximum RPM of the active mode
    for i in range(N):
        # Sum of rates allocated to GPU i in all modes
        prob += T[i] == pulp.lpSum(X[u][i][j] for u in range(M) for j in range(R)), f"Total_Allocated_Rate_{i}"
        prob += T[i] <= pulp.lpSum(G[i][j] * aggregate_rate_instance[i] for j in range(R)), f"Aggregate_Rate_{i}"

    # Latency Bound:
    # The latency for each user Uu must be within their latency bound Lu: f(i,j)(∑k=1u X[k,i,j]) ≤ Lu ∀i=1,…,N, ∀j=1,…,R, ∀u=1,…,M
    for i in range(N):
        for j in range(R):
            for u in range(M):
                prob += f_latency(i, j, pulp.lpSum(X[k][i][j] for k in range(u+1))) <= L[u], f"Latency_Bound_{users[u]}_{i}_{j}"

    # Solve the problem
    prob.solve()

    # Output results
    print("Status:", pulp.LpStatus[prob.status])
    print("Objective value:", pulp.value(prob.objective))

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
                if X[u][i][j].varValue > 0:
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
                        "Port": 8033,  # Assuming port 8033 as default
                        "Traffic": traffic_percentage,
                        "Priority": u + 1  # Assuming priority based on user index
                    }
                    user_info["Services"].append(service_info)
        user_config["Users"].append(user_info)

    # Write the JSON configuration to a file
    with open('user_config.json', 'w') as json_file:
        json.dump(user_config, json_file, indent=4)


# Function to update request rates and latency bounds periodically
def update_parameters_and_solve():
    t = 5  # Time interval in seconds for updates

    # Define new sets of request rates and latency bounds for updates
    new_Q = [
        [10, 60, 80],
        [5, 7, 10]
    ]
    new_L = [
        [50, 70, 100],
        [1000, 2000, 3000]
    ]

    # Number of updates
    num_updates = len(new_Q)

    for update in range(num_updates):
        # Update request rates and latency bounds
        Q = new_Q[update]
        L = new_L[update]

        # Solve the optimization problem with updated parameters
        solve_optimization(Q, L)

        # Wait for the next update interval
        time.sleep(t)


# Initial solve
solve_optimization(initial_Q,initial_L)
update_parameters_and_solve()

