import pulp

#both needed
##brew install glpk or sudo apt-get install glpk-utils
##pip3 install pulp
# Create the problem variable
prob = pulp.LpProblem("Minimize_G", pulp.LpMinimize)

# Define the dimensions
N = 10  # Number of i
R = 5   # Number of j
U = 3   # Number of u
#K = 4   # Number of k

# Create decision variables
G = pulp.LpVariable.dicts("G", (range(1, N+1), range(1, R+1)), cat='Binary')
X = pulp.LpVariable.dicts("X", (range(1, U+1), range(1, N+1), range(1, R+1)), lowBound=0, cat='Integer')

# Coefficients and parameters (example values, replace with actual data)
A = [[5 for _ in range(R)] for _ in range(N)]
Q = [10 for _ in range(U)]
L = [100 for _ in range(U)]

# Objective function
prob += pulp.lpSum(G[i][j] for i in range(1, N+1) for j in range(1, R+1)), "Minimize_G"

# Constraints
# 1. sum(j=1 to R) G_ij <= 1 for all i
for i in range(1, N+1):
    prob += pulp.lpSum(G[i][j] for j in range(1, R+1)) <= 1, f"Constraint_1_{i}"

# 2. sum(i, j) X_uij >= Q_u for all u
for u in range(1, U+1):
    prob += pulp.lpSum(X[u][i][j] for i in range(1, N+1) for j in range(1, R+1)) >= Q[u-1], f"Constraint_2_{u}"

# 3. sum(u,j) X_uij <= sum(j) G_ij * A_ij for all i
for i in range(1, N+1):
    prob += pulp.lpSum(X[u][i][j] for u in range(1, U+1) for j in range(1, R+1)) <= \
            pulp.lpSum(G[i][j] * A[i-1][j-1] for j in range(1, R+1)), f"Constraint_3_{i}"

# 4. f_{i,j}(X_{u,i,j}) <= L_k for all k, i, j
def f(x):
    # Example function, replace with actual function
    return x + 2  # Replace this with the actual function logic

for u in range(1, U+1):
    for i in range(1, N+1):
        for j in range(1, R+1):
            prob += f((X[u][i][j]))  <= L[u-1], f"Constraint_4_{u}_{i}_{j}"

# Solve the problem using GLPK solver
prob.solve(pulp.GLPK_CMD(msg=True))

# Print the status of the solution
print("Status:", pulp.LpStatus[prob.status])

# Print the ILP problem
print("Objective function:")
print(prob.objective)

print("\nConstraints:")
for name, constraint in prob.constraints.items():
    print(f"{name}: {constraint}")

# Print the values of the decision variables
print("\nDecision Variables:")
for i in range(1, N+1):
    for j in range(1, R+1):
        print(f"G[{i}][{j}] = {pulp.value(G[i][j])}")

for u in range(1, U+1):
    for i in range(1, N+1):
        for j in range(1, R+1):
            print(f"X[{u}][{i}][{j}] = {pulp.value(X[u][i][j])}")