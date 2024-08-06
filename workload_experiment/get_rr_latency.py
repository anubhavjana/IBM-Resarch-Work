
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# python3 get_rr_latency.py --model granite

parser = argparse.ArgumentParser(description='Process rr and op.')
parser.add_argument('--model', type=str, required=True, help='Model Name (llama3 or granite)')
args = parser.parse_args()

m = args.model


if(m == 'llama3'):
    rpm_values = np.array([10, 12, 15, 20, 23, 25]).reshape(-1, 1)  # RPM values
    latency_percentiles = np.array([24.21991028641683, 34.875489257345905, 352.54013655047106, 745.8313872170781, 927.3231012544129, 978.9659650779179])  # 95th percentile latencies
if(m == 'granite'):
    # rpm_values = np.array([30, 35, 40, 45, 50, 55, 60]).reshape(-1, 1)  # RPM values
    # latency_percentiles = np.array([6.865325208887861,7.708147933930741,8.541349597021028,9.60337202564868, 292.08460685302265, 463.2855040913525, 572.2186784752308])  # 95th percentile latencies

    rpm_values = np.array([45, 48, 50, 52, 55, 60]).reshape(-1, 1)  # RPM values
    latency_percentiles = np.array([9.60337202564868, 11.635328602024574, 292.08460685302265, 323.20282981332446, 463.2855040913525, 572.2186784752308])  # 95th percentile latencies



# Polynomial Regression
degree = 1
poly = PolynomialFeatures(degree)  # Create polynomial features
rpm_poly = poly.fit_transform(rpm_values)  # Transform RPM values to polynomial features

model = LinearRegression()  # Create a linear regression model
model.fit(rpm_poly, latency_percentiles)  # Fit the model with polynomial features and latency data
latency_pred = model.predict(rpm_poly)  # Predict latency using the model (for visualization)

plt.figure(figsize=(10, 6))

plt.scatter(rpm_values, latency_percentiles, color='blue', alpha=0.7, label='95th Percentile Latency Data Points')

# Plot the regression curve
unique_rpms = np.linspace(min(rpm_values), max(rpm_values), 500).reshape(-1, 1)  # Generate a range of RPM values
unique_rpms_poly = poly.transform(unique_rpms)  # Transform these RPM values to polynomial features
predicted_latencies = model.predict(unique_rpms_poly)  # Predict latencies for these RPM values
plt.plot(unique_rpms, predicted_latencies, color='red', label='Fitted Regression Curve')

plt.xlabel('Request Rate (RPM)')  
plt.ylabel('Latency (s)')  
plt.title(f'Relationship between Request Rate and Latency ({m} A100 80GB)')
plt.legend()
plt.grid(True)
plt.savefig(f'{m}_RPM_Latency_relation.png')

coefficients = model.coef_  # Get coefficients of the polynomial
intercept = model.intercept_  # Get the intercept of the polynomial

# print the final polynomial equation
equation = f"Latency = {intercept:.4f}" 
for i in range(degree, 0, -1): 
    coeff = coefficients[i]  # Get the coefficient for the current degree
    if coeff != 0:  
        if i == 1:
            equation += f" + {coeff:.4f} * RPM"  # Linear term
        else:
            equation += f" + {coeff:.4f} * RPM^{i}"  # Higher-order terms

print("Final Polynomial Equation:")
print(equation)
