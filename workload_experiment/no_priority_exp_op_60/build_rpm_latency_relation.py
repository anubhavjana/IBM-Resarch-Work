# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# import llama_no_priority_exp
# import granite_no_priority_exp
# import bloom_no_priority_exp

# def compute_cdf_from_hist(latency_values, bins=1000):
#     """
#     Compute the CDF from histogram of latency values.
#     """
#     counts, bin_edges = np.histogram(latency_values, bins=bins, density=True)
#     cdf = np.cumsum(counts)
#     cdf = cdf / cdf[-1]  # Normalize to get values between 0 and 1
#     return bin_edges[1:], cdf  # bin_edges[1:] to use the upper edges of bins

# def fit_piecewise_linear_model(latency_df, request_rate, model, num_segments=3):
#     """
#     Fit a piecewise linear model on the CDF of latency.
#     """
#     latency_values = latency_df['latency'].dropna().values

#     # Compute CDF
#     sorted_latencies, cdf = compute_cdf_from_hist(latency_values)
    
#     # Define breakpoints for segments
#     segment_breaks = np.linspace(0, len(cdf), num_segments+1).astype(int)
    
#     # Plot CDF using histogram
#     fig, axes = plt.subplots(2, 1, figsize=(10, 12))
#     counts, bin_edges, _ = axes[0].hist(latency_values, bins=1000, cumulative=True, density=True, histtype='step', linewidth=2, label='TGIS Latency')
#     axes[0].set_xlabel('Latency (s)')
#     axes[0].set_ylabel('CDF')
#     axes[0].set_title(f'{model} - CDF of Latency for Request Rate {request_rate} RPM')
#     axes[0].legend()
#     axes[0].grid(True)
    
#     slopes = []
#     intercepts = []
    
#     # Fit linear models for each segment
#     for i in range(num_segments):
#         start_idx = segment_breaks[i]
#         end_idx = segment_breaks[i+1]
        
#         X = cdf[start_idx:end_idx].reshape(-1, 1)
#         y = sorted_latencies[start_idx:end_idx]
        
#         linear_regressor = LinearRegression()
#         linear_regressor.fit(X, y)
        
#         slopes.append(linear_regressor.coef_[0])
#         intercepts.append(linear_regressor.intercept_)
        
#         # Predict values
#         latency_pred = linear_regressor.predict(X)
        
#         # Plot linear fit for each segment
#         axes[1].plot(cdf[start_idx:end_idx], sorted_latencies[start_idx:end_idx], marker='o', linestyle='-', linewidth=5, label=f'Computed CDF Data - Segment {i+1}')
#         axes[1].plot(cdf[start_idx:end_idx], latency_pred, linestyle='--', linewidth=5, label=f'Linear Fit Segment {i+1}')
    
#     axes[1].set_xlabel('CDF')
#     axes[1].set_ylabel('Latency (s)')
#     axes[1].set_title(f'{model} - Piecewise Linear Fit of CDF for Request Rate {request_rate} RPM')
#     axes[1].legend()
#     axes[1].grid(True)
    
#     plt.savefig(f'{model}_rr_{request_rate}_Latency_Piecewise_Linear_Fit.png')
    
#     # Print equations including request rate
#     for i in range(len(slopes)):
#         slope = slopes[i]
#         intercept = intercepts[i]
#         print(f"Segment {i+1} Equation: Latency = {slope} * CDF + {intercept}")
    
#     return slopes, intercepts

# # Select model and request rate for analysis
# model = "llama3"

# if model == "llama3":
#     rr = 20
#     op = 60
#     _, _, tgis_latency_df, _ = llama_no_priority_exp.get_stats(rr, op, 'Alan')
# elif model == "granite":
#     rr = 55
#     op = 60
#     _, _, tgis_latency_df, _ = granite_no_priority_exp.get_stats(rr, op, 'Alan')
# elif model == "bloom":
#     rr = 135
#     op = 60
#     _, _, tgis_latency_df, _ = bloom_no_priority_exp.get_stats(rr, op, 'Alan')
# else:
#     print(f"Model {model} does not exist.\n")

# # Prepare latency data for analysis
# tgis_latency_df = pd.DataFrame(tgis_latency_df, columns=['latency'])
# # Fit piecewise linear model on the CDF of latency
# slopes, intercepts = fit_piecewise_linear_model(tgis_latency_df, rr, model)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import llama_no_priority_exp
import granite_no_priority_exp
import bloom_no_priority_exp

def compute_cdf_from_hist(latency_values, bins=1000):
    """
    Compute the CDF from histogram of latency values.
    """
    counts, bin_edges = np.histogram(latency_values, bins=bins, density=True)
    cdf = np.cumsum(counts)
    cdf = cdf / cdf[-1]  # Normalize to get values between 0 and 1
    return bin_edges[1:], cdf  # bin_edges[1:] to use the upper edges of bins

def fit_piecewise_linear_model(latency_df, request_rate, model, num_segments=4):
    """
    Fit a piecewise linear model on the CDF of latency.
    """
    latency_values = latency_df['latency'].dropna().values

    # Compute CDF
    sorted_latencies, cdf = compute_cdf_from_hist(latency_values)
    
    # Compute Normalized Rate
    normalized_rate = cdf * request_rate
    
    # Define breakpoints for segments
    segment_breaks = np.linspace(0, len(normalized_rate), num_segments+1).astype(int)
    
    # Plot CDF using histogram
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    counts, bin_edges, _ = axes[0].hist(latency_values, bins=1000, cumulative=True, density=True, histtype='step', linewidth=2, label='TGIS Latency')
    axes[0].set_xlabel('Latency (s)')
    axes[0].set_ylabel('CDF')
    axes[0].set_title(f'{model} - CDF of Latency for Request Rate {request_rate} RPM')
    axes[0].legend()
    axes[0].grid(True)
    
    slopes = []
    intercepts = []
    
    # Fit linear models for each segment
    for i in range(num_segments):
        start_idx = segment_breaks[i]
        end_idx = segment_breaks[i+1]
        
        X = normalized_rate[start_idx:end_idx].reshape(-1, 1)
        y = sorted_latencies[start_idx:end_idx]
        
        linear_regressor = LinearRegression()
        linear_regressor.fit(X, y)
        
        slopes.append(linear_regressor.coef_[0])
        intercepts.append(linear_regressor.intercept_)
        
        # Predict values
        latency_pred = linear_regressor.predict(X)
        
        # Plot linear fit for each segment
        axes[1].plot(normalized_rate[start_idx:end_idx], sorted_latencies[start_idx:end_idx], marker='o', linestyle='-', linewidth=5, label=f'Computed CDF Data - Segment {i+1}')
        axes[1].plot(normalized_rate[start_idx:end_idx], latency_pred, linestyle='--', linewidth=5, label=f'Linear Fit Segment {i+1}')
    
    axes[1].set_xlabel('Normalized Rate')
    axes[1].set_ylabel('Latency (s)')
    axes[1].set_title(f'{model} - Piecewise Linear Fit of CDF for Request Rate {request_rate} RPM')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.savefig(f'normalized_{model}_rr_{request_rate}_Latency_Piecewise_Linear_Fit.png')
    
    # Print equations including request rate
    for i in range(len(slopes)):
        slope = slopes[i]
        intercept = intercepts[i]
        print(f"Segment {i+1} Equation: Latency = {slope} * Normalized Rate + {intercept}")
    
    return slopes, intercepts

# Select model and request rate for analysis
model = "llama3"

if model == "llama3":
    rr = 14
    op = 60
    _, _, tgis_latency_df, _ = llama_no_priority_exp.get_stats(12, op, 'Alan')
elif model == "granite":
    rr = 55
    op = 60
    _, _, tgis_latency_df, _ = granite_no_priority_exp.get_stats(rr, op, 'Alan')
elif model == "bloom":
    rr = 135
    op = 60
    _, _, tgis_latency_df, _ = bloom_no_priority_exp.get_stats(rr, op, 'Alan')
else:
    print(f"Model {model} does not exist.\n")

# Prepare latency data for analysis
tgis_latency_df = pd.DataFrame(tgis_latency_df, columns=['latency'])
# Fit piecewise linear model on the CDF of latency
slopes, intercepts = fit_piecewise_linear_model(tgis_latency_df, rr, model)
