import os
import torch
import numpy as np
import pandas as pd
from datetime import timedelta

import argparse


from scipy import stats
import matplotlib.pyplot as plt
from torch.autograd import Variable
from scipy.optimize import curve_fit


from matplotlib.pyplot import figure
pd.options.mode.chained_assignment = None


import warnings
warnings.filterwarnings('ignore')

import urllib3
urllib3.disable_warnings()


rr = 90
model = "llama"

# tgis_llama_data_folder = f'vllm_benchmark_data/vllm_rr_{rr}_{model}/tgis_vllm/'

tgis_llama_data_folder = f'vllm_benchmark_data/vllm_rr_{rr}_{model}_new/tgis_vllm/'

print(f"Reading file : {tgis_llama_data_folder}\n")

# Without priority is same as sending a single user 

tgi_columns = ['timestamp', 'job', "namespace", "pod", "value"]
tgi_merge_columns = ['timestamp', 'job', "namespace", "pod"]

tgi_priority_columns = ['timestamp', 'job', "namespace", "pod", "priority", "value"]
tgi_priority_merge_columns = ['timestamp', 'job', "namespace", "pod", "priority"]


def get_tgi_llama_metrics(metric_name, columns, filter_pod):
    files = os.listdir(tgis_llama_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{tgis_llama_data_folder}{i}')
            temp = temp[columns]
            dataframes.append(temp)
    result = pd.concat(dataframes)
    for key in filter_pod:
        result = result[result[key].isin(filter_pod[key])]
    result.drop_duplicates(inplace=True)
    return result.reset_index(drop=True)


def get_tgi_llama_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter,start_offset=0, end_offset=0):
    tgi_df = get_tgi_llama_metrics(metrics[0], tgi_columns, tgi_filter)
    tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_llama_metrics(metric, tgi_columns, tgi_filter)
        tgi_df = pd.merge(tgi_df, temp_df, on = tgi_merge_columns)
        tgi_df.rename(columns={"value":metric}, inplace=True)

    tgi_df = filter_timestamps(tgi_df, start_offset, end_offset)
    return tgi_df



def filter_timestamps(df, start_offset, end_offset):

    df['timestamp'] = pd.to_datetime(df['timestamp']) # change to datetime 
    
    # Get the minimum and maximum timestamps
    min_timestamp = df['timestamp'].min()
    max_timestamp = df['timestamp'].max()
    
    start_cutoff = min_timestamp + timedelta(minutes=start_offset)  # +8 minutes
    end_cutoff = max_timestamp - timedelta(minutes=end_offset)      # -3 minutes
    
    # Filter the DataFrame
    filtered_df = df[(df['timestamp'] >= start_cutoff) & (df['timestamp'] <= end_cutoff)]
    
    return filtered_df


def clean_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df.reset_index(drop=True)


tgi_metrics = [
    "tgi_request_generated_tokens_sum",
    "tgi_request_generated_tokens_count",
    # "tgi_batch_current_size",
    "tgi_request_duration_sum",
    "tgi_request_duration_count",
    # "tgi_request_queue_duration_sum",
    # "tgi_request_queue_duration_count",
    # "tgi_queue_size",
    # "tgi_request_input_length_sum",
    # "tgi_request_input_length_count",
    # "vllm:num_requests_running",
    # "vllm:num_requests_waiting",
    # "vllm:num_requests_swapped",
    # "vllm:gpu_cache_usage_perc",
    # "vllm:cpu_cache_usage_perc",
    # "vllm:num_preemptions_total",
    # "vllm:prompt_tokens_total",
    # "vllm:generation_tokens_total",
    # "vllm:request_success_total",
    # "vllm:avg_prompt_throughput_toks_per_s",
    # "vllm:avg_generation_throughput_toks_per_s"
]

vllm_priority_metrics = ["vllm:e2e_request_latency_seconds_sum","vllm:e2e_request_latency_seconds_count"]

tgi_filter_llama_priority = {'pod':['vllm-llama']}

tgi_filter_llama_old = {'pod':['vllm-llama3']}


tgi_df = clean_data(get_tgi_llama_df(vllm_priority_metrics, tgi_priority_columns, tgi_priority_merge_columns, tgi_filter_llama_priority)) # experiment with no priority - single user

# tgi_df = clean_data(get_tgi_llama_df(tgi_metrics, tgi_columns, tgi_merge_columns, tgi_filter_llama_old)) # experiment with no priority - no user (old data)


priority = 1
tmp_llama = tgi_df[tgi_df['priority'] == priority]


e2e_latency_df = tmp_llama['vllm:e2e_request_latency_seconds_sum'].diff() / tmp_llama['vllm:e2e_request_latency_seconds_count'].diff()

mean_e2e_latency = e2e_latency_df.mean()
percentile_95_e2e_latency = e2e_latency_df.quantile(0.95)
print(f'VLLM Mean E2E Latency Model = {model} Request Rate = {rr}  ----> {mean_e2e_latency}')
print(f'VLLM Mean 95 %ile E2E Latency Model = {model} Request Rate {rr}   ---> {percentile_95_e2e_latency}\n')


# e2e_latency_df = tgi_df['tgi_request_duration_sum'].diff() / tgi_df['tgi_request_duration_count'].diff()
# output_tokens_df = tgi_df['tgi_request_generated_tokens_sum'].diff() / tgi_df['tgi_request_generated_tokens_count'].diff()

# mean_e2e_latency = e2e_latency_df.mean()
# percentile_95_e2e_latency = e2e_latency_df.quantile(0.95)

# print(f'VLLM Mean E2E Latency for Model = {model} Request Rate = {rr} ----> {mean_e2e_latency}')
# print(f'VLLM Mean 95 %ile E2E Latency for Model = {model} Request Rate {rr} ---> {percentile_95_e2e_latency}\n')


# print(f'VLLM Mean Output Token Count for Model = {model} Request Rate = {rr} ----> {output_tokens_df.mean()}')
# print(f'VLLM 95 %ile Output Token Count for Model = {model} Request Rate {rr} ---> {output_tokens_df.quantile(0.95)}\n')

# PLOTTING CDF 

fig, axes = plt.subplots(3, 1, figsize=(10, 15))

axes[0].hist(e2e_latency_df, bins=1000, cumulative=True, density=True, histtype='step', label='TGIS Latency')
axes[0].set_xlabel('Latency (s)')
axes[0].set_ylabel('Cumulative Probability')
axes[0].set_title(f'CDF of VLLM E2E Latency (No Priority) - RR = {rr} OP = 60')
axes[0].legend(loc='upper left')
axes[0].grid(True)

plt.tight_layout()
# plt.savefig(f'LLama_CDF__rr_{rr}_op_{op}.png')
plt.show()

# axes[0].hist(mean_latency_per_tok, bins=1000, cumulative=True, density=True, histtype='step', label='Mean Latency Per Token')
# axes[0].set_xlabel('Latency (s)')
# axes[0].set_ylabel('Cumulative Probability')
# axes[0].set_title('CDF of Mean Latency Per Token')
# axes[0].legend(loc='upper left')
# axes[0].grid(True)

# axes[1].hist(tgis_queue_time_df, bins=1000, cumulative=True, density=True, histtype='step', label='Queue Time')
# axes[1].set_xlabel('Queue Time')
# axes[1].set_ylabel('Cumulative Probability')
# axes[1].set_title('CDF of Queue Time')
# axes[1].legend(loc='upper left')
# axes[1].grid(True)

# axes[2].hist(e2e_latency_df, bins=1000, cumulative=True, density=True, histtype='step', label='TGIS Latency')
# axes[2].set_xlabel('Latency (s)')
# axes[2].set_ylabel('Cumulative Probability')
# axes[2].set_title(f'CDF of TGIS E2E Latency on VLLM - RR ={rr} OP = {output_tokens_df.quantile(0.95)}')
# axes[2].legend(loc='upper left')
# axes[2].grid(True)




# print("------Completed---------")
