
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


rr = 30
op = 60
model = "llama"
mode ="no_premp"

vllm_llama_data_folder = f'vllm_priority_experiments/{model}/rr_{rr}_each_op_{op}_{mode}/tgis_vllm/'

# vllm_llama_data_folder = f'vllm_priority_experiments/{model}/rr_{rr}_each_op_{op}/tgis_vllm/'

print(f"Reading file : {vllm_llama_data_folder}\n")

tgi_priority_columns = ['timestamp', 'job', "namespace", "pod", "priority", "value"]
tgi_priority_merge_columns = ['timestamp', 'job', "namespace", "pod", "priority"]

tgi_columns = ['timestamp', 'job', "namespace", "pod", "value"]
tgi_merge_columns = ['timestamp', 'job', "namespace", "pod"]


def get_tgi_llama_metrics(metric_name, columns, filter_pod):
    files = os.listdir(vllm_llama_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{vllm_llama_data_folder}{i}')
            temp = temp[columns]
            dataframes.append(temp)
    result = pd.concat(dataframes)
    for key in filter_pod:
        result = result[result[key].isin(filter_pod[key])]
    result.drop_duplicates(inplace=True)
    return result.reset_index(drop=True)


def get_tgi_llama_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter, start_offset=0, end_offset=0):
    tgi_df = get_tgi_llama_metrics(metrics[0], tgi_columns, tgi_filter)
    tgi_df.rename(columns={"value": metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_llama_metrics(metric, tgi_columns, tgi_filter)
        tgi_df = pd.merge(tgi_df, temp_df, on=tgi_merge_columns)
        tgi_df.rename(columns={"value": metric}, inplace=True)

    tgi_df = filter_timestamps(tgi_df, start_offset, end_offset)
    return tgi_df


def filter_timestamps(df, start_offset, end_offset):
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # change to datetime

    # Get the minimum and maximum timestamps
    min_timestamp = df['timestamp'].min()
    max_timestamp = df['timestamp'].max()

    start_cutoff = min_timestamp + timedelta(minutes=start_offset)  # +8 minutes
    end_cutoff = max_timestamp - timedelta(minutes=end_offset)  # -3 minutes

    # Filter the DataFrame
    filtered_df = df[(df['timestamp'] >= start_cutoff) & (df['timestamp'] <= end_cutoff)]

    return filtered_df


def clean_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df.reset_index(drop=True)

vllm_filter_llama_priority = {'pod':['vllm-llama-no-preemption']}

vllm_metrics_without_priority = ["vllm:time_to_first_token_seconds_sum","vllm:time_to_first_token_seconds_count","vllm:time_per_output_token_seconds_sum",
    "vllm:time_per_output_token_seconds_count"]

vllm_metrics_priority = [
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count"]

vllm_df_priority = clean_data(get_tgi_llama_df(vllm_metrics_priority, tgi_priority_columns, tgi_priority_merge_columns, vllm_filter_llama_priority))

vllm_df_wo_priority = clean_data(get_tgi_llama_df(vllm_metrics_without_priority, tgi_columns, tgi_merge_columns, vllm_filter_llama_priority))

ttft_df = vllm_df_wo_priority['vllm:time_to_first_token_seconds_sum'].diff() / vllm_df_wo_priority['vllm:time_to_first_token_seconds_count'].diff()
tbot_df = vllm_df_wo_priority['vllm:time_per_output_token_seconds_sum'].diff() / vllm_df_wo_priority['vllm:time_per_output_token_seconds_count'].diff()


print(f'Mean TTFT: {ttft_df.mean()}')
print(f'Mean 95 %ile TTFT: {ttft_df.quantile(0.95)}\n')

print(f'Mean TBOT: {tbot_df.mean()}')
print(f'Mean 95 %ile TBOT: {tbot_df.quantile(0.95)}\n')

# Plotting the CDF for ttft_df
plt.figure(figsize=(12, 6))

# plt.hist(ttft_df.dropna(), bins=1000, cumulative=True, density=True, histtype='step', label='TTFT CDF')

# plt.xlabel('TTFT (seconds)')
# plt.ylabel('Cumulative Probability')
# plt.title(f'CDF of TTFT RR = {rr}:{rr}:{rr} with Priority')
# plt.legend(loc='upper left')
# plt.grid(True)

# plt.show()
# Get unique priorities
priorities = vllm_df_priority['priority'].unique()

# PLOTTING CDF FOR EACH PRIORITY
plt.figure(figsize=(10, 7))

for priority in priorities:
    tmp_llama = vllm_df_priority[vllm_df_priority['priority'] == priority]

    e2e_latency_df = tmp_llama['vllm:e2e_request_latency_seconds_sum'].diff() / tmp_llama['vllm:e2e_request_latency_seconds_count'].diff()

    mean_e2e_latency = e2e_latency_df.mean()
    percentile_95_e2e_latency = e2e_latency_df.quantile(0.95)

    print(f'VLLM Mean E2E Latency for Priority {priority} Model = {model} Request Rate = {rr} Each ----> {mean_e2e_latency}')
    print(f'VLLM Mean 95 %ile E2E Latency Priority {priority} for Model = {model} Request Rate {rr} Each  ---> {percentile_95_e2e_latency}\n')

    plt.hist(e2e_latency_df, bins=1000, cumulative=True, density=True, histtype='step', label=f'Priority {priority}')

plt.xlabel('Latency (s)')
plt.ylabel('Cumulative Probability')
plt.title(f'vLLM Priority: CDF of VLLM E2E Latency: RR = {rr} each')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
