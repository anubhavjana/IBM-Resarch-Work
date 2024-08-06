import os
import torch
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from torch.autograd import Variable
from matplotlib.pyplot import figure

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')
import urllib3
urllib3.disable_warnings()

model = 'vllm-granite'


if(model=='vllm-granite'):


    tgis_data_folder_180 = 'vllm_benchmark_data/vllm_rr_180_granite/tgis_vllm/'
    tgis_data_folder_240 = 'vllm_benchmark_data/vllm_rr_240_granite/tgis_vllm/'
    tgis_data_folder_300 = 'vllm_benchmark_data/vllm_rr_300_granite/tgis_vllm/'
    tgis_data_folder_400 = 'vllm_benchmark_data/vllm_rr_400_granite/tgis_vllm/'

if(model=='vllm-llama'):

    tgis_data_folder_60 = 'vllm_benchmark_data/vllm_rr_60_llama/tgis_vllm/'
    tgis_data_folder_70 = 'vllm_benchmark_data/vllm_rr_70_llama/tgis_vllm/'
    tgis_data_folder_80 = 'vllm_benchmark_data/vllm_rr_80_llama/tgis_vllm/'
    tgis_data_folder_90 = 'vllm_benchmark_data/vllm_rr_90_llama/tgis_vllm/'
    tgis_data_folder_120 = 'vllm_benchmark_data/vllm_rr_120_llama/tgis_vllm/'



tgi_columns = ['timestamp', 'job', "namespace", "pod", "value"]
tgi_merge_columns = ['timestamp', 'job', "namespace", "pod"]

tgi_filters = {
    'vllm-granite': {'pod':['vllm-granite']},
    'vllm-llama': {'pod':['vllm-llama3']}
}


def get_tgi_metrics(metric_name, columns, filter_pod, folder):
    files = os.listdir(folder)
    dataframes = []
    for i in files:
        if i.startswith(metric_name):
            temp = pd.read_csv(f'{folder}{i}')
            temp = temp[columns]
            dataframes.append(temp)
    result = pd.concat(dataframes)
    for key in filter_pod:
        result = result[result[key].isin(filter_pod[key])]
    result.drop_duplicates(inplace=True)
    return result.reset_index(drop=True)

def get_tgi_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter, folder):
    tgi_df = get_tgi_metrics(metrics[0], tgi_columns, tgi_filter, folder)
    tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_metrics(metric, tgi_columns, tgi_filter, folder)
        tgi_df = pd.merge(tgi_df, temp_df, on=tgi_merge_columns)
        tgi_df.rename(columns={"value":metric}, inplace=True)
    return tgi_df

def clean_data(df, counter_list):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df.reset_index(inplace=True, drop=True)
    return df

request_rates = {
    'vllm-granite': [180, 240, 300, 400],
    'vllm-llama': [60, 70, 80, 90, 120]
}


tgi_metrics = [
    "tgi_request_generated_tokens_sum",
    "tgi_request_generated_tokens_count",
    "tgi_batch_current_size",
    "tgi_request_duration_sum",
    "tgi_request_duration_count",
    "tgi_request_queue_duration_sum",
    "tgi_request_queue_duration_count",
    "tgi_queue_size",
    "tgi_request_input_length_sum",
    "tgi_request_input_length_count",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:avg_prompt_throughput_toks_per_s",
    "vllm:avg_generation_throughput_toks_per_s"
]

gpu_counters = [
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"
]

tgi_counters = [
    "tgi_request_generated_tokens_sum",
    "tgi_request_duration_sum",
    "tgi_request_queue_duration_sum",
    "tgi_queue_size",
    "tgi_request_duration_count",
    "tgi_request_generated_tokens_count",
    "tgi_request_input_length_sum",
    "tgi_request_input_length_count",
    "tgi_request_queue_duration_count",
    "tgi_request_count",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:avg_prompt_throughput_toks_per_s",
    "vllm:avg_generation_throughput_toks_per_s"
]

results = []

for model, rates in request_rates.items():
    tgi_filter = tgi_filters[model]
    for rate in rates:
        folder = f'vllm_benchmark_data/{model}_rr_{rate}_{model.split("-")[1]}/tgis_vllm/'
        tgi_df = get_tgi_df(tgi_metrics, tgi_columns, tgi_merge_columns, tgi_filter, folder)
        tgi_df = clean_data(tgi_df, tgi_metrics)
        
        tgi_data = tgi_df[['timestamp','vllm:gpu_cache_usage_perc','vllm:num_requests_running','tgi_request_queue_duration_sum', 'tgi_request_queue_duration_count','tgi_batch_current_size', 'tgi_queue_size', 'tgi_request_generated_tokens_sum', 'tgi_request_generated_tokens_count',  'tgi_request_duration_sum', 'tgi_request_duration_count','tgi_request_input_length_sum','tgi_request_input_length_count']]

        columns = ['tgi_request_duration_count','tgi_request_generated_tokens_sum','tgi_request_duration_sum','tgi_request_input_length_sum','tgi_request_input_length_count','tgi_request_generated_tokens_count']

        temp = tgi_data[columns]

        for col in columns:
            temp[col] = pd.to_numeric(temp[col])

        tgi_queue_size_values = tgi_data['tgi_queue_size']
        average_tgi_queue_size = tgi_queue_size_values.mean()
        percentile_95_queue_size = tgi_queue_size_values.quantile(0.95)

        tgi_batch_size_values = tgi_data['tgi_batch_current_size']
        vllm_batch_size_values = tgi_data['vllm:num_requests_running']
        average_tgi_batch_size = tgi_batch_size_values.mean()
        percentile_95_batch_size = tgi_batch_size_values.quantile(0.95)
        percentile_95_batch_size_vllm = vllm_batch_size_values.quantile(0.95)

        temp['queue_duration_count'] = tgi_data['tgi_request_queue_duration_count'].diff()
        temp['queue_duration_sum'] = tgi_data['tgi_request_queue_duration_sum'].diff()
        temp['mean_queue_time'] = temp['queue_duration_sum'] / temp['queue_duration_count']
        mean_queue_time = temp['mean_queue_time'].mean()
        percentile_95_queue_waiting_time = temp['mean_queue_time'].quantile(0.95)

        temp['request_duration_sum'] = tgi_data['tgi_request_duration_sum'].diff()
        temp['request_duration_count'] = tgi_data['tgi_request_duration_count'].diff()
        temp['e2e_latency'] = temp['request_duration_sum'] / temp['request_duration_count']
        mean_e2e_latency = temp['e2e_latency'].mean()
        percentile_95_e2e_latency = temp['e2e_latency'].quantile(0.95)

        results.append({
            'rate': rate,
            'average_tgi_queue_size': average_tgi_queue_size,
            'percentile_95_queue_size': percentile_95_queue_size,
            'average_tgi_batch_size': average_tgi_batch_size,
            'percentile_95_batch_size': percentile_95_batch_size,
            'percentile_95_batch_size_vllm': percentile_95_batch_size_vllm,
            'mean_queue_time': mean_queue_time,
            'percentile_95_queue_waiting_time': percentile_95_queue_waiting_time,
            'mean_e2e_latency': mean_e2e_latency,
            'percentile_95_e2e_latency': percentile_95_e2e_latency
        })

for result in results:
    print(f"Request Rate: {result['rate']} RPM")
    print(f"Mean TGIS Queue Size: {result['average_tgi_queue_size']}")
    print(f"95 percentile TGIS Mean Queue Size: {result['percentile_95_queue_size']}")
    print(f"95 percentile Mean Batch Size (vLLM) : {result['percentile_95_batch_size_vllm']}")
    print(f"95 percentile Mean Batch Size (TGIS) : {result['percentile_95_batch_size']}")
    print(f"Mean Batch Size : {result['average_tgi_batch_size']}")
    print(f"Mean TGIS Queue Waiting Time: {result['mean_queue_time']}")
    print(f"95 percentile Mean TGIS Queue Waiting Time: {result['percentile_95_queue_waiting_time']}")
    print(f"Mean E2E Latency: {result['mean_e2e_latency']}")
    print(f"95 percentile E2E Latency: {result['percentile_95_e2e_latency']}")
    print()

    plt.hist(temp['mean_queue_time'], cumulative=True, bins=1000, alpha=0.5, label=f'{result["rate"]} RPM')
    plt.hist(tgi_data['tgi_queue_size'], cumulative=True, bins=1000, alpha=0.5, label=f'{result["rate"]} RPM')
    plt.hist(temp['e2e_latency'], cumulative=True, bins=1000, alpha=0.5, label=f'{result["rate"]} RPM')

plt.xlabel("Metrics")
plt.ylabel("Samples")
plt.legend(loc='upper right')
plt.title("CDF of Queue Time, Queue Size, and E2E Latency for Different Request Rates")
plt.show()




gpu_filter = {'exported_pod':['vllm-llama3']}

tgi_filter = {'pod':['vllm-llama3']}


tgi_df = get_tgi_df(tgi_metrics, tgi_columns, tgi_merge_columns, tgi_filter)
tgi_df = clean_data(tgi_df, tgi_counters)





tgi_data = tgi_df[['timestamp','vllm:gpu_cache_usage_perc','vllm:num_requests_running','tgi_request_queue_duration_sum', 'tgi_request_queue_duration_count','tgi_batch_current_size', 'tgi_queue_size', 'tgi_request_generated_tokens_sum', 'tgi_request_generated_tokens_count',  'tgi_request_duration_sum', 'tgi_request_duration_count','tgi_request_input_length_sum','tgi_request_input_length_count']]

columns = ['tgi_request_duration_count','tgi_request_generated_tokens_sum','tgi_request_duration_sum','tgi_request_input_length_sum','tgi_request_input_length_count','tgi_request_generated_tokens_count']

temp = tgi_data[columns]

for col in columns:
    temp[col] = pd.to_numeric(temp[col])





#############################################################################################################

tgi_queue_size_values = tgi_data['tgi_queue_size']
percentile_95_queue_size = tgi_queue_size_values.quantile(0.95)
average_tgi_queue_size = tgi_queue_size_values.mean()
print("Mean TGIS Queue Size: ", average_tgi_queue_size)
print("95 percentile TGIS Mean Queue Size: ",percentile_95_queue_size)


# ############################################################################################################

tgi_batch_size_values = tgi_data['tgi_batch_current_size']
vllm_batch_size_values = tgi_data['vllm:num_requests_running']
percentile_95_batch_size = tgi_batch_size_values.quantile(0.95)
percentile_95_batch_size_vllm = vllm_batch_size_values.quantile(0.95)
average_tgi_batch_size = tgi_batch_size_values.mean()

print("95 percentile Mean Batch Size (vLLM) : ",percentile_95_batch_size_vllm)
print("95 percentile Mean Batch Size (TGIS) : ",percentile_95_batch_size)
print("Mean Batch Size : ", average_tgi_batch_size)

############################################################################################################


temp['queue_duration_count'] = tgi_data['tgi_request_queue_duration_count'].diff()
temp['queue_duration_sum'] = tgi_data['tgi_request_queue_duration_sum'].diff()

temp['mean_queue_time'] = temp['queue_duration_sum'] / temp['queue_duration_count']
mean_queue_time = temp['mean_queue_time'].mean()
percentile_95_queue_waiting_time = temp['mean_queue_time'].quantile(0.95)
print("Mean TGIS Queue Waiting Time: ",mean_queue_time)
print("95 percentile Mean TGIS Queue Waiting Time: ",percentile_95_queue_waiting_time)

############################################################################################################

temp['request_duration_sum'] = tgi_data['tgi_request_duration_sum'].diff()
temp['request_duration_count'] = tgi_data['tgi_request_duration_count'].diff()
temp['e2e_latency'] = temp['request_duration_sum'] / temp['request_duration_count']
mean_e2e_latency = temp['e2e_latency'].mean()
percentile_95_e2e_latency = temp['e2e_latency'].quantile(0.95)
print("Mean E2E Latency: ",mean_e2e_latency)
print("95 percentile E2E Latency: ", percentile_95_e2e_latency)

############################################################################################################


# temp['total_output_request_length'] = temp["tgi_request_generated_tokens_sum"].diff()
# temp['total_output_request_count'] = temp["tgi_request_generated_tokens_count"].diff()
# temp["mean_output_tokens_count"] = temp['total_output_request_length'] / temp['total_output_request_count']
# print("Mean Number of Output tokens: ",temp["mean_output_tokens_count"].mean())

##############################################################################################################


# # ##################################### AVERAGE INPUT TOKENS COUNT #####################################
# temp['total_input_request_length'] = temp['tgi_request_input_length_sum'].diff()
# temp['total_input_request_count'] = temp['tgi_request_input_length_count'].diff()
# temp["mean_input_tokens_count"] = temp['total_input_request_length'] / temp['total_input_request_count']
# print("----------Mean Number of input tokens----------",temp["mean_input_tokens_count"].mean())
# ###############################################################################################################



########################################################################################################################

plt.hist(temp['mean_queue_time'], cumulative=True, bins=1000)
plt.xlabel("TGIS Mean Queue Time")
plt.ylabel("Samples")
plt.title("Request rate : 70 RPM (Llama3 vLLM)")
plt.show()



plt.hist(tgi_data['tgi_queue_size'], cumulative=True, bins=1000)
plt.xlabel("Mean Queue Size")
plt.ylabel("Samples")
plt.title("Request rate : 70 RPM (Llama3 vLLM)")
plt.show()

plt.hist(temp['e2e_latency'], cumulative=True, bins=1000)
plt.xlabel("Mean E2E Latency (in sec)")
plt.ylabel("Samples")
plt.title("Request rate : 70 RPM (Llama3 vLLM)")
plt.show()



########################################################################################################################
