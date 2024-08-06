import os
import torch
import numpy as np
import pandas as pd
from datetime import timedelta

from scipy import stats
import matplotlib.pyplot as plt
from torch.autograd import Variable

from matplotlib.pyplot import figure
pd.options.mode.chained_assignment = None


import warnings
warnings.filterwarnings('ignore')

import urllib3
urllib3.disable_warnings()


rr = 6

tgis_llama_data_folder = 'router_multi_instance/llama_a100_4.4.4/tgis/'

router_data_folder = 'router_multi_instance/llama_a100_4.4.4/router/'

tgi_user_columns = ['timestamp', 'job', "namespace", "pod", "value", "user"]
tgi_user_merge_columns = ['timestamp', 'job', "namespace", "pod", "user"]

tgi_columns = ['timestamp', 'job', "namespace", "pod", "value"]
tgi_merge_columns = ['timestamp', 'job', "namespace", "pod"]


router_col = ['timestamp', 'job', "namespace", "pod", "user", "value"]
router_merge_columns = ['timestamp', 'job', "namespace", "pod", "user"]


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


def get_router_metrics(metric_name, columns, filter_pod):
    files = os.listdir(router_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{router_data_folder}{i}')
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



def get_router_df(metrics, router_columns, router_merge_columns, router_filter,start_offset=0, end_offset=0):
    router_df = get_router_metrics(metrics[0], router_columns, router_filter)
    router_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_router_metrics(metric, router_columns, router_filter)
        router_df = pd.merge(router_df, temp_df, on = router_merge_columns)
        router_df.rename(columns={"value":metric}, inplace=True)

    router_df = filter_timestamps(router_df, start_offset, end_offset)
    return router_df


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

# tgis with user metrics 
tgi_metrics_with_user = [
    "tgi_request_generated_tokens_sum",
    "tgi_request_generated_tokens_count",
    "tgi_request_duration_sum",
    "tgi_request_duration_count",
    "tgi_request_count",
    "tgi_request_inference_duration_count",
    "tgi_request_inference_duration_sum",
    "tgi_request_success"
   
   
]

tgi_metrics_without_user = [
    "tgi_queue_size",
    "tgi_request_queue_duration_count",
    "tgi_request_queue_duration_sum",
   
    
    "tgi_request_duration_sum",
    "tgi_request_duration_count"

]

router_metrics = [
    "router_request_duration_sum",
    "router_request_duration_count",
    "router_queue_duration_sum",
    "router_queue_duration_count",
    "router_request_count",
    "router_request_failure",
    "router_request_success"
]


tgi_filter_llama = {'pod':['llama3']}

router_filter = {"pod" : ['llm-router-multi-instance-v3']} 

tgi_df_user_llama = clean_data(get_tgi_llama_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama))


def get_stats(user):

    print(f"Stats for USER: {user}")


    tmp_llama = tgi_df_user_llama[tgi_df_user_llama['user'] == user]
    tmp_llama.dropna(inplace=True)


    print("Stats for LLama-A100...\n")

    tgis_latency_df = tmp_llama['tgi_request_duration_sum'].diff() / tmp_llama['tgi_request_duration_count'].diff()
    tgis_inference_df = tmp_llama['tgi_request_inference_duration_sum'].diff() / tmp_llama['tgi_request_inference_duration_count'].diff()
    tgis_queue_time_df = tgis_latency_df - tgis_inference_df

    print("------------------------------------------")
    print(f'95 %ile latency is: {tgis_latency_df.quantile(0.95)}')
    print(f'90 %ile latency is: {tgis_latency_df.quantile(0.90)}')
    print(f'80 %ile latency is: {tgis_latency_df.quantile(0.80)}')
    print("------------------------------------------")
    print(f'95 %ile QUEUE TIME is: {tgis_queue_time_df.quantile(0.95)}')
    print(f'90 %ile QUEUE TIME is: {tgis_queue_time_df.quantile(0.90)}')
    print(f'80 %ile QUEUE TIME is: {tgis_queue_time_df.quantile(0.80)}')
    print("------------------------------------------\n")
    if(pd.isna(tgis_queue_time_df.mean())):
        print(f"User {user} did not send request to Llama")
    else:
        print(f"User {user} : Mean Queue Time: {tgis_queue_time_df.mean()}")


    if(pd.isna(tgis_latency_df.mean())):
        print(f"User {user} did not send request to Llama")
    else:
        print(f"User {user} : Mean TGIS Latency: {tgis_latency_df.mean()}")


    router_df = clean_data(get_router_df(router_metrics, router_col, router_merge_columns, router_filter))
    router_tmp = router_df[router_df['user']==user]

    router_e2e_latency_df = router_tmp['router_request_duration_sum'].diff()/router_tmp['router_request_duration_count'].diff()
    print(f"\nUser {user} : Mean Router E2E Latency: {router_e2e_latency_df.mean()}")

    router_queue_time_df = router_tmp['router_queue_duration_sum'].diff()/router_tmp['router_queue_duration_count'].diff()
    print(f"\nUser {user} : Mean Router Queue Time: {router_queue_time_df.mean()}")

    queue_data = {
        'Llama MiG Queue Time': tgis_queue_time_df.dropna(),
        
        
    }

    latency_data = {
        'Llama MiG Latency': tgis_latency_df.dropna(),
        
    }

    # plot_histogram_with_cdf(queue_data, 'Queue Time (ms)', 'Cumulative Probability', f'Queue Time CDF for User {user}', f'tgis_same_pod_queue_time_cdf_user_{user}.png')
    # plot_histogram_with_cdf(latency_data, 'Latency (ms)', 'Cumulative Probability', f'Latency CDF for User {user}', f'tgis_same_pod_latency_cdf_user_{user}.png')



def plot_histogram_with_cdf(data, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for label, df in data.items():
        plt.hist(df, bins=1000, cumulative=True, density=True, histtype='step', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

users = ['Alan', 'Noel', 'Hari']

for user in users: 
    get_stats(user)
    print('-------------------------------------------------------------------------------------------------------------------------------------\n')


print("------Completed---------")
