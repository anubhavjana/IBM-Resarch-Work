import os
import torch
import numpy as np
import pandas as pd
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

tgis_llama1_data_folder = 'router_multi_instance/multi_pod_10.15.20/tgis_llama1/'
tgis_llama2_data_folder = 'router_multi_instance/multi_pod_10.15.20/tgis_llama2/'
tgis_llama3_data_folder = 'router_multi_instance/multi_pod_10.15.20/tgis_llama3/'

router_data_folder = 'router_multi_instance/multi_pod_10.15.20/router/'

tgi_user_columns = ['timestamp', 'job', "namespace", "pod", "value", "user"]
tgi_user_merge_columns = ['timestamp', 'job', "namespace", "pod", "user"]

tgi_columns = ['timestamp', 'job', "namespace", "pod", "value"]
tgi_merge_columns = ['timestamp', 'job', "namespace", "pod"]

router_col = ['timestamp', 'job', "namespace", "pod", "user", "value"]
router_merge_columns = ['timestamp', 'job', "namespace", "pod", "user"]

def get_tgi_llama1_metrics(metric_name, columns, filter_pod):
    files = os.listdir(tgis_llama1_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{tgis_llama1_data_folder}{i}')
            temp = temp[columns]
            dataframes.append(temp)
    result = pd.concat(dataframes)
    for key in filter_pod:
        result = result[result[key].isin(filter_pod[key])]
    result.drop_duplicates(inplace=True)
    return result.reset_index(drop=True)

def get_tgi_llama2_metrics(metric_name, columns, filter_pod):
    files = os.listdir(tgis_llama2_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{tgis_llama2_data_folder}{i}')
            temp = temp[columns]
            dataframes.append(temp)
    result = pd.concat(dataframes)
    for key in filter_pod:
        result = result[result[key].isin(filter_pod[key])]
    result.drop_duplicates(inplace=True)
    return result.reset_index(drop=True)

def get_tgi_llama3_metrics(metric_name, columns, filter_pod):
    files = os.listdir(tgis_llama3_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{tgis_llama3_data_folder}{i}')
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

def get_tgi_llama1_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter):
    tgi_df = get_tgi_llama1_metrics(metrics[0], tgi_columns, tgi_filter)
    tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_llama1_metrics(metric, tgi_columns, tgi_filter)
        tgi_df = pd.merge(tgi_df, temp_df, on = tgi_merge_columns)
        tgi_df.rename(columns={"value":metric}, inplace=True)
    return tgi_df

def get_tgi_llama2_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter):
    tgi_df = get_tgi_llama2_metrics(metrics[0], tgi_columns, tgi_filter)
    tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_llama2_metrics(metric, tgi_columns, tgi_filter)
        tgi_df = pd.merge(tgi_df, temp_df, on = tgi_merge_columns)
        tgi_df.rename(columns={"value":metric}, inplace=True)
    return tgi_df

def get_tgi_llama3_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter):
    tgi_df = get_tgi_llama3_metrics(metrics[0], tgi_columns, tgi_filter)
    tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_llama3_metrics(metric, tgi_columns, tgi_filter)
        tgi_df = pd.merge(tgi_df, temp_df, on = tgi_merge_columns)
        tgi_df.rename(columns={"value":metric}, inplace=True)
    return tgi_df

def get_router_df(metrics, router_columns, router_merge_columns, router_filter):
    router_df = get_router_metrics(metrics[0], router_columns, router_filter)
    router_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_router_metrics(metric, router_columns, router_filter)
        router_df = pd.merge(router_df, temp_df, on = router_merge_columns)
        router_df.rename(columns={"value":metric}, inplace=True)
    return router_df

def clean_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df.reset_index(drop=True)

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

tgi_filter_llama_1 = {'pod':['llama3-1']}
tgi_filter_llama_2 = {'pod':['llama3-2']}
tgi_filter_llama_3 = {'pod':['llama3-3']}

router_filter = {"pod" : ['llm-router-multi-instance-v3']}

tgi_df_user_llama_1 = clean_data(get_tgi_llama1_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama_1))
tgi_df_user_llama_2 = clean_data(get_tgi_llama2_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama_2))
tgi_df_usr_llama_3 = clean_data(get_tgi_llama3_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama_3))


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

def get_stats(user):
    print(f"Fetching stats for {user}")

    tmp_llama_1 = tgi_df_user_llama_1[tgi_df_user_llama_1['user'] == user]
    tmp_llama_2 = tgi_df_user_llama_2[tgi_df_user_llama_2['user'] == user]
    tmp_llama_3 = tgi_df_usr_llama_3[tgi_df_usr_llama_3['user'] == user]

    # Llama-1
    print("Stats for LLama-1...\n")
    tgis_latency_df_llama1 = tmp_llama_1['tgi_request_duration_sum'].diff() / tmp_llama_1['tgi_request_duration_count'].diff()
    tgis_inference_df_llama1 = tmp_llama_1['tgi_request_inference_duration_sum'].diff() / tmp_llama_1['tgi_request_inference_duration_count'].diff()
    tgis_queue_time_df_llama1 = tgis_latency_df_llama1 - tgis_inference_df_llama1

    if(pd.isna(tgis_queue_time_df_llama1.mean())):
        print(f"User {user} did not send request to Llama-1")
    else:
        print(f"User {user} : Mean Queue Time: {tgis_queue_time_df_llama1.mean()}")

    if(pd.isna(tgis_latency_df_llama1.mean())):
        print(f"User {user} did not send request to Llama 1")
    else:
        print(f"User {user} : Mean TGIS Latency: {tgis_latency_df_llama1.mean()}")
    

    # Llama-2
    print("\nStats for LLama-2...\n")
    tgis_latency_df_llama2 = tmp_llama_2['tgi_request_duration_sum'].diff() / tmp_llama_2['tgi_request_duration_count'].diff()
    tgis_inference_df_llama2 = tmp_llama_2['tgi_request_inference_duration_sum'].diff() / tmp_llama_2['tgi_request_inference_duration_count'].diff()
    tgis_queue_time_df_llama2 = tgis_latency_df_llama2 - tgis_inference_df_llama2

    if(pd.isna(tgis_queue_time_df_llama2.mean())):
        print(f"User {user} did not send request to Llama-2")
    else:
        print(f"User {user} : Mean Queue Time: {tgis_queue_time_df_llama2.mean()}")

    if(pd.isna(tgis_latency_df_llama2.mean())):
        print(f"User {user} did not send request to Llama-2")
    else:
        print(f"User {user} : Mean TGIS Latency: {tgis_latency_df_llama2.mean()}")

    print(f'90 %ile latency is: {tgis_latency_df_llama2.quantile(0.90)}')

    print(f'90 %ile QUEUE TIME is: {tgis_queue_time_df_llama2.quantile(0.90)}')

   
    # Llama-3
    print("\nStats for LLama-3...\n")
    tgis_latency_df_llama3 = tmp_llama_3['tgi_request_duration_sum'].diff() / tmp_llama_3['tgi_request_duration_count'].diff()
    tgis_inference_df_llama3 = tmp_llama_3['tgi_request_inference_duration_sum'].diff() / tmp_llama_3['tgi_request_inference_duration_count'].diff()
    tgis_queue_time_df_llama3 = tgis_latency_df_llama3 - tgis_inference_df_llama3

    if(pd.isna(tgis_queue_time_df_llama3.mean())):
        print(f"User {user} did not send request to Llama-3")
    else:
        print(f"User {user} : Mean Queue Time: {tgis_queue_time_df_llama3.mean()}")

    if(pd.isna(tgis_latency_df_llama3.mean())):
        print(f"User {user} did not send request to Llama-3")
    else:
        print(f"User {user} : Mean TGIS Latency: {tgis_latency_df_llama3.mean()}")


    router_df = clean_data(get_router_df(router_metrics, router_col, router_merge_columns, router_filter))
    router_tmp = router_df[router_df['user']==user]

    router_request_count_df  = router_tmp['router_request_count'].diff()
    router_request_count = router_request_count_df.sum()

    router_request_failure_df = router_tmp['router_request_failure'].diff()
    router_request_failure_count = router_request_failure_df.sum()

    router_request_success_df = router_tmp['router_request_success'].diff()
    router_request_success_count = router_request_success_df.sum()

    tgi_llama1_request_count_df = tmp_llama_1['tgi_request_count'].diff()
    tgi_llama1_request_count = tgi_llama1_request_count_df.sum()

    tgi_llam2_request_count_df = tmp_llama_2['tgi_request_count'].diff()
    tgi_llama2_request_count = tgi_llam2_request_count_df.sum()

    tgi_llama3_request_count_df = tmp_llama_3['tgi_request_count'].diff()
    tgi_llama3_request_count = tgi_llama3_request_count_df.sum()

    queue_data = {
        'Llama1 Queue Time': tgis_queue_time_df_llama1.dropna(),
        'Llama2 Queue Time': tgis_queue_time_df_llama2.dropna(),
        'Llama3 Queue Time': tgis_queue_time_df_llama3.dropna()
    }

    latency_data = {
        'Llama1 Latency': tgis_latency_df_llama1.dropna(),
        'Llama2 Latency': tgis_latency_df_llama2.dropna(),
        'Llama3 Latency': tgis_latency_df_llama3.dropna()
    }

    # Plot histograms with CDFs
    plot_histogram_with_cdf(queue_data, 'Queue Time (ms)', 'Cumulative Probability', f'Queue Time CDF for User {user}', f'images/queue_time_cdf_user_{user}.png')
    plot_histogram_with_cdf(latency_data, 'Latency (ms)', 'Cumulative Probability', f'Latency CDF for User {user}', f'images/latency_cdf_user_{user}.png')

    print(f"Generated CDF plots for User {user}")



users = ['Alan', 'Noel', 'Hari']

for user in users:
    get_stats(user)
    print('-------------------------------------------------------------------------------------------------------------------------------------\n')

print("------Completed---------")
