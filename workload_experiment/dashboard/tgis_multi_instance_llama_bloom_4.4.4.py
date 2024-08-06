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

tgis_llama_data_folder = 'router_multi_instance/multi_pod_4.4.4/tgis_llamamig/'
tgis_bloom_data_folder = 'router_multi_instance/multi_pod_4.4.4/tgis_bloommig/'

router_data_folder = 'router_multi_instance/multi_pod_4.4.4/router/'


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


def get_tgi_bloom_metrics(metric_name, columns, filter_pod):
    files = os.listdir(tgis_bloom_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{tgis_bloom_data_folder}{i}')
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

def get_tgi_llama_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter):
    tgi_df = get_tgi_llama_metrics(metrics[0], tgi_columns, tgi_filter)
    tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_llama_metrics(metric, tgi_columns, tgi_filter)
        tgi_df = pd.merge(tgi_df, temp_df, on = tgi_merge_columns)
        tgi_df.rename(columns={"value":metric}, inplace=True)
    return tgi_df

def get_tgi_bloom_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter):
    tgi_df = get_tgi_bloom_metrics(metrics[0], tgi_columns, tgi_filter)
    tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_bloom_metrics(metric, tgi_columns, tgi_filter)
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


tgi_filter_llama = {'pod':['llama3-mig3g']}
tgi_filter_bloom = {'pod':['bloom-mig3g']}


router_filter = {"pod" : ['llm-router-multi-instance-v3']}


tgi_df_user_llama = clean_data(get_tgi_llama_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama))
tgi_df_user_bloom = clean_data(get_tgi_bloom_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_bloom))





def get_stats(user):

    print(f"Fetching stats for {user}")


    tmp_llama = tgi_df_user_llama[tgi_df_user_llama['user'] == user]
    tmp_bloom = tgi_df_user_bloom[tgi_df_user_bloom['user'] == user]





    print("Stats for LLama-Mig...\n")

    tgis_latency_df = tmp_llama['tgi_request_duration_sum'].diff() / tmp_llama['tgi_request_duration_count'].diff()
    tgis_inference_df = tmp_llama['tgi_request_inference_duration_sum'].diff() / tmp_llama['tgi_request_inference_duration_count'].diff()
    tgis_queue_time_df = tgis_latency_df - tgis_inference_df

    if(pd.isna(tgis_queue_time_df.mean())):
        print(f"User {user} did not send request to Llama-Mig")
    else:
        print(f"User {user} : Mean Queue Time: {tgis_queue_time_df.mean()}")


    if(pd.isna(tgis_latency_df.mean())):
        print(f"User {user} did not send request to Llama-Mig")
    else:
        print(f"User {user} : Mean TGIS Latency: {tgis_latency_df.mean()}")

    
    print("\nStats for Bloom-Mig...\n")

    tgis_latency_df = tmp_bloom['tgi_request_duration_sum'].diff() / tmp_bloom['tgi_request_duration_count'].diff()
    tgis_inference_df = tmp_bloom['tgi_request_inference_duration_sum'].diff() / tmp_bloom['tgi_request_inference_duration_count'].diff()
    tgis_queue_time_df = tgis_latency_df - tgis_inference_df

    if(pd.isna(tgis_queue_time_df.mean())):
        print(f"User {user} did not send request to Bloom-Mig")
    else:
        print(f"User {user} : Mean Queue Time: {tgis_queue_time_df.mean()}")


    if(pd.isna(tgis_latency_df.mean())):
        print(f"User {user} did not send request to Bloom-Mig")
    else:
        print(f"User {user} : Mean TGIS Latency: {tgis_latency_df.mean()}")


    router_df = clean_data(get_router_df(router_metrics, router_col, router_merge_columns, router_filter))
    router_tmp = router_df[router_df['user']==user]



    router_request_count_df  = router_tmp['router_request_count'].diff()

    router_request_count = router_request_count_df.sum()

    router_request_failure_df = router_tmp['router_request_failure'].diff()
    router_request_failure_count = router_request_failure_df.sum()

    router_request_success_df = router_tmp['router_request_success'].diff()
    router_request_success_count = router_request_success_df.sum()



    # tgi_llama1_request_count_df = tmp_llama_1['tgi_request_count'].diff()
    # tgi_llama1_request_count = tgi_llama1_request_count_df.sum()

    # tgi_llam2_request_count_df = tmp_llama_2['tgi_request_count'].diff()
    # tgi_llama2_request_count = tgi_llam2_request_count_df.sum()

    # tgi_llama3_request_count_df = tmp_llama_3['tgi_request_count'].diff()
    # tgi_llama3_request_count = tgi_llama3_request_count_df.sum()
    # print(f"Total  Requests at Router : {router_request_count} -->  Llama-1 : {tgi_llama1_request_count} || Llama-2: {tgi_llama2_request_count} || Llama-3: {tgi_llama3_request_count} ")




        

users = ['Alan', 'Noel', 'Hari']


for user in users: 
    get_stats(user)
    print('-------------------------------------------------------------------------------------------------------------------------------------\n')


print("------Completed---------")






