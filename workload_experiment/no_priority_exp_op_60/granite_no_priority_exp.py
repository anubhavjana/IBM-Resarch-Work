import os
import torch
import numpy as np
import pandas as pd
from datetime import timedelta

import argparse


from scipy import stats
import matplotlib.pyplot as plt
from torch.autograd import Variable

from matplotlib.pyplot import figure
pd.options.mode.chained_assignment = None


import warnings
warnings.filterwarnings('ignore')

import urllib3
urllib3.disable_warnings()

# # Command-line argument parsing
# parser = argparse.ArgumentParser(description='Process rr and op.')
# parser.add_argument('--rr', type=int, required=True, help='Request rate (rpm)')
# parser.add_argument('--op', type=int, required=True, help='Output token count')
# parser.add_argument('--model', type=str, required=True, help='Model Name (llama3 or granite)')
# args = parser.parse_args()

# rr = args.rr
# op = args.op
# model = args.model

# change request rates
rr = 55
op = 60
model = "granite"

# python3 granite_no_priority_exp.py --rr 50 --op 60 --model granite

tgis_llama_data_folder = f'{model}_a100_no_priority/rr_{rr}_op_{op}/tgis/'
print(f"Reading file : {tgis_llama_data_folder}\n")

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

tgi_metrics_with_user = [
   
    "tgi_request_duration_sum",
    "tgi_request_duration_count",
    "tgi_request_inference_duration_count",
    "tgi_request_inference_duration_sum",
    "tgi_request_generated_tokens_sum",
    "tgi_request_generated_tokens_count",
    "tgi_request_count"
   
]

tgi_metrics_without_user = [
    "tgi_queue_size",
    "tgi_batch_current_size",
    "tgi_request_queue_duration_count",
    "tgi_request_queue_duration_sum",
    "tgi_request_duration_sum",
    "tgi_request_duration_count",
    "tgi_request_failure"
   
]

tgi_filter_llama = {'pod':['granite-v3']}
tgi_df_user_llama = clean_data(get_tgi_llama_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama))
tgi_df_llama = clean_data(get_tgi_llama_df(tgi_metrics_without_user, tgi_columns, tgi_merge_columns, tgi_filter_llama))

users = ['Alan']
if(len(users)==1):
    print(f"Experiment carried out for single user {users[0]} with RPM = {rr} with max output token count {op} for model {model}\n")

tgi_queue_time_df = tgi_df_llama['tgi_request_queue_duration_sum'].diff() / tgi_df_llama['tgi_request_queue_duration_count'].diff()
# print("Mean TGIS Queue Time: ", tgi_queue_time_df.mean())
# print(f"95 percentile Mean Queue Time: {tgi_queue_time_df.quantile(0.95)}\n")

########################################## BATCH SIZE STATISTICS ##########################################
tgi_batch_size_values = tgi_df_llama['tgi_batch_current_size']
percentile_95_batch_size = tgi_batch_size_values.quantile(0.95)
average_tgi_batch_size = tgi_batch_size_values.mean()

print(f"Mean Batch Size: {average_tgi_batch_size}")
print(f"95 percentile Mean Batch Size: {percentile_95_batch_size}\n")

########################################## QUEUE SIZE STATISTICS ##########################################
tgi_queue_size_df = tgi_df_llama['tgi_queue_size'].diff()
percentile_95_queue_size = tgi_queue_size_df.quantile(0.95)
average_tgi_queue_size = tgi_queue_size_df.mean()

# print("Mean Queue Size: ", average_tgi_queue_size)
# print(f"95 percentile Queue Size: {percentile_95_queue_size}\n")

########################################## FAILURE COUNT STATISTICS ##########################################
tgi_failure_count = tgi_df_llama['tgi_request_failure'].diff().sum()
print("Failure Count: ", tgi_failure_count)


def get_stats(rr,op,user):

    # print(f"Stats for USER : {user} with RPM {rr} : Output Token count {op} \n")

    tmp_llama = tgi_df_user_llama[tgi_df_user_llama['user'] == user]
    tmp_llama.dropna()

    total_request = tmp_llama['tgi_request_count'].diff().sum()
    print(f"Total Request Count:  {total_request}")

    failure_rate = (tgi_failure_count / total_request) * 100
    print(f"Failure Rate:  {failure_rate} %\n")

    ######################################### E2E LATENCY , INFERENCE LATENCY, QUEUE TIME ##########################################
    tgis_latency_df = tmp_llama['tgi_request_duration_sum'].diff() / tmp_llama['tgi_request_duration_count'].diff()
    tgis_latency_df.dropna()
    tgis_inference_df = tmp_llama['tgi_request_inference_duration_sum'].diff() / tmp_llama['tgi_request_inference_duration_count'].diff()
    tgis_inference_df.dropna()
    # tgis_queue_time_df = tgis_latency_df - tgis_inference_df

    ######################################### OUTPUT TOKEN STATISTICS ################################################################

    mean_op_token_df = tmp_llama["tgi_request_generated_tokens_sum"].diff() / tmp_llama["tgi_request_generated_tokens_count"].diff()
    print("Mean Number of Output tokens: ",mean_op_token_df.mean())
    print(f"95 percentile Mean Output Tokens Count: {mean_op_token_df.quantile(0.95)}\n")
    # print(f"Max Output Tokens Count: {mean_op_token_df.max()}\n")

    ################################################## MEAN LATENCY PER TOKEN #######################################################

    mean_latency_per_tok_df = tgis_latency_df / mean_op_token_df # Request E2E Latency / Total number of output tokens
    print("Mean Latency Per Output tokens: ",mean_latency_per_tok_df.mean())
    print(f"95 percentile Mean Latency Per Token : {mean_latency_per_tok_df.quantile(0.95)}\n")

    # mean_latency_per_tok = tmp_llama["tgi_request_duration_sum"].diff() / tmp_llama["tgi_request_generated_tokens_sum"].diff()
    # print("Mean Latency Per Output tokens: ",mean_latency_per_tok.mean())
    # print(f"95 percentile Mean Latency Per Token : {mean_latency_per_tok.quantile(0.95)}")
    # print(f"Max Mean Latency Per Token: {mean_latency_per_tok.max()}\n")

    ##########################################################################################################################

    if(pd.isna(tgi_queue_time_df.mean())):
        print(f"User {user} did not send request to Llama")
    else:
        print(f"Mean Queue Time: {tgi_queue_time_df.mean()}")
        print(f'95 %ile QUEUE TIME is: {tgi_queue_time_df.quantile(0.95)}\n')


    if(pd.isna(tgis_latency_df.mean())):
        print(f"User {user} did not send request to Llama")
    else:
        print(f"Mean TGIS Latency: {tgis_latency_df.mean()}")
        # print(f'40 %ile latency is: {tgis_latency_df.quantile(0.40)}')
        # print(f'80 %ile latency is: {tgis_latency_df.quantile(0.80)}')
        print(f'95 %ile latency is: {tgis_latency_df.quantile(0.95)}\n')


    queue_data = {
        'Llama  Queue Time': tgi_queue_time_df.dropna(),
        
    }

    latency_data = {
        'Llama  Latency': tgis_latency_df.dropna(),
        
    }

    return mean_latency_per_tok_df.dropna(), tgi_queue_size_df.dropna(), tgis_latency_df.dropna(), tgi_queue_time_df

    # plot_histogram_with_cdf(queue_data, 'Queue Time (s)', 'Cumulative Probability', f'Queue Time CDF for User {user} (MAX OP TOKENS = 60)', f'{user}_tgis_llama_queue_time_cdf.png')
    # plot_histogram_with_cdf(latency_data, 'Latency (s)', 'Cumulative Probability', f'Latency CDF for User {user} (MAX OP TOKENS = 60)', f'{user}_tgis_llama_latency_cdf.png')




for user in users:
    
    # GET STATS 
    
    mean_latency_per_tok, tgi_queue_size_df, tgis_latency_df, tgis_queue_time_df  = get_stats(rr,op,user)

#     # PLOTTING CDF 

    # fig, axes = plt.subplots(3, 1, figsize=(10, 15))

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

    # axes[2].hist(tgis_latency_df, bins=1000, cumulative=True, density=True, histtype='step', label='TGIS Latency')
    # axes[2].set_xlabel('Latency (s)')
    # axes[2].set_ylabel('Cumulative Probability')
    # axes[2].set_title('CDF of TGIS E2E Latency')
    # axes[2].legend(loc='upper left')
    # axes[2].grid(True)

    # plt.tight_layout()
    # plt.savefig(f'Granite_CDF_rr_{rr}_op_{op}.png')
    # # plt.show()


# print("------Completed---------")

