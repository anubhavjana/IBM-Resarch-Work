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


root_folder = "testing_framework_exps"
test_case_numbers = ["exp_05_08"]
iteration_1_pods = ["llama_2.1","llama_2.2"]
iteration_2_pods = ["llama_2.1","llama_2.2","llama_2.3"]


# iteration 1

tgis_llama2_1_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_1_10.14_30mins/tgis/{iteration_1_pods[0]}/'
tgis_llama2_2_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_1_10.14_30mins/tgis/{iteration_1_pods[1]}/'
router_data_folder_iteration_1 = 'testing_framework_exps/exp_05_08/iteration_1_10.14_30mins/router/'


# iteration 2

tgis_llama2_1_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_2_15.21_30mins/tgis/{iteration_2_pods[0]}/'
tgis_llama2_2_data_folder = f'{root_folder}/{test_case_numbers[0]}iteration_2_15.21_30mins/tgis/{iteration_2_pods[1]}/'
tgis_llama2_3_data_folder = f'{root_folder}/{test_case_numbers[0]}/iiteration_2_15.21_30mins/tgis/{iteration_2_pods[2]}/'




tgi_user_columns = ['timestamp', 'job', "namespace", "pod", "value", "user"]
tgi_user_merge_columns = ['timestamp', 'job', "namespace", "pod", "user"]

tgi_columns = ['timestamp', 'job', "namespace", "pod", "value"]
tgi_merge_columns = ['timestamp', 'job', "namespace", "pod"]

router_col = ['timestamp', 'job', "namespace", "pod", "user", "value"]
router_merge_columns = ['timestamp', 'job', "namespace", "pod", "user"]

def get_tgi_llama2_1_metrics(metric_name, columns, filter_pod, iteration):
    if(iteration == 1):
        tgis_llama2_1_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_1_10.14_30mins/tgis/{iteration_1_pods[0]}/'
    elif(iteration == 1.5):
        tgis_llama2_1_data_folder = f'{root_folder}/{test_case_numbers[0]}/3_mins_newload_old_router/tgis/llama_2.1/'
    elif(iteration==2):
        tgis_llama2_1_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_2_15.21_30mins/tgis/{iteration_2_pods[0]}/'


    files = os.listdir(tgis_llama2_1_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{tgis_llama2_1_data_folder}{i}')
            temp = temp[columns]
            dataframes.append(temp)
    result = pd.concat(dataframes)
    for key in filter_pod:
        result = result[result[key].isin(filter_pod[key])]
    result.drop_duplicates(inplace=True)
    return result.reset_index(drop=True)

def get_tgi_llama2_2_metrics(metric_name, columns, filter_pod, iteration):
    if(iteration == 1):
        tgis_llama2_2_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_1_10.14_30mins/tgis/{iteration_1_pods[1]}/'
    elif(iteration == 1.5):
        tgis_llama2_2_data_folder = f'{root_folder}/{test_case_numbers[0]}/3_mins_newload_old_router/tgis/llama_2.2/'
    elif(iteration==2):
        tgis_llama2_2_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_2_15.21_30mins/tgis/{iteration_2_pods[1]}/'
    files = os.listdir(tgis_llama2_2_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{tgis_llama2_2_data_folder}{i}')
            temp = temp[columns]
            dataframes.append(temp)
    result = pd.concat(dataframes)
    for key in filter_pod:
        result = result[result[key].isin(filter_pod[key])]
    result.drop_duplicates(inplace=True)
    return result.reset_index(drop=True)

def get_tgi_llama2_3_metrics(metric_name, columns, filter_pod, iteration):
    # only in iteration 2 
    if(iteration==1 or iteration==1.5):
        print(f'No data found for pod - {iteration_2_pods[2]}\n')
        return

    
    tgis_llama2_3_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_2_15.21_30mins/tgis/{iteration_2_pods[2]}/'

    files = os.listdir(tgis_llama2_3_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{tgis_llama2_3_data_folder}{i}')
            temp = temp[columns]
            dataframes.append(temp)
    result = pd.concat(dataframes)
    for key in filter_pod:
        result = result[result[key].isin(filter_pod[key])]
    result.drop_duplicates(inplace=True)
    return result.reset_index(drop=True)

# def get_router_metrics(metric_name, columns, filter_pod):
#     files = os.listdir(router_data_folder)
#     dataframes = []
#     for i in files:
#         if i[:len(metric_name)] == metric_name:
#             temp = pd.read_csv(f'{router_data_folder}{i}')
#             temp = temp[columns]
#             dataframes.append(temp)
#     result = pd.concat(dataframes)
#     for key in filter_pod:
#         result = result[result[key].isin(filter_pod[key])]
#     result.drop_duplicates(inplace=True)
#     return result.reset_index(drop=True)

def get_tgi_llama1_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter,iteration):
    tgi_df = get_tgi_llama2_1_metrics(metrics[0], tgi_columns, tgi_filter,iteration)
    tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_llama2_1_metrics(metric, tgi_columns, tgi_filter,iteration)
        tgi_df = pd.merge(tgi_df, temp_df, on = tgi_merge_columns)
        tgi_df.rename(columns={"value":metric}, inplace=True)
    return tgi_df

def get_tgi_llama2_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter,iteration):
    tgi_df = get_tgi_llama2_2_metrics(metrics[0], tgi_columns, tgi_filter, iteration)
    tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_llama2_2_metrics(metric, tgi_columns, tgi_filter,iteration)
        tgi_df = pd.merge(tgi_df, temp_df, on = tgi_merge_columns)
        tgi_df.rename(columns={"value":metric}, inplace=True)
    return tgi_df

def get_tgi_llama3_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter, iteration):
    tgi_df = get_tgi_llama2_3_metrics(metrics[0], tgi_columns, tgi_filter, iteration)
    tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_llama2_3_metrics(metric, tgi_columns, tgi_filter,iteration)
        tgi_df = pd.merge(tgi_df, temp_df, on = tgi_merge_columns)
        tgi_df.rename(columns={"value":metric}, inplace=True)
    return tgi_df

# def get_router_df(metrics, router_columns, router_merge_columns, router_filter):
#     router_df = get_router_metrics(metrics[0], router_columns, router_filter)
#     router_df.rename(columns={"value":metrics[0]}, inplace=True)
#     for metric in metrics[1:]:
#         temp_df = get_router_metrics(metric, router_columns, router_filter)
#         router_df = pd.merge(router_df, temp_df, on = router_merge_columns)
#         router_df.rename(columns={"value":metric}, inplace=True)
#     return router_df

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

tgi_filter_llama_2_1 = {'pod':['llama-2.1']}
tgi_filter_llama_2_2 = {'pod':['llama-2.2']}
tgi_filter_llama_2_3 = {'pod':['llama-2.3']}

# router_filter = {"pod" : ['llm-router-multi-instance-v3-0']}

iteration = 2

tgi_df_user_llama_2_1 = clean_data(get_tgi_llama1_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama_2_1, iteration))
tgi_df_user_llama_2_2 = clean_data(get_tgi_llama2_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama_2_2, iteration))
tgi_df_user_llama_2_3 = clean_data(get_tgi_llama3_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama_2_3, iteration))





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

    tmp_llama_1 = tgi_df_user_llama_2_1[tgi_df_user_llama_2_1['user'] == user]
    tmp_llama_2 = tgi_df_user_llama_2_2[tgi_df_user_llama_2_2['user'] == user]
    tmp_llama_3 = tgi_df_user_llama_2_3[tgi_df_user_llama_2_3['user'] == user]

    # Llama-1
    print("Stats for LLama-2.1...\n")
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
    print("\nStats for LLama-2.2...\n")
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

   
    # # Llama-3
    print("\nStats for LLama-2.3...\n")
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


    # router_df = clean_data(get_router_df(router_metrics, router_col, router_merge_columns, router_filter))
    # router_tmp = router_df[router_df['user']==user]

    # router_request_count_df  = router_tmp['router_request_count'].diff()
    # router_request_count = router_request_count_df.sum()

    # router_request_failure_df = router_tmp['router_request_failure'].diff()
    # router_request_failure_count = router_request_failure_df.sum()

    # router_request_success_df = router_tmp['router_request_success'].diff()
    # router_request_success_count = router_request_success_df.sum()

    # tgi_llama1_request_count_df = tmp_llama_1['tgi_request_count'].diff()
    # tgi_llama1_request_count = tgi_llama1_request_count_df.sum()

    # tgi_llam2_request_count_df = tmp_llama_2['tgi_request_count'].diff()
    # tgi_llama2_request_count = tgi_llam2_request_count_df.sum()

    # tgi_llama3_request_count_df = tmp_llama_3['tgi_request_count'].diff()
    # tgi_llama3_request_count = tgi_llama3_request_count_df.sum()

    queue_data = {
        'Llama-2.1 Queue Time': tgis_queue_time_df_llama1.dropna(),
        'Llama-2.2 Queue Time': tgis_queue_time_df_llama2.dropna(),
        'Llama-2.3 Queue Time': tgis_queue_time_df_llama3.dropna()
    }

    latency_data = {
        'Llama-2.1 Latency': tgis_latency_df_llama1.dropna(),
        'Llama-2.2 Latency': tgis_latency_df_llama2.dropna(),
        'Llama-2.3 Latency': tgis_latency_df_llama3.dropna()
    }

    # Plot histograms with CDFs
    plot_histogram_with_cdf(queue_data, 'Queue Time (s)', 'Cumulative Probability', f'Queue Time CDF for User {user}', f'iteration_{iteration}_queue_time_cdf_user_{user}.png')
    plot_histogram_with_cdf(latency_data, 'Latency (s)', 'Cumulative Probability', f'Latency CDF for User {user}', f'iteration_{iteration}_latency_cdf_user_{user}.png')

    print(f"Generated CDF plots for User {user}")



users = ['Alan', 'Noel']

for user in users:
    get_stats(user)
    print('-------------------------------------------------------------------------------------------------------------------------------------\n')

print("------Completed---------")

# import os
# import torch
# import numpy as np
# import pandas as pd
# from scipy import stats
# import matplotlib.pyplot as plt
# from torch.autograd import Variable
# from matplotlib.pyplot import figure
# pd.options.mode.chained_assignment = None

# import warnings
# warnings.filterwarnings('ignore')

# import urllib3
# urllib3.disable_warnings()


# root_folder = "testing_framework_exps"
# test_case_numbers = ["exp_05_08"]
# iteration_1_pods = ["llama_2.1","llama_2.2"]
# iteration_2_pods = ["llama_2.1","llama_2.2","llama_2.3"]


# # iteration 1

# tgis_llama2_1_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_1_10.14_30mins/tgis/{iteration_1_pods[0]}/'
# tgis_llama2_2_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_1_10.14_30mins/tgis/{iteration_1_pods[1]}/'
# router_data_folder_iteration_1 = 'testing_framework_exps/exp_05_08/iteration_1_10.14_30mins/router/'


# # iteration 2

# tgis_llama2_1_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_2_15.21_30mins/tgis/{iteration_2_pods[0]}/'
# tgis_llama2_2_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_2_15.21_30mins/tgis/{iteration_2_pods[1]}/'
# tgis_llama2_3_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_2_15.21_30mins/tgis/{iteration_2_pods[2]}/'

# tgi_user_columns = ['timestamp', 'job', "namespace", "pod", "value", "user"]
# tgi_user_merge_columns = ['timestamp', 'job', "namespace", "pod", "user"]

# tgi_columns = ['timestamp', 'job', "namespace", "pod", "value"]
# tgi_merge_columns = ['timestamp', 'job', "namespace", "pod"]

# router_col = ['timestamp', 'job', "namespace", "pod", "user", "value"]
# router_merge_columns = ['timestamp', 'job', "namespace", "pod", "user"]

# def get_tgi_llama2_1_metrics(metric_name, columns, filter_pod, iteration):
#     if iteration == 1:
#         tgis_llama2_1_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_1_10.14_30mins/tgis/{iteration_1_pods[0]}/'
#     elif iteration == 1.5:
#         tgis_llama2_1_data_folder = f'{root_folder}/{test_case_numbers[0]}/3_mins_newload_old_router/tgis/llama_2.1/'
#     elif iteration == 2:
#         tgis_llama2_1_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_2_15.21_30mins/tgis/{iteration_2_pods[0]}/'

#     files = os.listdir(tgis_llama2_1_data_folder)
#     dataframes = []
#     for i in files:
#         if i[:len(metric_name)] == metric_name:
#             temp = pd.read_csv(f'{tgis_llama2_1_data_folder}{i}')
#             temp = temp[columns]
#             dataframes.append(temp)
#     result = pd.concat(dataframes)
#     for key in filter_pod:
#         result = result[result[key].isin(filter_pod[key])]
#     result.drop_duplicates(inplace=True)
#     return result.reset_index(drop=True)

# def get_tgi_llama2_2_metrics(metric_name, columns, filter_pod, iteration):
#     if iteration == 1:
#         tgis_llama2_2_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_1_10.14_30mins/tgis/{iteration_1_pods[1]}/'
#     elif iteration == 1.5:
#         tgis_llama2_2_data_folder = f'{root_folder}/{test_case_numbers[0]}/3_mins_newload_old_router/tgis/llama_2.2/'
#     elif iteration == 2:
#         tgis_llama2_2_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_2_15.21_30mins/tgis/{iteration_2_pods[1]}/'
#     files = os.listdir(tgis_llama2_2_data_folder)
#     dataframes = []
#     for i in files:
#         if i[:len(metric_name)] == metric_name:
#             temp = pd.read_csv(f'{tgis_llama2_2_data_folder}{i}')
#             temp = temp[columns]
#             dataframes.append(temp)
#     result = pd.concat(dataframes)
#     for key in filter_pod:
#         result = result[result[key].isin(filter_pod[key])]
#     result.drop_duplicates(inplace=True)
#     return result.reset_index(drop=True)

# def get_tgi_llama2_3_metrics(metric_name, columns, filter_pod, iteration):
#     # only in iteration 2 
#     if iteration == 1 or iteration == 1.5:
#         print(f'No data found for pod - {iteration_2_pods[2]}\n')
#         return

#     tgis_llama2_3_data_folder = f'{root_folder}/{test_case_numbers[0]}/iteration_2_15.21_30mins/tgis/{iteration_2_pods[2]}/'

#     files = os.listdir(tgis_llama2_3_data_folder)
#     dataframes = []
#     for i in files:
#         if i[:len(metric_name)] == metric_name:
#             temp = pd.read_csv(f'{tgis_llama2_3_data_folder}{i}')
#             temp = temp[columns]
#             dataframes.append(temp)
#     result = pd.concat(dataframes)
#     for key in filter_pod:
#         result = result[result[key].isin(filter_pod[key])]
#     result.drop_duplicates(inplace=True)
#     return result.reset_index(drop=True)

# def get_tgi_llama1_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter, iteration):
#     tgi_df = get_tgi_llama2_1_metrics(metrics[0], tgi_columns, tgi_filter, iteration)
#     tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
#     for metric in metrics[1:]:
#         temp_df = get_tgi_llama2_1_metrics(metric, tgi_columns, tgi_filter, iteration)
#         tgi_df = pd.merge(tgi_df, temp_df, on=tgi_merge_columns)
#         tgi_df.rename(columns={"value":metric}, inplace=True)
#     return tgi_df

# def get_tgi_llama2_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter, iteration):
#     tgi_df = get_tgi_llama2_2_metrics(metrics[0], tgi_columns, tgi_filter, iteration)
#     tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
#     for metric in metrics[1:]:
#         temp_df = get_tgi_llama2_2_metrics(metric, tgi_columns, tgi_filter, iteration)
#         tgi_df = pd.merge(tgi_df, temp_df, on=tgi_merge_columns)
#         tgi_df.rename(columns={"value":metric}, inplace=True)
#     return tgi_df

# # def get_tgi_llama3_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter, iteration):
# #     tgi_df = get_tgi_llama2_3_metrics(metrics[0], tgi_columns, tgi_filter, iteration)
# #     tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
# #     for metric in metrics[1:]:
# #         temp_df = get_tgi_llama2_3_metrics(metric, tgi_columns, tgi_filter, iteration)
# #         tgi_df = pd.merge(tgi_df, temp_df, on=tgi_merge_columns)
# #         tgi_df.rename(columns={"value":metric}, inplace=True)
# #     return tgi_df

# def clean_data(df):
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df = df.sort_values('timestamp')
#     return df.reset_index(drop=True)

# tgi_metrics_with_user = [
#     "tgi_request_generated_tokens_sum",
#     "tgi_request_generated_tokens_count",
#     "tgi_request_duration_sum",
#     "tgi_request_duration_count",
#     "tgi_request_count",
#     "tgi_request_inference_duration_count",
#     "tgi_request_inference_duration_sum",
#     "tgi_request_success"
# ]

# tgi_metrics_without_user = [
#     "tgi_queue_size",
#     "tgi_request_queue_duration_count",
#     "tgi_request_queue_duration_sum",
#     "tgi_request_duration_sum",
#     "tgi_request_duration_count"
# ]

# router_metrics = [
#     "router_request_duration_sum",
#     "router_request_duration_count",
#     "router_queue_duration_sum",
#     "router_queue_duration_count",
#     "router_request_count",
#     "router_request_failure",
#     "router_request_success"
# ]

# tgi_filter_llama_2_1 = {'pod':['llama_2.1']}
# tgi_filter_llama_2_2 = {'pod':['llama_2.2']}
# tgi_filter_llama_2_3 = {'pod':['llama_2.3']}


# router_filter = {'pod':["router-7c96bb7c7c-rx9x9"]}

# def get_user_stats(user, iteration):
#     # TGI pod - llama_2.1
#     tgi_llama_2_1_user_df = get_tgi_llama1_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama_2_1, iteration)
#     tgi_llama_2_1_user_df = clean_data(tgi_llama_2_1_user_df)
#     tgi_llama_2_1_user_df = tgi_llama_2_1_user_df[tgi_llama_2_1_user_df["user"] == user]

#     # TGI pod - llama_2.2
#     tgi_llama_2_2_user_df = get_tgi_llama2_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama_2_2, iteration)
#     tgi_llama_2_2_user_df = clean_data(tgi_llama_2_2_user_df)
#     tgi_llama_2_2_user_df = tgi_llama_2_2_user_df[tgi_llama_2_2_user_df["user"] == user]

#     # # TGI pod - llama_2.3
#     # tgi_llama_2_3_user_df = get_tgi_llama3_df(tgi_metrics_with_user, tgi_user_columns, tgi_user_merge_columns, tgi_filter_llama_2_3, iteration)
#     # tgi_llama_2_3_user_df = clean_data(tgi_llama_2_3_user_df)
#     # tgi_llama_2_3_user_df = tgi_llama_2_3_user_df[tgi_llama_2_3_user_df["user"] == user]

#     # # router
#     # router_df = get_router_metrics(router_metrics, router_col, router_filter, iteration)
#     # router_df = clean_data(router_df)
#     # router_df = router_df[router_df["user"] == user]

#     tgi_llama_2_1_user_df['e2e_latency'] = tgi_llama_2_1_user_df['tgi_request_duration_sum'].diff() / tgi_llama_2_1_user_df['tgi_request_duration_count'].diff()
#     tgi_llama_2_1_user_df['inference_time'] = tgi_llama_2_1_user_df['tgi_request_inference_duration_sum'].diff() / tgi_llama_2_1_user_df['tgi_request_inference_duration_count'].diff()
#     tgi_llama_2_1_user_df['queue_time'] =tgi_llama_2_1_user_df['e2e_latency'] - tgi_llama_2_1_user_df['inference_time']


#     tgi_llama_2_2_user_df['e2e_latency'] = tgi_llama_2_2_user_df['tgi_request_duration_sum'].diff() / tgi_llama_2_2_user_df['tgi_request_duration_count'].diff()
#     tgi_llama_2_2_user_df['inference_time'] = tgi_llama_2_2_user_df['tgi_request_inference_duration_sum'].diff() / tgi_llama_2_2_user_df['tgi_request_inference_duration_count'].diff()
#     tgi_llama_2_2_user_df['queue_time'] = tgi_llama_2_2_user_df['e2e_latency'] - tgi_llama_2_2_user_df['inference_time']

# #     tgis_inference_df_llama2 = tmp_llama_2['tgi_request_inference_duration_sum'].diff() / tmp_llama_2['tgi_request_inference_duration_count'].diff()
# #     tgis_queue_time_df_llama2 = tgis_latency_df_llama2 - tgis_inference_df_llama2


#     return tgi_llama_2_1_user_df, tgi_llama_2_2_user_df

# def get_stats(user_list, iteration):
#     all_user_data = []
#     for user in user_list:
#         user_data = get_user_stats(user, iteration)
#         all_user_data.append(user_data)
#     return all_user_data


# def plot_histogram_with_cdf(stats, user_list):
#     fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # 3 Llama instances, 2 metrics (latency and queue time) each

#     for i, user_data in enumerate(stats):
#         for j, data in enumerate(user_data):  # Exclude router data
#             instance_name = iteration_2_pods[j]
            
#             # Latency CDF
#             axs[j, 0].hist(data['e2e_latency'], bins=100, density=True, alpha=0.6, label=f'User {user_list[i]}')
#             axs[j, 0].set_title(f'{instance_name} Latency CDF')
#             axs[j, 0].set_xlabel('Latency (ms)')
#             axs[j, 0].set_ylabel('CDF')

#             # Queue Time CDF
#             axs[j, 1].hist(data['e2e_latency'], bins=100, density=True, alpha=0.6, label=f'User {user_list[i]}')
#             axs[j, 1].set_title(f'{instance_name} Queue Time CDF')
#             axs[j, 1].set_xlabel('Queue Time (ms)')
#             axs[j, 1].set_ylabel('CDF')
            
#     # for ax in axs.flat:
#     #     ax.legend()

#     plt.tight_layout()
#     plt.savefig(f'Iteration_{iteration}.png')


# user_list = ['user_0', 'user_1', 'user_2']
# iteration = 1 # change as needed

# all_user_data = get_stats(user_list, iteration)
# plot_histogram_with_cdf(all_user_data, user_list)

