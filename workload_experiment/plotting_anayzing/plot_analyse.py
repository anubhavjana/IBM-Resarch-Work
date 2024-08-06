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




rr = 5

model = 'vllm-granite'
# print ("request rate :",rr)

tgis_data_folder = 'metrics_data/granite/tgis/'+str(rr)+'/'
gpu_data_folder = 'metrics_data/granite/gpu/'+str(rr)+'/'
router_data_folder = 'metrics_data/granite/gpu/'+str(rr)+'/'






dcgm_columns = ['timestamp', 'Hostname', 'UUID', 'exported_container','exported_namespace', 'exported_pod', 'value']
gpu_merge_columns = ['timestamp', 'Hostname', 'UUID', 'exported_container','exported_namespace', 'exported_pod']
tgi_columns = ['timestamp', 'job', "namespace", "pod", "value"]
tgi_merge_columns = ['timestamp', 'job', "namespace", "pod"]


router_columns = ['timestamp', 'job', "namespace", "pod", "user", "value"]
router_merge_columns = ['timestamp', 'job', "namespace", "pod", "user"]


import urllib3
urllib3.disable_warnings()

def get_gpu_metrics(metric_name, columns, filter_gpu):
    files = os.listdir(gpu_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{gpu_data_folder}{i}')
            temp = temp[columns]
            dataframes.append(temp)
    result = pd.concat(dataframes)
    for key in filter_gpu:
        result = result[result[key].isin(filter_gpu[key])]
    result.drop_duplicates(inplace=True)
    return result.reset_index(drop=True)

def get_tgi_metrics(metric_name, columns, filter_pod):
    files = os.listdir(tgis_data_folder)
    dataframes = []
    for i in files:
        if i[:len(metric_name)] == metric_name:
            temp = pd.read_csv(f'{tgis_data_folder}{i}')
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

def get_gpu_df(metrics, gpu_columns, merge_columns, gpu_filter):
    gpu_df = get_gpu_metrics(metrics[0], dcgm_columns, gpu_filter)
    gpu_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_gpu_metrics(metric, gpu_columns, gpu_filter)
        gpu_df = pd.merge(gpu_df, temp_df, on = merge_columns)
        gpu_df.rename(columns={"value":metric}, inplace=True)
    return gpu_df

def get_tgi_df(metrics, tgi_columns, tgi_merge_columns, tgi_filter,start_offset=8, end_offset=3):
    tgi_df = get_tgi_metrics(metrics[0], tgi_columns, tgi_filter)
    tgi_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_tgi_metrics(metric, tgi_columns, tgi_filter)
        tgi_df = pd.merge(tgi_df, temp_df, on = tgi_merge_columns)
        tgi_df.rename(columns={"value":metric}, inplace=True)

    tgi_df = filter_timestamps(tgi_df, start_offset, end_offset) # add this call to discard first 8 mins and last 3 minutes data

    return tgi_df


def get_router_df(metrics, router_columns, router_merge_columns, router_filter,start_offset=8, end_offset=3):
    router_df = get_router_metrics(metrics[0], router_columns, router_filter)
    router_df.rename(columns={"value":metrics[0]}, inplace=True)
    for metric in metrics[1:]:
        temp_df = get_router_metrics(metric, router_columns, router_filter)
        router_df = pd.merge(router_df, temp_df, on = router_merge_columns)
        router_df.rename(columns={"value":metric}, inplace=True)


    router_df = filter_timestamps(router_df, start_offset, end_offset) # add this call to discard first 8 mins and last 3 minutes data

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


def clean_data(df, counter_list):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df.reset_index(inplace=True, drop=True)
    return df


gpu_metrics = [
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
    "DCGM_FI_PROF_SM_ACTIVE",
    "DCGM_FI_PROF_DRAM_ACTIVE",
    "DCGM_FI_PROF_SM_OCCUPANCY",
    "DCGM_FI_DEV_POWER_USAGE",
    "DCGM_FI_DEV_SM_CLOCK",
    "DCGM_FI_PROF_PIPE_FP16_ACTIVE"
]

tgi_metrics = [
    "tgi_request_generated_tokens_sum",
    "tgi_request_inference_duration_sum",
    "tgi_batch_current_size",
    "tgi_request_duration_sum",
    "tgi_request_total_tokens_sum",
    "tgi_request_queue_duration_sum",
    "tgi_request_queue_duration_count",
    "tgi_queue_size",
    "tgi_request_mean_time_per_token_duration_sum",
    "tgi_request_duration_count",
    "tgi_request_inference_duration_count",
    "tgi_request_generated_tokens_count",
    "tgi_request_input_length_sum",
    "tgi_request_input_length_count",
    # "tgi_request_failure",
    "tgi_request_count"

]

gpu_counters = [
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"
]

tgi_counters = [
    "tgi_request_generated_tokens_sum",
    "tgi_request_inference_duration_sum",
    "tgi_request_duration_sum",
    "tgi_request_total_tokens_sum",
    "tgi_request_queue_duration_sum",
    "tgi_queue_size",
    "tgi_request_mean_time_per_token_duration_sum",
    "tgi_request_duration_count",
    "tgi_request_inference_duration_count",
    "tgi_request_generated_tokens_count",
    "tgi_request_input_length_sum",
    "tgi_request_input_length_count",
    # "tgi_request_failure",
    "tgi_request_queue_duration_count",
    "tgi_request_count"
]

router_metrics = ["router_queue_duration_sum",
    "router_queue_duration_count",
    "router_request_duration_sum",
    "router_request_duration_count",
    # "router_request_count",
    # "router_request_success"
   
    ]

router_counters = ["router_queue_duration_sum",
    "router_queue_duration_count",
    "router_request_duration_sum",
    "router_request_duration_count",
    # "router_request_count",
    # "router_request_success"
    ]

# gpu_filter = {'exported_pod':['llama3-mig3g']}
# tgi_filter = {'pod':['llama3-mig3g']}
# router_filter = {"pod":["llm-router-v9"],'user':['Alan']}

gpu_filter = {'exported_pod':['llama3-new']}

tgi_filter = {'pod':['llama3-new']}



router_filter = {"pod":["llm-router-v7"],'user':['Alan']}

# router_filter = {"pod":["llm-router-v7"]}


# gpu_df = get_gpu_df(gpu_metrics, dcgm_columns, gpu_merge_columns, gpu_filter)
tgi_df = get_tgi_df(tgi_metrics, tgi_columns, tgi_merge_columns, tgi_filter)


# router_df = get_router_df(router_metrics, router_columns, router_merge_columns, router_filter)

# gpu_df = clean_data(gpu_df, gpu_counters)
tgi_df = clean_data(tgi_df, tgi_counters)

# router_df = clean_data(router_df, router_counters)

# print(router_df)




tgi_data = tgi_df[['timestamp', 'tgi_request_queue_duration_sum', 'tgi_request_count', 'tgi_request_queue_duration_count','tgi_batch_current_size', 'tgi_queue_size', 'tgi_request_generated_tokens_sum', 'tgi_request_generated_tokens_count', 'tgi_request_mean_time_per_token_duration_sum', 'tgi_request_duration_sum', 'tgi_request_duration_count', 'tgi_request_inference_duration_sum', 'tgi_request_inference_duration_count','tgi_request_total_tokens_sum','tgi_request_input_length_sum','tgi_request_input_length_count']]

columns = ['tgi_request_inference_duration_count', 'tgi_request_count','tgi_request_inference_duration_sum', 'tgi_request_duration_count','tgi_request_generated_tokens_sum', 'tgi_request_mean_time_per_token_duration_sum','tgi_request_duration_sum','tgi_request_input_length_sum','tgi_request_input_length_count','tgi_request_generated_tokens_count',]

temp = tgi_data[columns]

for col in columns:
    temp[col] = pd.to_numeric(temp[col])

# router_data = router_df[['timestamp','router_queue_duration_sum','router_queue_duration_count','router_request_duration_sum','router_request_duration_count']]

# router_queue_time_values = router_data['router_queue_duration_sum'].diff()/router_data['router_queue_duration_count'].diff()

# mean_router_queue_time = router_queue_time_values.mean()
# print("Mean Router Queue Time:", router_queue_time_values.mean())
# print("95 %ile Router Queue Time:",router_queue_time_values.quantile(0.95))

# router_request_count = router_data['router_request_count'].diff().mean()
# print("Mean Router Request Count:", router_request_count)




tgi_request_count = tgi_data['tgi_request_count'].diff()
average_tgi_request_count = tgi_request_count.mean()
print("Mean TGIS Request Count:", average_tgi_request_count)

# ############################################################################################################

tgi_queue_size_values = tgi_data['tgi_queue_size']
percentile_95_queue_size = tgi_queue_size_values.quantile(0.95)
average_tgi_queue_size = tgi_queue_size_values.mean()
print("Mean TGIS Queue Size: ", average_tgi_queue_size)
print("95 percentile TGIS Mean Queue Size: ",percentile_95_queue_size)


# ############################################################################################################

# tgi_batch_size_values = tgi_data['tgi_batch_current_size']
# percentile_95_batch_size = tgi_batch_size_values.quantile(0.95)
# print("--------------95 percentile Mean Batch Size--------------",percentile_95_batch_size)
# print("----------------------Max Batch size-------",tgi_batch_size_values.max())
# average_tgi_batch_size = tgi_batch_size_values.mean()
# print("----------------------Mean Batch Size:----------------------", average_tgi_batch_size)



# temp['count'] = temp['tgi_request_inference_duration_count'].diff()
# temp['duration'] = temp['tgi_request_inference_duration_sum'].diff()
# temp['tokens'] = temp['tgi_request_generated_tokens_sum'].diff()
# temp['latency'] = temp['tgi_request_duration_sum'].diff()

# temp.dropna(inplace=True)




# temp['mean_token_inference_time'] = temp['tgi_request_inference_duration_sum'].diff() / temp['tokens'] # without queue 
# percentile_95_mean_token_inference = temp['mean_token_inference_time'].quantile(0.95)
# print("--------------95 percentile Mean Token Inference Time--------------",percentile_95_mean_token_inference)
# mean_token_inference_time = temp['mean_token_inference_time'].mean()
# print("Mean token inference latency--",mean_token_inference_time)

# temp.dropna(inplace=True)


# temp['request_count'] =temp['tgi_request_duration_count'].diff()
# temp['mean_request_latency'] = temp['latency'] / temp['request_count'] # with queue (dividing by the number of total requests)
# mean_request_latency = temp['mean_request_latency'].mean()
# print("--------------Mean total latency --------------",mean_request_latency)


temp['queue_duration_count'] = tgi_data['tgi_request_queue_duration_count'].diff()
temp['queue_duration_sum'] = tgi_data['tgi_request_queue_duration_sum'].diff()

temp['mean_queue_time'] = temp['queue_duration_sum'] / temp['queue_duration_count']
mean_queue_time = temp['mean_queue_time'].mean()
percentile_95_queue_waiting_time = temp['mean_queue_time'].quantile(0.95)
print("Mean TGIS Queue Waiting Time: ",mean_queue_time)
print("95 percentile Mean TGIS Queue Waiting Time: ",percentile_95_queue_waiting_time)



# temp['queue_time'] =  temp['mean_token_latency'] - temp['mean_token_inference_time'] 
# mean_queue_time = temp['queue_time'].mean()

# # ##################################### AVERAGE INPUT TOKENS COUNT #####################################
# temp['total_input_request_length'] = temp['tgi_request_input_length_sum'].diff()
# temp['total_input_request_count'] = temp['tgi_request_input_length_count'].diff()
# temp["mean_input_tokens_count"] = temp['total_input_request_length'] / temp['total_input_request_count']
# print("----------Mean Number of input tokens----------",temp["mean_input_tokens_count"].mean())
# ###############################################################################################################

# temp['total_output_request_length'] = temp["tgi_request_generated_tokens_sum"].diff()
# temp['total_output_request_count'] = temp["tgi_request_generated_tokens_count"].diff()
# temp["mean_output_tokens_count"] = temp['total_output_request_length'] / temp['total_output_request_count']
# print("----------Mean Number of Output tokens----------",temp["mean_output_tokens_count"].mean())
# # ###############################################################################################################


# temp.dropna(inplace=True)




# temp['mean_token_inference_time'] = temp['mean_token_inference_time']/temp['count']

# data = pd.DataFrame({
#     'tgi_batch_current_size': tgi_batch_size_values,
#     'mean_token_time': temp['mean_token_inference_time']
# })

# # Drop any rows with NaN values
# data = data.dropna()

# # Calculate Pearson's correlation coefficient
# correlation, p_value = pearsonr(data['tgi_batch_current_size'], data['mean_token_time'])
# print(f"Pearson's correlation coefficient: {correlation:.2f}, p-value: {p_value:.2e}")

# # Fit a linear regression model
# X = data['tgi_batch_current_size'].values.reshape(-1, 1)
# y = data['mean_token_time'].values
# reg = LinearRegression().fit(X, y)
# y_pred = reg.predict(X)

# # Print the regression coefficients
# print(f"Intercept: {reg.intercept_:.2f}")
# print(f"Coefficient: {reg.coef_[0]:.2f}")

# # Plot the data and the regression line
# plt.figure(figsize=(10, 6))
# plt.scatter(data['tgi_batch_current_size'], data['mean_token_time'], color='blue', label='Data points')
# plt.plot(data['tgi_batch_current_size'], y_pred, color='red', label='Regression line')
# plt.xlabel('Batch Size')
# plt.ylabel('Mean Token Time')
# plt.title('Relationship between Batch Size and Mean Token Time')
# plt.legend()
# plt.show()


########################################################################################################################


# plt.plot(temp['mean_token_latency'] , '*')
# plt.xlabel("Samples")
# plt.ylabel("mean time per token")
# plt.title("Request rate : 70 RPM (Max Waiting Tokens = 20)")
# plt.show()

########################################################################################################################


# plt.hist(temp['mean_token_latency'] , cumulative=True, bins=1000)

# plt.xlabel("Mean Total Latency Time Per Token (with queue)")
# plt.ylabel("Samples")
# plt.title("Request rate : 100 RPM (Previous Prompt)")
# plt.show()

########################################################################################################################


########################################################################################################################


# plt.hist(temp['mean_token_inference_time'] , cumulative=True, bins=1000)
# plt.xlabel("Mean Inference Latency Time Per Token")
# plt.ylabel("Samples")
# plt.title("Request rate : 70 RPM (Bloom mig3g - New Prompt)")
# plt.show()


# plt.hist(tgi_data['tgi_queue_size'], cumulative=True, bins=1000)
# plt.xlabel("Mean Queue Size")
# plt.ylabel("Samples")
# plt.title("Request rate : 70 RPM (Bloom mig3g - New Prompt)")
# plt.show()

# plt.hist(tgi_data['tgi_batch_current_size'], cumulative=True, bins=1000)
# plt.xlabel("Mean Batch Size")
# plt.ylabel("Samples")
# plt.title("Request rate : 15 RPM (New Prompt)")
# plt.show()

########################################################################################################################


# plt.hist(temp['mean_queue_time'], cumulative=True, bins=1000)
# plt.xlabel("TGIS Mean Queue Time")
# plt.ylabel("Samples")
# plt.title("Request rate : 5 RPM Per User (Without Router)")
# plt.show()


# plt.hist(tgi_data['tgi_queue_size'], cumulative=True, bins=1000)
# plt.xlabel("Queue Size")
# plt.ylabel("Samples")
# plt.title("Request rate : 70 RPM (Previous Prompt)")
# plt.show()



# plt.hist(temp["mean_output_tokens_count"], cumulative=True, bins=1000)
# plt.xlabel("Mean Output Token Count")
# plt.ylabel("Samples")
# plt.title("Request rate : 70 RPM (Mig3g.40gb Old Prompt)")
# plt.show()


# Normal Distribution: For the mean  is also the median.
# Right-Skewed Distribution: In a distribution with a long tail to the right, the mean will be greater than the median.
# Left-Skewed Distribution: In a distribution with a long tail to the left, the mean will be less than the median.

# The median (50 %ile) is the value that separates the higher half from the lower half of the probability distribution. 
# It is the point at which the cumulative distribution function (CDF) equals 0.5

# The mean (or expected value) of a random variable is the average of all possible values it can take,
# weighted by their probabilities.

########################################################################################################################



# plt.plot(merged_dataframe['tgi_batch_current_size'])
