import json
import pandas
from datetime import timedelta, datetime
from prometheus_api_client.utils import parse_datetime
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame

import urllib3
urllib3.disable_warnings()


def get_url_token(env="dev"):
    dev_promurl = 'https://prometheus-k8s-openshift-monitoring.apps.fmaas-devstage-backend.fmaas.res.ibm.com'
    dev_promurl1 = 'https://thanos-querier-openshift-monitoring.apps.fmaas-devstage-backend.fmaas.res.ibm.com'
    dev_token = "eyJhbGciOiJSUzI1NiIsImtpZCI6InNVQlVKeHVPaGJYYllQSVBETDZDLXE1TTNfcEdjdFlqaDR5elY3YmV5U2sifQ.eyJhdWQiOlsiaHR0cHM6Ly9rdWJlcm5ldGVzLmRlZmF1bHQuc3ZjIl0sImV4cCI6MTczMjEwMzQzOCwiaWF0IjoxNzAwNTY3NDM4LCJpc3MiOiJodHRwczovL2t1YmVybmV0ZXMuZGVmYXVsdC5zdmMiLCJrdWJlcm5ldGVzLmlvIjp7Im5hbWVzcGFjZSI6ImZtYWFzLW1vbml0b3JpbmciLCJzZXJ2aWNlYWNjb3VudCI6eyJuYW1lIjoiZm1hYXMtcHJvbWV0aGV1cyIsInVpZCI6IjQzN2JiNzAyLTZiOWMtNDFlZi1iNWEwLTZjNzM5NWYzZDk5ZiJ9fSwibmJmIjoxNzAwNTY3NDM4LCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6Zm1hYXMtbW9uaXRvcmluZzpmbWFhcy1wcm9tZXRoZXVzIn0.HNtCV3qCkvFgenM4wiB5M14ruDFun1z-ZPxrLph4NMwRNVBPaYpyq1A_mV0_yudfJJnDj-KH6nKx-kvH33zHIV_LiCO9PNTU3YdDBZTKH3IBX5t9a-k8zsJpAUmTAQsV_Wf5UVCebUhnEB3aweH4Z6qg7g5S-WEoRxYHAWYsYsQM-v4wvqjHwG3UigNbXZqgh-GA11eu3U1YEbSuWfrCXzQoVNML2-DMkl0jl5v8Yc5G0K7V6pCERmlubi_qzTcvZPjkY2kdQoe2tF6RYRR_lGy3BwCbknh4IyCjYLARFs0TWucT_KSEKtYkAjXHVFh8Zmln3gsSr4uTPsqWJhyFERKZNEcebhi8KQImYnlEOPp2Vngc-qelCkfNGjAeLTLCZwxY3HqGMMPqhRbWfELPXg86-TRFeQLkZoqeWCnr7qn_OrlZ34XHP66AC6HCAaPYdMZ7dwCEfIgVvlN-EbMM88HbjAnAIU0JPU2KkKJgkOOiG0AEAlZPVDRMyQmpxj5txfeY_LlF281xdNlbG1EuHBBYJennQ6E5mPt8crc_LFh2QMH_TTol1MiiUyfPT2LyHMm9hxFN9aqQxZszT8MQnAjp3f_7rLNvXZEubFIjva0hrjGV3j_WCFLQi7s-HvOasP0g4PY67FHbfxGH9JGFJS1ftWxRiPWL-RDVxj9w8wo"
    prod_promurl = 'https://prometheus-k8s-openshift-monitoring.apps.fmaas-backend.fmaas.res.ibm.com'
    prod_token = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImVySzNkN1hmdE9EMWNSZkdKQVVoUFMyRXM2cUo3UUFvbGFzSEpiX0NLZWsifQ.eyJhdWQiOlsiaHR0cHM6Ly9rdWJlcm5ldGVzLmRlZmF1bHQuc3ZjIl0sImV4cCI6MTczNjMzODg2OSwiaWF0IjoxNzA0ODAyODY5LCJpc3MiOiJodHRwczovL2t1YmVybmV0ZXMuZGVmYXVsdC5zdmMiLCJrdWJlcm5ldGVzLmlvIjp7Im5hbWVzcGFjZSI6ImZtYWFzLW1vbml0b3JpbmciLCJzZXJ2aWNlYWNjb3VudCI6eyJuYW1lIjoiZm1hYXMtcHJvbWV0aGV1cyIsInVpZCI6ImUxZWEzZmZhLWE2NGMtNGIwOC05ZTQzLTc5YTgxMzMyNzVhMyJ9fSwibmJmIjoxNzA0ODAyODY5LCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6Zm1hYXMtbW9uaXRvcmluZzpmbWFhcy1wcm9tZXRoZXVzIn0.NlqSlaQWR0YKejK23lSfGnNSOtjub-TiKRrh5cyRIESEnwWXC0QUj3KOgReUyI0VH5iCSdmg-nhmauN8x-KdNpKLnYnvcF0L7rV0vupON4lOb0_w9aMSxB9G0WTGLsg-pNwmcA4Kus3jAcnoKA_dvQmaNMXfdUc53dqMfNvPZ98HYLzHwnJr1J8UryGVUOsz4Ac-Zz8D_KR_z1fqThJngxggRTwm0Nsn9ZX5EPmmlcoI2DFdZ-mlPYIsRjwoO1J6sGiBBpK30BMglpR2eTCTi42xkJAeLPgVulUJgRivryNIGFW0pKcILI_H1sYBcFZZqdhuggf3byGF8B2m3DjYYG6RzDN5JJPmLR6HmuaRpMV8iKqT_rxWfSIB87UHr4U2NX9pjRKD8S0jIi6U8bKVzbHpspRCpgxBq-mhDQjFdoTxgxCwncFW04MNDJhywN0wQ9Cpe0Ee-LTqmVZb6-Q_P4QOoyFH-if9xPFD1ZSydL2hMuiVkq79IbeGOwjldwUtAbeA33MSJhuxK4yHUuVGuLKhvBTToFoo-24Yfvwc6Z_NTxGdgCNpOsJNz7FhP6nwj8xvqLQJ92ByzX_oXNXk20ZyayJpUNiEiLlt56gBK2vLauumAemqENOt785P2Y5yqBs3X-R-66jBXRbISJz7BAuTHhv-M6sVCKQ1pJPO4fY"
    if env == "dev":
        return dev_promurl, dev_token
    elif env == "dev1":
        return dev_promurl1, dev_token
    return prod_promurl, prod_token


def download_gpu_metrics(client, metric, start_time, end_time, chunk_size,rr):
    # gpu_label = {"Hostname":"fmaas-devstage-backen-9cjgj-worker-a100-il-rdma-3-jx6tm"}

    gpu_label = {'exported_pod':'llm-router-multi-instance-v3-0'}
    gpu_data = client.get_metric_range_data(metric, start_time = start_time, end_time = end_time, chunk_size = chunk_size, label_config = gpu_label)

    if (len(gpu_data) == 0):
        return
    
    df = MetricRangeDataFrame(gpu_data)
    # filename=f"metrics_data/granite/gpu/{rr}/{metric}_{str(datetime.now())}.csv"
    
    filename=f"testing_framework_exps/test_case_1/iteration_1_8.10/router/{metric}_{str(datetime.now())}.csv"

    df.to_csv(filename)


def download_tgis_metrics(client, metric, start_time, end_time, chunk_size):

    tgis_label_1 = {"pod":"llama-2.1"}
    tgis_label_2 = {"pod":"llama-2.2"}
    tgis_label_3 = {"pod":"llama-2.3"}
    # tgis_label_3 = {"pod":"llama3-4"}
   
    tgis_data = client.get_metric_range_data(metric, start_time = start_time, end_time = end_time, chunk_size = chunk_size, label_config = tgis_label_1)
    tgis_data_1 = client.get_metric_range_data(metric, start_time = start_time, end_time = end_time, chunk_size = chunk_size, label_config = tgis_label_2)
    tgis_data_2 = client.get_metric_range_data(metric, start_time = start_time, end_time = end_time, chunk_size = chunk_size, label_config = tgis_label_3)


    if(len(tgis_data)==0):
        return 

    if(len(tgis_data_1)==0):
        return 

    if(len(tgis_data_2)==0):
        return 
 
    df = MetricRangeDataFrame(tgis_data)
    df_1 = MetricRangeDataFrame(tgis_data_1)
    df_2 = MetricRangeDataFrame(tgis_data_2)
    
    filename=f"testing_framework_exps/exp_05_08/iteration_2_15.21_30mins/tgis/llama_2.1/{metric}_{str(datetime.now())}.csv"
    filename_1=f"testing_framework_exps/exp_05_08/iteration_2_15.21_30mins/tgis/llama_2.2/{metric}_{str(datetime.now())}.csv"
    filename_2=f"testing_framework_exps/exp_05_08/iteration_2_15.21_30mins/tgis/llama_2.3/{metric}_{str(datetime.now())}.csv"
    
    df.to_csv(filename)
    df_1.to_csv(filename_1)
    df_2.to_csv(filename_2)


def download_router_metrics(client, metric, start_time, end_time, chunk_size):

    router_label = {"pod":"llm-router-multi-instance-v3-1"}


    router_data = client.get_metric_range_data(metric, start_time = start_time, end_time = end_time, chunk_size = chunk_size, label_config = router_label)


    if(len(router_data)==0):
        return 
    
    df = MetricRangeDataFrame(router_data)

    filename = f"testing_framework_exps/exp_05_08/iteration_2_15.21_30mins/router/{metric}_{str(datetime.now())}.csv"

    df.to_csv(filename)




def get_timestamp(file_name, target_request_rate):
    with open(file_name, "r") as f:
        for line in f:
            entry = json.loads(line)
            
            if entry["request_rate"] == target_request_rate:
                start_time = datetime.strptime(entry["start_time"], "%Y-%m-%d %H:%M:%S")
                end_time = datetime.strptime(entry["end_time"], "%Y-%m-%d %H:%M:%S")
                return start_time, end_time
                
    return None, None


url1, token = get_url_token("dev1")
prom = PrometheusConnect(url=url1, headers={"Authorization" : "Bearer {}".format(token)}, disable_ssl=True)

# start_time, end_time = get_timestamp("granite_a100.jsonl",rr)

# +4 , -3 

start_time = "2024-08-05 10:49:25"
end_time =  "2024-08-05 11:12:26"

# end_time = datetime.now()

start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

# end_time = datetime.now()
# start_time = end_time - timedelta(hours=20)

chunk_size = timedelta(minutes=10)

print("Start Time:", start_time)
print("End Time:", end_time)



gpu_metrics_list = ["DCGM_FI_DEV_BOARD_LIMIT_VIOLATION",
    "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS",
    "DCGM_FI_DEV_DEC_UTIL",
    "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL",
    "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
    "DCGM_FI_DEV_ECC_SBE_AGG_TOTAL",
    "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL",
    "DCGM_FI_DEV_ENC_UTIL",
    "DCGM_FI_DEV_FB_FREE",
    "DCGM_FI_DEV_FB_USED",
    "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_DEV_GPU_UTIL",
    "DCGM_FI_DEV_LOW_UTIL_VIOLATION",
    "DCGM_FI_DEV_MEMORY_TEMP",
    "DCGM_FI_DEV_MEM_CLOCK",
    "DCGM_FI_DEV_MEM_COPY_UTIL",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L0",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL",
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL",
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL",
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL",
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL",
    "DCGM_FI_DEV_PCIE_REPLAY_COUNTER",
    "DCGM_FI_DEV_POWER_USAGE",
    "DCGM_FI_DEV_POWER_VIOLATION",
    "DCGM_FI_DEV_RELIABILITY_VIOLATION",
    "DCGM_FI_DEV_ROW_REMAP_FAILURE",
    "DCGM_FI_DEV_SM_CLOCK",
    "DCGM_FI_DEV_SYNC_BOOST_VIOLATION",
    "DCGM_FI_DEV_THERMAL_VIOLATION",
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
    "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS",
    "DCGM_FI_DEV_VGPU_LICENSE_STATUS",
    "DCGM_FI_DEV_XID_ERRORS",
    "DCGM_FI_PROF_DRAM_ACTIVE",
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE",
    "DCGM_FI_PROF_PCIE_RX_BYTES",
    "DCGM_FI_PROF_PCIE_TX_BYTES",
    "DCGM_FI_PROF_PIPE_FP16_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP64_ACTIVE",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "DCGM_FI_PROF_SM_ACTIVE",
    "DCGM_FI_PROF_SM_OCCUPANCY"]


tgis_metrics_list = [ 
    #"tgi_batch_concat",
    "tgi_batch_current_max_tokens",
    "tgi_batch_current_size",
    "tgi_batch_inference_batch_size_count",
    "tgi_batch_inference_batch_size_sum",
    "tgi_batch_inference_count",
    "tgi_batch_inference_duration_count",
    "tgi_batch_inference_duration_sum",
    "tgi_batch_inference_forward_duration_count",
    "tgi_batch_inference_forward_duration_sum",
    "tgi_batch_inference_success",
    "tgi_batch_inference_tokproc_duration_count",
    "tgi_batch_inference_tokproc_duration_sum",
    "tgi_batch_input_tokens",
    "tgi_batch_max_remaining_tokens",
    "tgi_batch_next_size_count",
    "tgi_batch_next_size_sum",
    "tgi_batch_next_tokens",
    "tgi_batch_next_tokens_count",
    "tgi_batch_next_tokens_sum",
    "tgi_prompt_load_duration_count",
    "tgi_prompt_load_duration_sum",
    "tgi_prompt_load_failure",
    "tgi_queue_size",
    "tgi_request_count",
    "tgi_request_duration_count",
    "tgi_request_duration_sum",
    "tgi_request_failure",
    "tgi_request_generated_tokens_count",
    "tgi_request_generated_tokens_sum",
    "tgi_request_inference_duration_count",
    "tgi_request_inference_duration_sum",
    "tgi_request_input_count",
    "tgi_request_input_length_count",
    "tgi_request_input_length_sum",
    "tgi_request_max_new_tokens_count",
    "tgi_request_max_new_tokens_sum",
    "tgi_request_mean_time_per_token_duration_count",
    "tgi_request_mean_time_per_token_duration_sum",
    "tgi_request_queue_duration_count",
    "tgi_request_queue_duration_sum",
    "tgi_request_raw_input_length",
    "tgi_request_raw_input_length_count",
    "tgi_request_raw_input_length_sum",
    "tgi_request_success",
    "tgi_request_total_tokens_count",
    "tgi_request_total_tokens_sum",
    "tgi_request_validation_duration_count",
    "tgi_request_validation_duration_sum",
    "tgi_tokenize_request_input_count"]
    # "vllm:num_requests_running",
    # "vllm:num_requests_waiting",
    # "vllm:num_requests_swapped",
    # "vllm:gpu_cache_usage_perc",
    # "vllm:cpu_cache_usage_perc",
    # "vllm:num_preemptions_total",
    # "vllm:prompt_tokens_total",
    # "vllm:generation_tokens_total",
    # "vllm:time_to_first_token_seconds",
    # "vllm:time_per_output_token_seconds",
    # "vllm:request_prompt_tokens",
    # "vllm:request_generation_tokens",
    # "vllm:request_success_total",
    # "vllm:avg_prompt_throughput_toks_per_s",
    # "vllm:avg_generation_throughput_toks_per_s",
    # "vllm:e2e_request_latency_seconds_sum",
    # "vllm:e2e_request_latency_seconds_count",
    # "vllm:time_to_first_token_seconds_sum",
    # "vllm:time_to_first_token_seconds_count",
    # "vllm:time_per_output_token_seconds_sum",
    # "vllm:time_per_output_token_seconds_count"]

router_metrics_list = ["router_queue_duration_sum",
    "router_queue_duration_count",
    "router_request_duration_sum",
    "router_request_duration_count",
    "router_request_count",
    "router_request_success",
    "router_request_failure"
    ]

# print("Downalod GPU metrics....")
# for gpu_metric in gpu_metrics_list:
#     download_gpu_metrics(prom, gpu_metric,start_time, end_time, chunk_size,rr)

print("Downalod TGIS metrics....")

for tgis_metric in tgis_metrics_list:
    download_tgis_metrics(prom, tgis_metric, start_time, end_time, chunk_size)

print("Download Router metrics....")

for router_metric in router_metrics_list:
    download_router_metrics(prom, router_metric, start_time, end_time, chunk_size)
