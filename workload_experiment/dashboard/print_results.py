def get_vllm(results):
    results_text = ""
    for result in results:
        results_text += f"Request Rate: {result['rate']} RPM\n"
        results_text += f"Mean TGIS Queue Size: {result['average_tgi_queue_size']}\n"
        results_text += f"95 percentile TGIS Mean Queue Size: {result['percentile_95_queue_size']}\n"
        results_text += f"95 percentile Mean Batch Size (vLLM) : {result['percentile_95_batch_size_vllm']}\n"
        results_text += f"Mean Batch Size : {result['average_tgi_batch_size']}\n"
        results_text += f"Mean TGIS Queue Waiting Time: {result['mean_queue_time']}\n"
        results_text += f"95 percentile Mean TGIS Queue Waiting Time: {result['percentile_95_queue_waiting_time']}\n"
        results_text += f"Mean E2E Latency: {result['mean_e2e_latency']}\n"
        results_text += f"95 percentile E2E Latency: {result['percentile_95_e2e_latency']}\n"
        results_text += "---------------------------------------------------------------------\n"
    return results_text




def get_tgi(results):
    results_text = ""
    for result in results:
        results_text += f"Request Rate: {result['rate']} RPM\n"
        results_text += f"Mean Batch Size : {result['average_tgi_batch_size']}\n"
        results_text += f"95%ile Batch Size : {result['percentile_95_batch_size']}\n"
        results_text += f"Mean TGIS Queue Waiting Time: {result['mean_queue_time']}\n"
        results_text += f"95 percentile Mean TGIS Queue Waiting Time: {result['percentile_95_queue_waiting_time']}\n"
        results_text += f"Mean E2E Latency: {result['mean_e2e_latency']}\n"
        results_text += f"95 percentile E2E Latency: {result['percentile_95_e2e_latency']}\n"
        results_text += "---------------------------------------------------------------------\n"
    return results_text
