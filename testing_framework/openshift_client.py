import yaml
import time
import json
from datetime import datetime

from kubernetes import config, dynamic
from kubernetes.client import api_client
from openshift.dynamic import DynamicClient
import requests
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

NAMESPACE = "fmaas-monitoring"
POLL_INTERVAL = 5  # Interval in seconds between each poll to fetch IP address of router pod
MAX_RETRIES = 12  # Maximum number of retries (1 minute)
LOG_FILE = 'log.jsonl'


class OpenShiftClient:
    def __init__(self):
        self.client = self.get_openshift_client()

    def get_openshift_client(self):
        try:
            config.load_kube_config()
        except:
            print("Loading In-cluster config")
            config.load_incluster_config()
        return dynamic.DynamicClient(api_client.ApiClient())

    def get_resource(self, w_type):
        filename = f'manifests/{w_type}.yaml'
        with open(filename, 'r') as file:
            resource = yaml.safe_load(file)
        return resource

    def delete_pod(self, name):
        delete_pod = self.client.resources.get(api_version='v1', kind='Pod')
        try:
            delete_pod.delete(name=name, namespace=NAMESPACE)
            print(f"Deleted pod: {name}")
            self.log_pod_activity(name, end_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        except Exception as e:
            print(f"Error deleting pod {name}: {e.body} or pod is not present")

    def create_pod(self, w_type, env_vars, name_suffix=""):
        create_pod = self.client.resources.get(api_version='v1', kind='Pod')
        resource = self.get_resource(w_type)
        resource['metadata']['name'] += name_suffix
        for env in resource['spec']['containers'][0]['env']:
            if env['name'] in env_vars:
                env['value'] = env_vars[env['name']]
        try:
            create_pod.create(body=resource, namespace=NAMESPACE)
            print(f"Created {w_type} pod with suffix {name_suffix}")
            self.log_pod_activity(resource['metadata']['name'], start_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        except Exception as e:
            print(f"Error creating {w_type} pod: {e.body}")

    # Create the multi-instance router pod
    def create_router_pod(self, updated_users_map, counter):
        env_vars = {
            "UPDATED_USERS_MAP": json.dumps(updated_users_map)
        }
        router_name_suffix = f"-{counter}"
        
        self.create_pod('multi-instance-router', env_vars, router_name_suffix)
        router_ip = self.poll_pod_ip(f"llm-router-multi-instance-v4{router_name_suffix}")
        return router_ip



    # Create the load generator pod
    def create_load_generator_pod(self, users_config, router_ip, counter):
        env_vars = {
            "USERS": json.dumps(users_config),
            "REQUEST_URL": f"{router_ip}:8032"
        }
        load_gen_name_suffix = f"-{counter}"
        self.create_pod('load-generator', env_vars, load_gen_name_suffix)

    # Poll till the pod gets an IP and return the IP
    def poll_pod_ip(self, pod_name):
        pods = self.client.resources.get(api_version='v1', kind='Pod')
        for _ in range(MAX_RETRIES):
            pod = pods.get(name=pod_name, namespace=NAMESPACE)
            pod_ip = pod.status.podIP
            if pod_ip:
                return pod_ip
            time.sleep(POLL_INTERVAL)
        raise TimeoutError(f"IP address for pod {pod_name} not assigned within the expected time.")
    
    # Log the start and end time of a pod
    def log_pod_activity(self, pod_name, start_time=None, end_time=None):
        if start_time:
            log_entry = {
                "pod_name": pod_name,
                "start_time": start_time,
                "end_time": None
            }
            with open(LOG_FILE, 'a') as log_file:
                log_file.write(json.dumps(log_entry) + '\n')
        if end_time:
            with open(LOG_FILE, 'r') as log_file:
                logs = log_file.readlines()
            with open(LOG_FILE, 'w') as log_file:
                for log_entry in logs:
                    entry = json.loads(log_entry)
                    if entry["pod_name"] == pod_name:
                        entry["end_time"] = end_time
                    log_file.write(json.dumps(entry) + '\n')

def generate_load_generator_config(request_rates, latencies):
    users_config = []
    for i, user_id in enumerate(["Alan", "Noel"]): # add Hari
        user_config = {
            "user_id": user_id,
            "model_id": "llama3",
            "latency": latencies[i],
            "request_rate": request_rates[i],
            "out_token_size": 60,
            "batch_size": 1,
            "variable_token_size": False,
            "iterations": 1800
        }
        users_config.append(user_config)
    return users_config

def read_config_file(config_file):
    with open(config_file, 'r') as file:
        config_data = json.load(file)
    return config_data

def main():
    openshift_client = OpenShiftClient()

    router_ips = {}


    def update_load_gen_config(config_file,router_ip):
        config_data = read_config_file(config_file)
        updates = config_data.get("updates", [])

        time_to_wait = 0
        index = 0
        
        for i, update in enumerate(updates):

            index = i
            time_to_wait = update["time"] * 60
            request_rates = update["request_rates"]
            latencies = update["latencies"]
            if i == 0:
                # Skip the first update as it's already processed
                continue
            
            previous_router_ip = router_ips[i-1]

        
                
            print(f"Starting new Load Generator after {update['time']} minutes...")
            time.sleep(time_to_wait)
            openshift_client.delete_pod(f"load-generator-tgis-{i-1}")
            # openshift_client.delete_pod(f"llm-router-multi-instance-v4-{i-1}")

            load_gen_config = generate_load_generator_config(request_rates, latencies)
            print(f'Create load generator - Iteration {i}')

            # create new load generator with new request rates which will send to old router for 3 mins to show latency rising
            # with higher request rates and strict latency SLOs

            openshift_client.create_load_generator_pod(load_gen_config, previous_router_ip, i)

            print(f"New load-generator-tgis-{i} sending request to the router llm-router-multi-instance-v4-{i-1} for 3 mins..")

            time.sleep(180)

            # delete the old router
            openshift_client.delete_pod(f"llm-router-multi-instance-v4-{i-1}")


            # update new router decision
            with open('user_config.json', 'r') as router_rules:
                updated_users_map = json.load(router_rules)

            try:
                router_ip = openshift_client.create_router_pod(updated_users_map, i)
                router_ips[i] = router_ip
                print(f"Router IP Address: {router_ip} --> Iteration {i}")
            except TimeoutError as e:
                print(e)
                return time_to_wait, index
            

            # update --> delete + create new 
            # delete the new load gen using old router 
            print(f'Updating load-generator-tgis-{i}..\n')
            openshift_client.delete_pod(f"load-generator-tgis-{i}")
            time.sleep(10)
            # create the load gen to point to new router
            openshift_client.create_load_generator_pod(load_gen_config, router_ip, i) 



        
        return time_to_wait,index

    ############################################################ INIIAL CONFIG ####################################################################################
    
     
    # Create initial router 
    with open('user_config.json', 'r') as router_rules:
        updated_users_map = json.load(router_rules)

    try:
        router_ip = openshift_client.create_router_pod(updated_users_map, 0)
        router_ips[0] = router_ip
        print(f"Router IP Address: {router_ip} ---> Iteration 0")
    except TimeoutError as e:
        print(e)
        return
    # Create load generator
    # Read initial configuration
    config_data = read_config_file('input_config.json')
    updates = config_data.get("updates", [])
    
   
    initial_update = updates[0]
    initial_request_rates = initial_update["request_rates"]
    initial_latencies = initial_update["latencies"]

    # Process the initial update
    load_gen_config = generate_load_generator_config(initial_request_rates, initial_latencies)
    print(f'Create load generator - Iteration 0')
    openshift_client.create_load_generator_pod(load_gen_config, router_ip,0)


    ############################################################ UPDATE  ####################################################################################

    time_to_wait , latest_index = update_load_gen_config('input_config.json',router_ip)


    ############################################################ CLEAN UP  ####################################################################################


    # Delete the last exisiting load generator and router  after completing all time intervals
    print(f"Cleaning up the last pods after waiting for {time_to_wait//60} minutes...\n")
    time.sleep(time_to_wait)
    openshift_client.delete_pod(f"load-generator-tgis-{latest_index}")
    openshift_client.delete_pod(f"llm-router-multi-instance-v4-{latest_index}")


    

if __name__ == "__main__":
    main()
