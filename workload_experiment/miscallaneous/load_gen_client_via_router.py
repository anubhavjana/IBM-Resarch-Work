import yaml
import time
from datetime import datetime
import json

import numpy as np
from kubernetes import config, dynamic
from kubernetes.client import api_client
from openshift.dynamic import DynamicClient

import requests
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

NAMESPACE = "fmaas-monitoring"

class OpenShiftClient:
	def __init__(self):
		self.client = self.get_openshift_client()

	def get_openshift_client(self):
		c = None
		try:
			c = config.load_kube_config()
		except:
			print("Loading In cluster config")
			c = config.load_incluster_config()
		client = dynamic.DynamicClient(api_client.ApiClient(configuration=c))
		return client

	def get_resource(self, w_type):
		filename = f'manifests/{w_type}.yaml'
		file = open(filename, 'r')
		resource = yaml.safe_load(file)
		return resource

	def delete_pod(self, name):
		delete_pod = self.client.resources.get(api_version='v1', kind='Pod')
		try:
			resp = delete_pod.delete(name=name, namespace=NAMESPACE)
			print("Delete pod")
		except Exception as e:
			print("Error :", e.body)

	def create_pod(self, w_type):
		create_pod = self.client.resources.get(api_version='v1', kind='Pod')
		resource = self.get_resource(w_type)
		for env in resource['spec']['containers'][0]['env']:
			pass
			
			# if env['name'] == "USER":
			# 	env['value'] = user_id
			# if env['name'] == "REQUEST_RATE":
			# 	env['value'] = str(request_rate)
		try:
			resp = create_pod.create(body=resource, namespace=NAMESPACE)
			print(f"Create workload pod")
		except Exception as e:
			print("Error :", e.body)

	def watch(self, name):
		watch_pod = self.client.resources.get(api_version='v1', kind='Pod')
		for event in watch_pod.watch(namespace=NAMESPACE, timeout=5):
			if event['object']['metadata']['name'] == name:
				return event['object']['status']
		return None

	def log_data(self, start_time, end_time):
		"""
		Logs the experiment start and end time data to a JSON lines file.
		"""
		entry = {
			
			"start_time": start_time,
			"end_time": end_time,
			
		}

		file_name = "llama3_router_a100_10_June.jsonl"

		with open(file_name, "a") as f:
			json.dump(entry, f)
			f.write('\n')

def main():

	# create openshift client 
	openshift_client = OpenShiftClient()

	# delete the pod if exists before starting the experiment
	openshift_client.delete_pod("tgis-grpc-load-llama3-router")

	time.sleep(1)

	# mean = 1000
	# std = 300


	# user = "Alan"

	# for rr in range(10,120,10): # request rate
	
	openshift_client.create_pod("llama")

	start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

	time.sleep(2400) # time for which the experiment is carried out (in seconds)

	end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

	openshift_client.log_data(start, end)

	openshift_client.delete_pod("tgis-grpc-load-llama3-router") #experiment over ---  delete pod

	
	time.sleep(10)  # wait for the pod to shutdown gracefully



if __name__ == "__main__":
    main()
