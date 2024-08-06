import yaml
import time
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

	def create_pod(self, w_type, threads=1):
		create_pod = self.client.resources.get(api_version='v1', kind='Pod')
		resource = self.get_resource(w_type)
		for env in resource['spec']['containers'][0]['env']:
			if env['name'] == "THREADS":
				env['value'] = str(threads)
		try:
			resp = create_pod.create(body=resource, namespace=NAMESPACE)
			print(f"Create workload pod with {threads}")
		except Exception as e:
			print("Error :", e.body)

	def watch(self, name):
		watch_pod = self.client.resources.get(api_version='v1', kind='Pod')
		for event in watch_pod.watch(namespace=NAMESPACE, timeout=5):
			if event['object']['metadata']['name'] == name:
				return event['object']['status']
		return None

c = OpenShiftClient()

c.delete_pod("tgis-grpc-load-llama")
c.delete_pod("tgis-grpc-load-granite")
# c.delete_pod("tgis-grpc-load-llama7b")
# c.delete_pod("tgis-grpc-load-llama13b")
time.sleep(1)

scale_up = "Normal"
mean = 1000
std = 300

for i in range(0, 500):
	if scale_up == "Normal":
		scale = abs(int(np.random.normal(mean, std)))
	else:
		scale = scale_up

	c.create_pod("llama", scale)
	c.create_pod("granite", scale)
	# c.create_pod("llama7b", scale)
	# c.create_pod("llama13b", scale)
	time.sleep(2 * 60)
	c.delete_pod("tgis-grpc-load-llama")
	c.delete_pod("tgis-grpc-load-granite")
	# c.delete_pod("tgis-grpc-load-llama7b")
	# c.delete_pod("tgis-grpc-load-llama13b")
	time.sleep(10)
