import requests
import datetime
import json
import time

url = "http://127.0.0.1:8080/metrics"
filename = "result.csv"
sleep_time = 30
scrape_duration = 3

file = open(filename, 'a')
file.write('[')

def get_metrics(url):
	res = requests.get(url)
	if res.status_code == 200:
		data = res.text
		data = data.split("\n")
		return data
	return ""

def process_metrics(data):
	metric_dict = {'timestamp':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

	for metric in data:
		if len(metric) == 0:
			continue
		if '#' in metric:
			continue
		if 'bucket' in metric:
			continue
		m = metric.split(' ')
		print(m)
		metric_dict[m[0]] = m[1]
	return json.dumps(metric_dict)

def scrape_for_n_minutes(n):
	end_time = time.time() + 60 * n
	while time.time() < end_time:	
		data = get_metrics(url)
		file.write(process_metrics(data))
		file.write(',\n')
		time.sleep(1 * sleep_time)

scrape_for_n_minutes(scrape_duration)
file.write(']')
file.close()
