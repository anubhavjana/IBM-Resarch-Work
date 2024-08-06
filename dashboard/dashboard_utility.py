import dash
import dash_uploader as du
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import subprocess
import os
import base64
import logging

def get_vllm_llama_a100_info():
    image_path_1 = 'vllm_llama_benchmark_numbers.png'
    image_path_2 = 'vllm_llama_numbers_2.png'
    image_path_3 = 'vllm_llama_queue_time_cdf.png'
    image_path_4 = 'vllm_e2e_cdf.png'
    
        
    images = []

    if os.path.exists(image_path_1):
        encoded_image_1 = base64.b64encode(open(image_path_1, 'rb').read()).decode('ascii')
        images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image_1), style={'width': '50%'}))
    else:
        logging.debug("Image file 1 not found.")
        images.append("Image file 1 not found.")

    if os.path.exists(image_path_2):
        encoded_image_2 = base64.b64encode(open(image_path_2, 'rb').read()).decode('ascii')
        images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image_2), style={'width': '50%'}))
    else:
        logging.debug("Image file 2 not found.")
        images.append("Image file 2 not found.")

    if os.path.exists(image_path_3):
        encoded_image_3 = base64.b64encode(open(image_path_3, 'rb').read()).decode('ascii')
        images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image_3), style={'width': '50%'}))
    else:
        logging.debug("Image file 3 not found.")
        images.append("Image file 3 not found.")

    if os.path.exists(image_path_4):
        encoded_image_4 = base64.b64encode(open(image_path_4, 'rb').read()).decode('ascii')
        images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image_4), style={'width': '50%'}))
    else:
        logging.debug("Image file 4 not found.")
        images.append("Image file 4 not found.")

    return images


def get_vllm_granite_a100_info():
    image_path_1 = 'vllm_granite_numbers.png'
    images = []

    if os.path.exists(image_path_1):
        encoded_image_1 = base64.b64encode(open(image_path_1, 'rb').read()).decode('ascii')
        images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image_1), style={'width': '50%'}))
    else:
        logging.debug("Image file 1 not found.")
        images.append("Image file 1 not found.")

    return images



def get_tgis_granite_a100_info():
    image_path_1 = 'tgis_granite_numbers.png'
    images = []

    if os.path.exists(image_path_1):
        encoded_image_1 = base64.b64encode(open(image_path_1, 'rb').read()).decode('ascii')
        images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image_1), style={'width': '50%'}))
    else:
        logging.debug("Image file 1 not found.")
        images.append("Image file 1 not found.")

    return images
