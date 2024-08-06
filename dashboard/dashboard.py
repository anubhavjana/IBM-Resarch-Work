import importlib
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
import zipfile
import logging
import json

import dashboard_utility
import print_results
import vllm_a100_llama_granite
import tgis_granite_a100
import tgis_llama_mig
import tgis_llama_a100
import tgis_bloom_mig
import tgis_bloom_a100

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(message)s')

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
du.configure_upload(app, folder='uploads')

# Encode the LLM Router image to base64
with open('images/llm_router.png', 'rb') as img_file:
    encoded_image_llm_router = base64.b64encode(img_file.read()).decode('utf-8')


def encode_image(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def generate_image_html(encoded_image, title):
    return html.Div([
        html.H3(title),
        html.Img(src=f'data:image/png;base64,{encoded_image}', style={'width': '100%', 'height': 'auto'}),
    ], style={'display': 'inline-block', 'width': '33%'})

with open('config.json') as config_file:
    config = json.load(config_file)

def generate_users(users):
    return html.Div([
        html.H3("Users: ", style={'color': 'red', 'display': 'inline'})] +
        [html.Span(user['name'], style={'color': user['color'], 'fontWeight': 'bold', 'marginRight': '10px'}) for user in users],
        style={'marginBottom': '20px'}
    )

def generate_buttons(buttons):
    return html.Div(
        [html.Button(button['label'], id=button['id'], n_clicks=0, style=button['style']) for button in buttons],
        style={'display': 'flex', 'justifyContent': 'space-between'}
    )

def generate_outputs(outputs):
    return [html.Div(id=output) for output in outputs]

def generate_dropdowns(dropdowns):
    return [html.Div([
                html.Label(dropdown['label']),
                dcc.Dropdown(
                    id=dropdown['id'],
                    options=dropdown['options'],
                    value=dropdown['default']
                )
            ]) for dropdown in dropdowns]

tabs = []
for tab in config['tabs']:
    tab_content = []
    if 'users' in tab:
        tab_content.append(generate_users(tab['users']))
    if 'buttons' in tab:
        tab_content.append(generate_buttons(tab['buttons']))
    if 'dropdowns' in tab:
        tab_content.extend(generate_dropdowns(tab['dropdowns']))
    if 'image' in tab:
        tab_content.append(html.Img(src=tab['image'], style={'width': '80%'}))
    if 'outputs' in tab:
        tab_content.extend(generate_outputs(tab['outputs']))

    tabs.append(dcc.Tab(label=tab['label'], children=tab_content))

app.layout = html.Div([
    html.H1(config['title']),
    dcc.Tabs(tabs)
])
# Global variable to store uploaded files
uploaded_files = []

def generate_image_layout(user):
    queue_time_image_path = f'images/queue_time_cdf_user_{user}.png'
    latency_image_path = f'images/latency_cdf_user_{user}.png'

    with open(queue_time_image_path, 'rb') as img_file:
        encoded_queue_time_image = base64.b64encode(img_file.read()).decode('utf-8')

    with open(latency_image_path, 'rb') as img_file:
        encoded_latency_image = base64.b64encode(img_file.read()).decode('utf-8')

    return html.Div([
        html.Div([
            html.H3(f"Queue Time CDF for {user}"),
            html.Img(src=f'data:image/png;base64,{encoded_queue_time_image}', style={'width': '100%', 'height': 'auto'}),
        ], style={'display': 'inline-block', 'width': '33%'}),

        html.Div([
            html.H3(f"Latency CDF for {user}"),
            html.Img(src=f'data:image/png;base64,{encoded_latency_image}', style={'width': '100%', 'height': 'auto'}),
        ], style={'display': 'inline-block', 'width': '33%'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'})


def get_cdf_plot_priority_non_priority(user, priority):
    if priority:
        queue_time_image_path = f'images/{user}_tgis_llama_queue_time_cdf.png'
        latency_image_path = f'images/{user}_tgis_llama_latency_cdf.png'
    else:
        queue_time_image_path = f'images/{user}_no_priority_tgis_llama_queue_time_cdf.png'
        latency_image_path = f'images/{user}_no_priority_tgis_llama_latency_cdf.png'

    encoded_queue_time_image = encode_image(queue_time_image_path)
    encoded_latency_image = encode_image(latency_image_path)

    queue_time_html = generate_image_html(encoded_queue_time_image, f"Queue Time CDF for {user}")
    latency_html = generate_image_html(encoded_latency_image, f"Latency CDF for {user}")

    return html.Div([
        queue_time_html,
        latency_html
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'})


# Call back for runing priority vs non-priority experiment ( max output tokens = 60)

@app.callback(
    [Output('priority-tgis-6-6-6-output', 'children'),
     Output('no-priority-tgis-6-6-6-output', 'children')],
    [Input('priority-tgis-6-6-6', 'n_clicks'),
     Input('no-priority-tgis-6-6-6', 'n_clicks')],
)

def run_priority_vs_non_priority(n_clicks_multi, n_clicks_llama_bloom):
    ctx = dash.callback_context

    if not ctx.triggered:
        return "", ""

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'priority-tgis-6-6-6':
       
        try:
            result = subprocess.run(['python3', 'priority_tgis.py'], capture_output=True, text=True)
            config_message = html.Div([
                html.H3("Configuration:"),
                html.P("Alan - Llama-8B [Priority = 1]"),
                html.P("Noel - Llama-8B [Priority = 2]"),
                html.P("Hari - Llama-8B [Priority = 3]"),
                html.P("RR = 6:6:6"),
                html.P("Maximum Output Tokens: 60")
                
            ], style={'color': 'blue', 'margin': '10px 0'})

            alan_images = get_cdf_plot_priority_non_priority('Alan',True)
            noel_images = get_cdf_plot_priority_non_priority('Noel',True)
            hari_images = get_cdf_plot_priority_non_priority('Hari',True)

            return [
                html.Div([config_message, html.Pre(result.stdout)]),
                html.Div([alan_images, noel_images, hari_images])
                
                
            ]
        except Exception as e:
            return [html.Pre(str(e)), ""]
        
    elif button_id == 'no-priority-tgis-6-6-6':
        try:
            result = subprocess.run(['python3', 'no_priority_tgis.py'], capture_output=True, text=True)
            config_message = html.Div([
                html.H3("Configuration:"),
                html.P("Alan - Llama-8B [Priority = 1]"),
                html.P("Noel - Llama-8B [Priority = 2]"),
                html.P("Hari - Llama-8B [Priority = 3]"),
                html.P("RR = 6:6:6"),
                html.P("Maximum Output Tokens: 60")
                
            ], style={'color': 'blue', 'margin': '10px 0'})

            alan_images = get_cdf_plot_priority_non_priority('Alan',False)
            noel_images = get_cdf_plot_priority_non_priority('Noel',False)
            hari_images = get_cdf_plot_priority_non_priority('Hari',False)

            return [
                html.Div([config_message, html.Pre(result.stdout)]),
                html.Div([alan_images, noel_images, hari_images])
                
                
            ]
        except Exception as e:
            return [html.Pre(str(e)), ""]

        


# Define the callback for running the multi-instance script
@app.callback(
    [Output('multi-instance-output', 'children'),
     Output('llama-bloom-output', 'children'),
     Output('tgis-all-pod-output', 'children'),
     Output('tgis-llama-a100-4-4-4-output', 'children')],
    [Input('tgis-multi-instance-10-15-20', 'n_clicks'),
     Input('run-tgis-llama-bloom', 'n_clicks'),
     Input('run-tgis-llama', 'n_clicks'),
     Input('run-tgis-llama-a100-4-4-4', 'n_clicks')],
)
def run_priority_based_multi_instance(n_clicks_multi, n_clicks_llama_bloom, n_clicks_llama,n_clicks_all_llama_4_4_4):
    ctx = dash.callback_context

    if not ctx.triggered:
        return "", "", "", ""

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'tgis-multi-instance-10-15-20':
       
        try:
            result = subprocess.run(['python3', 'tgis_multi_instance_router_10.15.20.py'], capture_output=True, text=True)
            config_message = html.Div([
                html.H3("Configuration:"),
                html.P("Alan - llama3-1, llama3-2, llama3-3 (34%-33%-33%) [Priority = 1]"),
                html.P("Noel - llama3-1, llama3-2, llama3-3 (34%-33%-33%) [Priority = 2]"),
                html.P("Hari - llama3-1, llama3-2, llama3-3 (34%-33%-33%) [Priority = 3]"),
                html.P("RR = 10:15:20"),
                html.P("Maximum Output Tokens: 200")
            ], style={'color': 'blue', 'margin': '10px 0'})

            alan_images = generate_image_layout('Alan')
            noel_images = generate_image_layout('Noel')
            hari_images = generate_image_layout('Hari')

            return [
                html.Div([config_message, html.Pre(result.stdout)]),
                html.Div([alan_images, noel_images, hari_images]),
                "",
                ""
            ]
            # return [html.Div([config_message, html.Pre(result.stdout)]), "", ""]
        except Exception as e:
            return [html.Pre(str(e)), "", "",""]
    elif button_id == 'run-tgis-llama-bloom':
        try:
            result = subprocess.run(['python3', 'tgis_multi_instance_llama_bloom_4.4.4.py'], capture_output=True, text=True)
            config_message = html.Div([
                html.H3("Configuration:"),
                html.P("Alan -- Llama3-Mig (40 %) [Priority = 1], Bloom760-Mig (60 %) [Priority = 2]"),
                html.P("Noel -- Bloom760-Mig (100 %) [Priority = 3]"),
                html.P("Hari -- Llama3-Mig (100 %) [Priority = 1]"),
                html.P("RR = 4:4:4"),
                html.P("Maximum Output Tokens: 200")
            ], style={'color': 'blue', 'margin': '10px 0'})
            return ["", html.Div([config_message, html.Pre(result.stdout)]), "", ""]
        except Exception as e:
            return ["", html.Pre(str(e)), "",""]
        
    elif button_id == 'run-tgis-llama-a100-4-4-4':
        try:
            result = subprocess.run(['python3', 'tgis_multiinstance_lllama_a100_4_4_4.py'], capture_output=True, text=True)
            config_message = html.Div([
                html.H3("Configuration:"),
                html.P("Alan -- Llama3 A100 (100 %) [Priority = 1]"),
                html.P("Noel -- Llama3 A100 (100 %) [Priority = 2]"),
                html.P("Hari -- Llama3 A100 (100 %) [Priority = 3]"),
                html.P("RR = 4:4:4"),
                html.P("Maximum Output Tokens: 200")
            ], style={'color': 'blue', 'margin': '10px 0'})
            return ["", "",html.Div([config_message, html.Pre(result.stdout)]),""]
        except Exception as e:
            return ["", "", html.Pre(str(e)),""]
    elif button_id == 'run-tgis-llama':
        try:
            result = subprocess.run(['python3', 'tgis_users_to_same_pod_4.4.4.py'], capture_output=True, text=True)
            config_message = html.Div([
                html.H3("Configuration:"),
                html.P("Alan -- Llama3-Mig (100 %) [Priority = 1]"),
                html.P("Noel -- Llama3-Mig (100 %) [Priority = 2]"),
                html.P("Hari -- Llama3-Mig (100 %) [Priority = 3]"),
                html.P("RR = 4:4:4"),
                html.P("Maximum Output Tokens: 200")
            ], style={'color': 'blue', 'margin': '10px 0'})
            return ["", "", "",html.Div([config_message, html.Pre(result.stdout)])]
        except Exception as e:
            print("Error",e)
            return ["", "", "",html.Pre(str(e))]
    return "", "", "",""

@app.callback(
    [Output('selection-output', 'children'), Output('image-output', 'children')],
    [Input('inference-server-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('gpu-dropdown', 'value')]
)
def update_selection(inference_server, model, gpu_type):

    default_message = f"No benchmarking exists for {inference_server} for model {model} on {gpu_type}"
    default_image = []

    message = f"You have chosen {inference_server} for model {model} on {gpu_type}"
    logging.debug(f"Selections - Inference Server: {inference_server}, Model: {model}, GPU Type: {gpu_type}")

    benchmark_info = html.Div(
        "The benchmark experiment was run for 40 minutes by discarding the first 8 mins and the last 3 minutes of the experiment for stable state",
        style={'color': 'red', 'margin': '10px 0'}
    )

    ################################################### VLLM-Llama8b-A100 ###################################################
    if inference_server == 'vllm' and model == 'llama3-8b' and gpu_type == 'a100-80gb':
        model = 'llama'
        plot_vllm = importlib.import_module('vllm_a100_llama_granite')
        results, tgi_data_dict = plot_vllm.analyze_model(model)
        plot_vllm.plot_results(tgi_data_dict, model)

        image_paths = [
            "images/" + model + "_mean_queue_time.png",
            "images/" + model + "_queue_size.png",
            "images/" + model + "_e2e_latency.png"
        ]

        images = []

        for image_path in image_paths:
            if os.path.exists(image_path):
                encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
                images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '30%', 'margin': '10px'}))
            else:
                logging.debug(f"Image file {image_path} not found.")
                images.append(html.Div(f"Image file {image_path} not found."))

        results_text = print_results.get_vllm(results)

        saturation_message = html.Div([
        html.H3("Saturation Point: 80 RPM"),
        html.H3("Output Tokens: 60"),
        html.P("This is the point where the system starts to saturate under the given workload conditions."),
        ], style={'color': 'blue', 'margin': '10px 0'})

        return html.Div([benchmark_info, saturation_message, html.Pre(results_text)]), images
        return html.Div([benchmark_info, html.Pre(results_text)]), images
    
    ################################################### VLLM-Granite-A100 ###################################################

    if inference_server == 'vllm' and model == 'granite-3b' and gpu_type == 'a100-80gb':
        model = 'granite'
        plot_vllm = importlib.import_module('vllm_a100_llama_granite')
        results, tgi_data_dict = plot_vllm.analyze_model(model)
        plot_vllm.plot_results(tgi_data_dict, model)

        image_paths = [
            "images/" + model + "_mean_queue_time.png",
            "images/" + model + "_queue_size.png",
            "images/" + model + "_e2e_latency.png"
        ]

        images = []

        for image_path in image_paths:
            if os.path.exists(image_path):
                encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
                images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '30%', 'margin': '10px'}))
            else:
                logging.debug(f"Image file {image_path} not found.")
                images.append(html.Div(f"Image file {image_path} not found."))


        results_text = print_results.get_vllm(results)
        saturation_message = html.Div([
        html.H3("Saturation Point: 300 RPM"),
        html.H3("Output Tokens: 60"),
        html.P("This is the point where the system starts to saturate under the given workload conditions."),
        ], style={'color': 'blue', 'margin': '10px 0'})

        return html.Div([benchmark_info, saturation_message, html.Pre(results_text)]), images
        return html.Div([benchmark_info, html.Pre(results_text)]), images
    
    ################################################### TGIS-Granite-A100 ###################################################

    if inference_server == 'tgis' and model == 'granite-3b' and gpu_type == 'a100-80gb':
        model = 'granite'
        plot_vllm = importlib.import_module('tgis_granite_a100')
        results, tgi_data_dict = plot_vllm.analyze_model(model)
        plot_vllm.plot_results(tgi_data_dict, model)

        image_paths = [
            "images/" + model + "_tgis_mean_queue_time.png",
            "images/" + model + "_tgis_e2e_latency.png"
        ]

        images = []

        for image_path in image_paths:
            if os.path.exists(image_path):
                encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
                images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '30%', 'margin': '10px'}))
            else:
                logging.debug(f"Image file {image_path} not found.")
                images.append(html.Div(f"Image file {image_path} not found."))

        results_text = print_results.get_tgi(results)

        saturation_message = html.Div([
        html.H3("Saturation Point: 50 RPM"),
        html.H3("Output Tokens: 60"),
        html.P("This is the point where the system starts to saturate under the given workload conditions."),
        ], style={'color': 'blue', 'margin': '10px 0'})

        return html.Div([benchmark_info, saturation_message, html.Pre(results_text)]), images
        return html.Div([benchmark_info, html.Pre(results_text)]), images
    

    ###################################################  TGIS-Llama-Mig   ###################################################

    if inference_server == 'tgis' and model == 'llama3-8b' and gpu_type == 'mig-3g-40gb':
        model = 'llama'
        plot_vllm = importlib.import_module('tgis_llama_mig')
        results, tgi_data_dict = plot_vllm.analyze_model(model)
        plot_vllm.plot_results(tgi_data_dict, model)

        image_paths = [
            "images/" + model + "_mig_tgis_mean_queue_time.png",
            "images/" + model + "_mig_tgis_queue_size.png",
            "images/" + model + "_mig_tgis_e2e_latency.png"
        ]

        images = []

        for image_path in image_paths:
            if os.path.exists(image_path):
                encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
                images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '30%', 'margin': '10px'}))
            else:
                logging.debug(f"Image file {image_path} not found.")
                images.append(html.Div(f"Image file {image_path} not found."))

        results_text = print_results.get_tgi(results)

        return html.Div([benchmark_info, html.Pre(results_text)]), images
    
    ###################################################  TGIS-Bloom-Mig   ###################################################

    if inference_server == 'tgis' and model == 'bloom-760' and gpu_type == 'mig-3g-40gb':
        model = 'bloom'
        plot_vllm = importlib.import_module('tgis_bloom_mig') # just change this file name when you want to add a new configuration
        results, tgi_data_dict = plot_vllm.analyze_model(model)
        plot_vllm.plot_results(tgi_data_dict, model)

        image_paths = [
            "images/" + model + "_mig_tgis_mean_queue_time.png",
            "images/" + model + "_mig_tgis_queue_size.png",
            "images/" + model + "_mig_tgis_e2e_latency.png"
        ]

        images = []

        for image_path in image_paths:
            if os.path.exists(image_path):
                encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
                images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '30%', 'margin': '10px'}))
            else:
                logging.debug(f"Image file {image_path} not found.")
                images.append(html.Div(f"Image file {image_path} not found."))

        results_text = print_results.get_tgi(results)

        saturation_message = html.Div([
        html.H3("Saturation Point: 70 RPM"),
        html.P("This is the point where the system starts to saturate under the given workload conditions."),
        ], style={'color': 'blue', 'margin': '10px 0'})

        return html.Div([benchmark_info, saturation_message, html.Pre(results_text)]), images
    

    ###################################################  TGIS-Bloom-A100   ###################################################

    if inference_server == 'tgis' and model == 'bloom-760' and gpu_type == 'a100-80gb':
        model = 'bloom'
        plot_vllm = importlib.import_module('tgis_bloom_a100')
        results, tgi_data_dict = plot_vllm.analyze_model(model)
        plot_vllm.plot_results(tgi_data_dict, model)

        image_paths = [
            "images/" + model + "_a100_tgis_mean_queue_time.png",
            "images/" + model + "_tgis_e2e_latency.png"
        ]

        images = []

        for image_path in image_paths:
            if os.path.exists(image_path):
                encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
                images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '30%', 'margin': '10px'}))
            else:
                logging.debug(f"Image file {image_path} not found.")
                images.append(html.Div(f"Image file {image_path} not found."))

        results_text = print_results.get_tgi(results)

        saturation_message = html.Div([
        html.H3("Saturation Point: 155 RPM"),
        html.H3("Output Tokens = 60"),
        html.P("This is the point where the system starts to saturate under the given workload conditions."),
        ], style={'color': 'blue', 'margin': '10px 0'})

        return html.Div([benchmark_info, saturation_message, html.Pre(results_text)]), images

    
    ###################################################  TGIS-Llama-A100   ###################################################

    if inference_server == 'tgis' and model == 'llama3-8b' and gpu_type == 'a100-80gb':
        model = 'llama'
        plot_vllm = importlib.import_module('tgis_llama_a100')
        results, tgi_data_dict = plot_vllm.analyze_model(model)
        plot_vllm.plot_results(tgi_data_dict, model)

        image_paths = [
            "images/" + model + "_tgis_mean_queue_time.png",
            "images/" + model + "_tgis_e2e_latency.png"
        ]

        images = []

        for image_path in image_paths:
            if os.path.exists(image_path):
                encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
                images.append(html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'width': '30%', 'margin': '10px'}))
            else:
                logging.debug(f"Image file {image_path} not found.")
                images.append(html.Div(f"Image file {image_path} not found."))

        results_text = print_results.get_tgi(results)


        saturation_message = html.Div([
        html.H3("Saturation Point: 15 RPM"),
        html.H3("Output Tokens = 60"),
        html.P("This is the point where the system starts to saturate under the given workload conditions."),
        ], style={'color': 'blue', 'margin': '10px 0'})

        return html.Div([benchmark_info, saturation_message, html.Pre(results_text)]), images
        # return html.Div([benchmark_info, html.Pre(results_text)]), images
    
   
    
    return html.Div(default_message), default_image


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
