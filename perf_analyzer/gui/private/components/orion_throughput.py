##########################################################################
##  Web GUI for LPU Performance Profiling
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

import os
import altair as alt
import pandas as pd
# Gradio Library
import gradio as gr

# Model information
from perf_analyzer.models import AutoConfig, AutoModelForCausalLM
# Utilites
from perf_analyzer.utils import orion_throughput

##########################################################################
## Global
##########################################################################

predictor = orion_throughput.UtilPredictor("./data/throughput.json")

##########################################################################
## Function
##########################################################################

# Download models from web
def download_model(model_id: str):
  # HuggingFace Access Token
  access_token = "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"
  # Check if the model is valid
  try:
    model = AutoModelForCausalLM.from_pretrained(model_id=model_id, token=access_token)
    return model
  except:
    raise gr.Error("Checkpoint is Wrong! Please check again.")

def get_model_throughput(model_id: str, input_token: int, output_token: int):
  # Download HuggingFace model
  model = download_model(model_id)
  # Calculate model size
  num_devices_list = [1, 2, 4, 8, 16]
  perf_list = []
  for n in num_devices_list:
    perf = orion_throughput.predict(model, predictor, tp=n)
    perf = perf * (output_token) / (input_token + output_token - 1)
    perf_list.append(perf)
  return (
    "{:.3f} token/sec".format(perf_list[0]),
    "{:.3f} token/sec".format(perf_list[1]),
    "{:.3f} token/sec".format(perf_list[2]),
    "{:.3f} token/sec".format(perf_list[3]),
    "{:.3f} token/sec".format(perf_list[4]),
    scalability_plot(model, input_token, output_token)
  )

def scalability_plot(model: AutoModelForCausalLM, input_token: int, output_token: int):
  # Candidates
  num_devices_list = [1, 2, 4, 8]
  perf_list = []
  # Get each throughputs
  for n in num_devices_list:
    perf = orion_throughput.predict(model, predictor, tp=n)
    perf = perf * (output_token) / (input_token + output_token - 1)
    perf_list.append(perf)
  # Plot graph
  data = pd.DataFrame(
    {
      "Scalability": [y / perf_list[0] for y in perf_list],
      "Number of Devices": [str(x) for x in num_devices_list]
    }
  )
  return alt.Chart(data).mark_line(point=True, strokeWidth=3).encode(
    x="Number of Devices",
    y="Scalability",
    color=alt.value("#12A5E9")
  ).properties(
    width=400,
    height=200
  )

##########################################################################
## Gradio Page Layout
##########################################################################

def page():

  # User input for checkpoint
  with gr.Row(equal_height=True, variant="panel"):
    with gr.Column(scale=4):
      # Model checkpoint
      model_id_box = gr.Textbox(
        label="HuggingFace Checkpoint Name",
        placeholder="facebook/opt-125m"
      )
      # Checkpoint examples
      gr.Examples(
        examples=[
          "huggyllama/llama-7b",
          "huggyllama/llama-13b",
          "huggyllama/llama-30b",
          "meta-llama/Llama-2-70b-hf",
          "mistralai/Mistral-7B-v0.1",
          "mistralai/Mixtral-8x7B-v0.1",
          "facebook/opt-6.7b",
          "facebook/opt-13b",
          "facebook/opt-30b"
        ],
        inputs=[model_id_box],
        label="HuggingFace Checkpoint Examples"
      )
      with gr.Row():
        # Input token length
        input_token_slider = gr.Slider(
          label="Input Token Length",
          minimum=1,
          maximum=65536,
          value=1
        )
        # Output token length
        output_token_slider = gr.Slider(
          label="Output Token Length",
          minimum=1,
          maximum=65536,
          value=1
        )
        # Batch size
        batch_size_slider = gr.Slider(
          label="Batch Size",
          minimum=1,
          maximum=1024,
          value=1
        )
      
    with gr.Column(scale=1):
      # Submit button
      submit_btn = gr.Button(
        value="Submit",
        variant="primary",
        scale=3
      )
      # Clear button
      clear_btn = gr.ClearButton(
        components=[
          model_id_box,
          input_token_slider,
          output_token_slider,
          batch_size_slider
        ],
        value="Clear",
        variant="secondary",
        scale=1
      )

  # Output
  with gr.Row(equal_height=True):
    with gr.Column(variant="panel"):
      # Throughput estimation
      throughput_1_box = gr.Textbox(
        label="Throughput Prediction for 1 device",
        placeholder="Report the throughput in token/sec"
      )
      with gr.Row(equal_height=True):
        # Throughput estimation
        throughput_2_box = gr.Textbox(
          label="Throughput Prediction for 2 device",
          placeholder="Report the throughput in token/sec"
        )
        # Throughput estimation
        throughput_4_box = gr.Textbox(
          label="Throughput Prediction for 4 device",
          placeholder="Report the throughput in token/sec"
        )
      with gr.Row(equal_height=True):
        # Throughput estimation
        throughput_8_box = gr.Textbox(
          label="Throughput Prediction for 8 device",
          placeholder="Report the throughput in token/sec"
        )
        # Throughput estimation
        throughput_16_box = gr.Textbox(
          label="Throughput Prediction for 16 device",
          placeholder="Report the throughput in token/sec"
        )
    with gr.Column(variant="panel"):
      # Scalability report
      scalability_plot = gr.Plot(
        label="Scalability of LPU"
      )

  # Button events
  # Submit button
  submit_btn.click(
    fn=get_model_throughput,
    inputs=[
      model_id_box,
      input_token_slider,
      output_token_slider
    ],
    outputs=[
      throughput_1_box,
      throughput_2_box,
      throughput_4_box,
      throughput_8_box,
      throughput_16_box,
      scalability_plot
    ]
  )

##########################################################################