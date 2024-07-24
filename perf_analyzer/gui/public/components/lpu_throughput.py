##########################################################################
##  Web GUI for LPU Performance Profiling
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

import os
import pandas as pd
# Gradio Library
import gradio as gr

# Model information
from perf_analyzer.models import AutoConfig, AutoModelForCausalLM
# Utilites
from perf_analyzer.utils import lpu_throughput
from perf_analyzer.utils.lpu_throughput import HardwareSpec

##########################################################################
## Defines
##########################################################################

# Architecture Configurations
arch = HardwareSpec(
  mac_per_core  = 64 * 16 * 2,  # (Vector Dimension) x (Vector Lane)
  num_core      = 8 * 16,       # 2D-array
  frequency     = 1.5,          # GHz
  max_batch     = 32,           # Maximum support batch size
  sum_util      = 0.9,          # Summarization utilization
  mem_util      = 0.9,          # Memory bandwidth utilization
  area          = 190,          # mm^2
  power         = 120,          # W
  memory        = "LPDDR5X",
  bandwidth     = 512,          # GB/s
  capacity      = 128           # GB
)

model_list = [
  "EleutherAI/gpt-j-6b",
  "meta-llama/Llama-2-7b-hf",
  "meta-llama/Llama-2-13b-hf",
  "meta-llama/Llama-2-70b-hf",
  "mistralai/Mistral-7B-v0.1",
  "google/gemma-2b",
  "google/gemma-7b",
  "microsoft/phi-2",
  "tiiuae/falcon-7b",
  "tiiuae/falcon-40b",
  "tiiuae/falcon-180b"
]

##########################################################################
## Function
##########################################################################

# Download models from web
def download_model(model_id: str):
  # HuggingFace Access Token
  access_token = "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"
  if model_id not in model_list:
    raise gr.Error("Checkpoint is not in the example list! Please check again.")
  # Check if the model is valid
  try:
    model = AutoModelForCausalLM.from_pretrained(model_id=model_id, token=access_token)
    return model
  except:
    raise gr.Error("Checkpoint is Wrong! Please check again.")

# Calculate throughput
def throughput(model_id, input_token, output_token, batch_size, data_type):
  # Download from HuggingFace
  model = download_model(model_id)
  n_op = model.n_op()
  # Calculate peak performance
  performance = lpu_throughput.peak_performance(data_type, arch)
  # Calculate the overal throughput
  throughput, _ = lpu_throughput.throughput(model, input_token, output_token, batch_size, data_type, arch)
  return (
    "{:.2f} TFLOPS".format(performance),
    "{:.3f} token/sec".format(throughput),
    make_plot(model, data_type)
  )

def make_plot(model, data_type):
  # Data point candidates
  ratio_list = [0.125, 0.25, 0.5, 0.625, 0.75, 0.875]
  batch_list = [1, 2, 4, 8, 16, 32]
  label_list = []
  data_list = []
  for ratio in ratio_list:
    input_token = int(model.config.max_length * ratio)
    output_token = int(model.config.max_length - input_token)
    label = "{:d}-{:d}".format(input_token, output_token)
    throughput_list = []
    for batch_size in batch_list:
      # Calculate the overal throughput
      throughput = lpu_throughput.throughput(model, input_token, output_token, batch_size, data_type, arch)
      throughput_list.append(throughput)
    label_list.append(label)
    data_list.append(throughput_list)
  # Convert to pandas
  dataset = pd.DataFrame(data_list, columns=batch_list, index=label_list).transpose()
  dataset.index.name = "Batch Size"
  dataset = dataset.reset_index().melt('Batch Size', var_name='category', value_name='Performance (token/sec)')
  return dataset

##########################################################################
## Gradio Page Layout
##########################################################################

def page():

  # User input for checkpoint
  with gr.Row():
    with gr.Column(variant="panel", scale=4):
      with gr.Row():
        # Model checkpoint
        model_id_box = gr.Textbox(
          label="HuggingFace Checkpoint Name",
          placeholder="facebook/opt-125m"
        )
        # Data precision
        data_type_radio = gr.Radio(
          choices=["float32", "float16", "bfloat16", "float8", "int8"],
          value="float16",
          label="Data Precision"
        )
      # Checkpoint examples
      gr.Examples(
        examples=model_list,
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
    with gr.Column(variant="panel", scale=1):
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
          batch_size_slider,
          data_type_radio
        ],
        value="Clear",
        variant="secondary",
        scale=1
      )

  with gr.Row():
    with gr.Column(variant="panel"):
      with gr.Row():
        # Peak performance
        performance_box = gr.Textbox(
          label="Peak Performance"
        )
        # Overall throughput
        throughput_box = gr.Textbox(
          label="Throughput"
        )
    with gr.Column(variant="panel"):
      throughput_plot = gr.LinePlot(
        x="Batch Size",
        y="Performance (token/sec)",
        color="category",
        color_legend_position="bottom",
        tooltip=["Batch Size", "Performance (token/sec)", "category"],
        width=600,
        height=300
      )

  # Button events
  # Submit button
  submit_btn.click(
    fn=throughput,
    inputs=[
      model_id_box,
      input_token_slider,
      output_token_slider,
      batch_size_slider,
      data_type_radio
    ],
    outputs=[
      performance_box,
      throughput_box,
      throughput_plot
    ]
  )

##########################################################################