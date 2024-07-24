##########################################################################
##  Web GUI for LPU Performance Profiling
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

import os
import math
import altair as alt
import pandas as pd
# Gradio Library
import gradio as gr
import matplotlib.pyplot as plt

# Model information
from perf_analyzer.models import AutoConfig, AutoModelForCausalLM
# Utilites
from perf_analyzer.utils import lpu_throughput
from perf_analyzer.utils.lpu_throughput import HardwareSpec

##########################################################################
## Defines
##########################################################################

# Architecture Configurations
MAC               = 32*32*2       # (Vector Dimension) x (Vector Lane) x 2
NUM_CORE          = 512        # 2D-array
MAX_BATCH         = 512            # Maximum supported Batch
LOGIC_FREQUENCY   = 1           # GHz
CHIP_AREA         = 190           # mm^2
MAX_POWER         = 120           # W
# Memory Specification
MEMORY_TYPE       = "LPDDR5X"
MEMORY_CHANNEL    = 8
MEMORY_BANDWIDTH  = 1100           # GB/s
MEMORY_CAPACITY   = 512           # GB

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

# Calculate throughput
def throughput(model_id, input_token, output_token, batch_size, data_type, logic_spec, memory_spec, sum_util, gen_util, mem_util):
  # Hardware configuration
  arch = HardwareSpec(
    mac_per_core=int(logic_spec.iloc[0, 0]),
    num_core=int(logic_spec.iloc[0, 1]),
    frequency=float(logic_spec.iloc[0, 2]),
    max_batch=int(logic_spec.iloc[0, 3]),
    sum_util=float(sum_util),
    gen_util=float(gen_util),
    mem_util=float(mem_util),
    area=int(logic_spec.iloc[0, 4]),
    power=int(logic_spec.iloc[0, 5]),
    memory=str(memory_spec.iloc[0, 0]),
    capacity=int(memory_spec.iloc[0, 2]),
    bandwidth=float(memory_spec.iloc[0, 3])
  )
  # Download from HuggingFace
  model = download_model(model_id)
  # Calculate peak performance
  performance = lpu_throughput.peak_performance(data_type, arch)
  # Calculate the overal throughput
  throughput, latency = lpu_throughput.throughput(model, input_token, output_token, batch_size, data_type, arch)
  return (
    "{:.2f} TFLOPS".format(performance),
    "{:.3f} token/sec".format(throughput),
    "{:.3f} token/sec/W".format(throughput / arch.power),
    "{:.3f} token/sec/mm2".format(throughput / arch.area),
    "{:.3f} sec".format(latency),
    make_plot(model, input_token, output_token, data_type, arch)
  )

def make_plot(model, input_token, output_token, data_type, arch):
  # Data point candidates
  batch_list = [1, 16, 32, 48, 64, 80, 96, 112, 128, 256, 512]
  label_list = []
  data_list = []

  label = f"Input: {input_token}, Output: {output_token}, Data Type: {data_type}"
  x_axis = []
  throughput_list = []
  for batch_size in batch_list:
    # Calculate the overal throughput
    throughput, _ = lpu_throughput.throughput(model, input_token, output_token, batch_size, data_type, arch)
    if throughput!=0:
      throughput_list.append(throughput)
      x_axis.append(batch_size)
  label_list.append(label)
  data_list.append(throughput_list)
  # Convert to pandas
  dataset = pd.DataFrame(data_list, columns=x_axis, index=label_list).transpose()
  dataset.index.name = "Batch Size"
  dataset = dataset.reset_index().melt('Batch Size', var_name='category', value_name='Performance (token/sec)')
  return dataset

##########################################################################
## Gradio Page Layout
##########################################################################

def page():

  # User input for checkpoint
  with gr.Row():
    with gr.Column(variant="panel"):
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
      with gr.Row():
        # Data precision
        data_type_radio = gr.Radio(
          choices=["float32", "float16", "bfloat16", "float8", "int8", "oaken", "int4"],
          value="float16",
          label="Data Precision"
        )

    with gr.Column(variant="panel"):
      # Logic specification
      logic_spec_frame = gr.DataFrame(
        headers=["# of MAC", "# of Core", "Frequency (GHz)", "Batch", "Area (mm^2)", "Power (W)"],
        value=[[MAC, NUM_CORE, LOGIC_FREQUENCY, MAX_BATCH, CHIP_AREA, MAX_POWER]],
        datatype="number",
        label="Hardware Specification",
        row_count=(1, "fixed"),
        col_count=(6, "fixed")
      )
      # Memory specification
      memory_spec_frame = gr.DataFrame(
        headers=["Memory Type", "Channel", "Total Capacity", "Bandwidth"],
        value=[[MEMORY_TYPE, MEMORY_CHANNEL, MEMORY_CAPACITY, MEMORY_BANDWIDTH]],
        datatype=["str", "number", "number", "number"],
        label="Memory Specification",
        row_count=(1, "fixed"),
        col_count=(4, "fixed")
      )
      with gr.Row():
        sum_util_slider = gr.Slider(
          label="Summarization Util",
          minimum=0,
          maximum=1.0,
          value=0.8
        )
        gen_util_slider = gr.Slider(
          label="Generation Util",
          minimum=0,
          maximum=1.0,
          value=0.9
        )
        mem_util_slider = gr.Slider(
          label="Bandwidth Util",
          minimum=0,
          maximum=1.0,
          value=0.9
        )
      with gr.Row():
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
            data_type_radio,
            sum_util_slider,
            gen_util_slider,
            mem_util_slider
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
      with gr.Row():
        # Power efficiency
        power_eff_box = gr.Textbox(
          label="Power Efficiency"
        )
        # Area efficiency
        area_eff_box = gr.Textbox(
          label="Area Efficiency"
        )
      # Total latency
      latency_box = gr.Textbox(
        label="Total Latency"
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
      data_type_radio,
      logic_spec_frame,
      memory_spec_frame,
      sum_util_slider,
      gen_util_slider,
      mem_util_slider
    ],
    outputs=[
      performance_box,
      throughput_box,
      power_eff_box,
      area_eff_box,
      latency_box,
      throughput_plot
    ]
  )

##########################################################################
