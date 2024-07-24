##########################################################################
##  Web GUI for Key-Value Calculator
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

import os
import matplotlib.pyplot as plt
# Gradio Library
import gradio as gr

# Model information
from perf_analyzer.models import AutoConfig, AutoModelForCausalLM
# Utilites
from perf_analyzer.utils import kv_cache

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

def get_model_info(model_id: str, batch_size: int):
  # Download HuggingFace model
  model = download_model(model_id)
  # Calculate model size
  model_size = model.n_param() * 2 / 1024 / 1024 / 1024
  # Calculate key-value cache size
  kv_cache_size = kv_cache.data_size(model.config, batch_size=batch_size)
  return (
    model.config.hidden_size,
    model.config.num_hidden_layers,
    model.config.dim_attention_heads,
    model.config.num_attention_heads,
    model.config.num_key_value_heads,
    model.config.max_length,
    model.config.sliding_window is not None,
    model.config.sliding_window,
    "{:.3f} GiB".format(model_size),
    "{:.3f} GiB".format(kv_cache_size),
    kv_cache_pie_chart(model_size, kv_cache_size)
  )

def regenerate_model_info(
    model_id: str,
    batch_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    dim_attention_heads: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    max_length: int,
    use_sliding_window: bool,
    sliding_window: int
  ):
  # Download HuggingFace model
  model = download_model(model_id)
  # Generate mimic config
  model.config.hidden_size = hidden_size
  model.config.num_hidden_layers = num_hidden_layers
  model.config.dim_attention_heads = dim_attention_heads
  model.config.num_attention_heads = num_attention_heads
  model.config.num_key_value_heads = num_key_value_heads
  model.config.max_length = max_length
  model.config.sliding_window = sliding_window if use_sliding_window else None
  # Calculate model size
  model_size = model.n_param() * 2 / 1024 / 1024 / 1024
  # Calculate key-value cache size
  kv_cache_size = kv_cache.data_size(model.config, batch_size=batch_size)
  return (
    "{:.3f} GiB".format(model_size),
    "{:.3f} GiB".format(kv_cache_size),
    kv_cache_pie_chart(model_size, kv_cache_size)
  )

def kv_cache_pie_chart(model_size, kv_cache_size):
  fig, ax = plt.subplots(figsize=(6, 3))
  wedges, texts, autotexts = ax.pie(
    [model_size, kv_cache_size],
    labels=["Model Parameters", "Key-Value Cache"],
    autopct=lambda pct: "{:.1f}%".format(pct),
    startangle=90
  )
  # Change the color of the main labels
  for text in texts:
    text.set_color('red') 
  # Change the color of the percentage labels
  for autotext in autotexts:
    autotext.set_color('black')
  fig.patch.set_facecolor('none')
  fig.patch.set_alpha(0.0)
  ax.set_facecolor('none')
  ax.patch.set_alpha(0.0)
  return fig

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
          batch_size_slider
        ],
        value="Clear",
        variant="secondary",
        scale=1
      )
  
  # Report Model Configuration
  with gr.Row(equal_height=True):
    with gr.Column(variant="panel"):
      with gr.Row():
        # Hidden size
        hidden_size_slider = gr.Slider(
          label="Size of Hidden States",
          minimum=0,
          maximum=65536,
          value=0
        )
        # Hidden layers
        hidden_layers_slider = gr.Slider(
          label="Number of Hidden Layers",
          minimum=0,
          maximum=512,
          value=0
        )
      with gr.Row():
        # The dimension of key-value heads
        dim_attention_heads_slider = gr.Slider(
          label="Dimension of Attention Heads",
          minimum=0,
          maximum=1024,
          value=0
        )
        # The number of attention heads
        num_attention_heads_slider = gr.Slider(
          label="Number of Attention Heads",
          minimum=0,
          maximum=256,
          value=0
        )
        # The number of key-value heads
        num_key_value_heads_slider = gr.Slider(
          label="Number of Key-Value Heads",
          minimum=0,
          maximum=256,
          value=0
        )
      # Maximum token length
      max_length_slider = gr.Slider(
        label="Maximum Token Length",
        minimum=0,
        maximum=65536,
        value=0
      )
      with gr.Row():
        use_sliding_window_box = gr.Checkbox(
          label="Use Sliding",
          value=False,
          scale=1
        )
        sliding_window_slider = gr.Slider(
          label="Size of Sliding Window",
          minimum=0,
          maximum=65536,
          value=0,
          scale=3
        )
      # Regenerate button
      regenerate_btn = gr.Button(
        value="Regenerate",
        variant="primary"
      )

    with gr.Column(variant="panel"):
      with gr.Row():
        # Total data size of model parameter
        model_size_box = gr.Textbox(
          label="Total Size of Model Parameters",
          placeholder="Report the size of model parameters in GiB"
        )
        # Total data size of key-value
        kv_cache_size_box = gr.Textbox(
          label="Total Size of Key-Value Cache",
          placeholder="Report the size of key-value in GiB"
        )
      with gr.Row():
        kv_cache_size_plot = gr.Plot(
          label="Percentage of Key-Value Cache"
        )
  
  # Button events
  # Model id
  model_id_box.submit(
    fn=get_model_info,
    inputs=[
      model_id_box,
      batch_size_slider
    ],
    outputs=[
      hidden_size_slider,
      hidden_layers_slider,
      dim_attention_heads_slider,
      num_attention_heads_slider,
      num_key_value_heads_slider,
      max_length_slider,
      use_sliding_window_box,
      sliding_window_slider,
      model_size_box,
      kv_cache_size_box,
      kv_cache_size_plot
    ]
  )
  # Submit button
  submit_btn.click(
    fn=get_model_info,
    inputs=[
      model_id_box,
      batch_size_slider
    ],
    outputs=[
      hidden_size_slider,
      hidden_layers_slider,
      dim_attention_heads_slider,
      num_attention_heads_slider,
      num_key_value_heads_slider,
      max_length_slider,
      use_sliding_window_box,
      sliding_window_slider,
      model_size_box,
      kv_cache_size_box,
      kv_cache_size_plot
    ]
  )
  # Re-generate button
  regenerate_btn.click(
    fn=regenerate_model_info,
    inputs=[
      model_id_box,
      batch_size_slider,
      hidden_size_slider,
      hidden_layers_slider,
      dim_attention_heads_slider,
      num_attention_heads_slider,
      num_key_value_heads_slider,
      max_length_slider,
      use_sliding_window_box,
      sliding_window_slider
    ],
    outputs=[
      model_size_box,
      kv_cache_size_box,
      kv_cache_size_plot
    ]
  )

##########################################################################