##########################################################################
##  Web GUI for Compute Operation
##
##  Authors:  Gyubin    Choi  ( gb.choi@hyperaccel.ai       )
##  Version:  1.3.1
##  Date:     2024-03-06      ( v1.3.1, init                )
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
from perf_analyzer.utils import compute_op

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

def get_model_operation(model_id: str, input_token: int, output_token: int, batch_size: int):
  # Download HuggingFace model
  model = download_model(model_id)
  # Calculate model size
  op_nodes = compute_op.aggregate_op_nodes(model, input_token + output_token - 1)
  module_nodes = compute_op.aggregate_module_nodes(model, input_token + output_token - 1)
  return (
    staged_op_pie_chart(module_nodes),
    staged_op_pie_chart(op_nodes)
  )

def staged_op_pie_chart(staged_op_data):
  # Convert your data into a DataFrame
  data = pd.DataFrame({
    'Category': [f'{key}:{value}' for key, value in staged_op_data.items()],
    'Values': list(staged_op_data.values())
  })
  # Create the base of the chart
  base = alt.Chart(data).encode(
    theta=alt.Theta(field="Values", type="quantitative", stack=True),
    color=alt.Color(field="Category", type="nominal"),
    tooltip=['Category', 'Values']  # Tooltips on hover
  )

  # Create the pie chart
  pie = base.mark_arc(outerRadius=120)
  # Add tooltips for interactive exploration
  pie = pie.add_selection(
    alt.selection_single(fields=['Category'], bind='legend')
  )
  
  text = base.mark_text(radius=120, size=10).encode(
    text="Category",
    color=alt.value('black')
  )
  return pie + text
  
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
      # Scalability report
      staged_op_pie_chart_2 = gr.Plot(
        label="Compute Operation"
      )
#   with gr.Row(equal_height=True):
    with gr.Column(variant="panel"):
      # Scalability report
      staged_op_pie_chart_3 = gr.Plot(
        label="Detailed Compute Operation"
      )

  # Button events
  # Submit button
  submit_btn.click(
    fn=get_model_operation,
    inputs=[
      model_id_box,
      input_token_slider,
      output_token_slider,
      batch_size_slider
    ],
    outputs=[
      staged_op_pie_chart_2,
      staged_op_pie_chart_3
    ]
  )

##########################################################################
