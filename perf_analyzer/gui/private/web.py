##########################################################################
##  Main Gradio Function for Performance Analyzer
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

import os
import gradio as gr

# Web pages
from perf_analyzer.gui.private.components import kv_cache, orion_throughput, lpu_throughput, compute_op

##########################################################################
## Gradio Main
##########################################################################

if __name__ == "__main__":

  # Construct gradio GUI
  with gr.Blocks() as demo:
    # Performance Calculator
    with gr.Tab("Orion Performance Analysis"):
      orion_throughput.page()
    with gr.Tab("LPU Performance Analysis"):
      lpu_throughput.page()
    # Key-Value Calculator
    with gr.Tab("Calculate Key-Value data size"):
      kv_cache.page()
    # Operation Analysis
    with gr.Tab("Compute Operation Analysis"):
      compute_op.page()

  # Launch GUI
  demo.launch(server_name="0.0.0.0")

##########################################################################
