##########################################################################
##  Main Gradio Function for Performance Analyzer
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

import os
import argparse
import gradio as gr

# Web pages
from perf_analyzer.gui.public.components import orion_throughput, lpu_throughput

##########################################################################
## Argument Parser
##########################################################################

def parse_args():
  parser = argparse.ArgumentParser(
    description="Web GUI for Performance Analysis"
  )
  # Web-server arguments
  parser.add_argument("--host", type=str, default="127.0.0.1",
                      help="host name")
  parser.add_argument("--port", type=int, default=7860,
                      help="port number")
  return parser.parse_args()

##########################################################################
## Authentication
##########################################################################

auth = [
  ("hyperaccel", "*hyper123"),
  ("guest", "*guest123")
]

##########################################################################
## Gradio Main
##########################################################################

if __name__ == "__main__":

  # Argumetns
  args = parse_args()

  # Construct gradio GUI
  with gr.Blocks() as demo:
    # Performance Calculator
    with gr.Tab("Orion Performance Analysis"):
      orion_throughput.page()
    with gr.Tab("LPU Performance Analysis"):
      lpu_throughput.page()

  # Launch GUI
  demo.launch(server_name=args.host, server_port=args.port, auth=auth)

##########################################################################