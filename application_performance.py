##########################################################################
##  Web GUI for LPU Performance Profiling
##
##  Authors:  Soongyu Choi   ( soongyu1291@kaist.ac.kr    )
##  Version:  Oaken
##  Date:     2024-07-25      ( Oaken, init                )
##
##########################################################################

import pandas as pd
# Model information
# import sys
# sys.path.append('../perf_analyzer')
from perf_analyzer.models import AutoConfig, AutoModelForCausalLM
# Utilites
from perf_analyzer.utils import lpu_throughput
from perf_analyzer.utils.lpu_throughput import HardwareSpec

##########################################################################
## Defines
##########################################################################

# Architecture Configurations
MAC               = 32*32*2       # (Vector Dimension) x (Vector Lane) x 2
NUM_CORE          = 256        # 2D-array
MAX_BATCH         = 256            # Maximum supported Batch
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
    print("Error")
    return False

# Calculate throughput
def throughput(model_id, data_type):
  # Hardware configuration
  arch = HardwareSpec(
    mac_per_core = MAC,
    num_core = NUM_CORE,
    frequency = LOGIC_FREQUENCY,
    max_batch=MAX_BATCH,
    sum_util=1,
    gen_util=1,
    mem_util=1,
    area=CHIP_AREA,
    power=MAX_POWER,
    memory=MEMORY_TYPE,
    capacity=MEMORY_CAPACITY,
    bandwidth=MEMORY_BANDWIDTH
  )
  # Download from HuggingFace
  model = download_model(model_id)
  # Calculate peak performance
  throughput = avg_application_performance(model, model_id, data_type, arch)
  # print(throughput)
  return throughput


def avg_application_performance(model, model_id, data_type, arch):
  result = {}
  application_type = ["code", "conv"]
  batch_list = [64, 128, 256]
  for ap_type in application_type:
    for batch_size in batch_list:
      throughputs_per_batch = []
      for i in range(5):
        filename = f"batch_sample/{ap_type}_batch_batch_{batch_size}_{i}.csv"
        df = pd.read_csv(filename, header=0)
        input_token = df.iloc[:, 0]
        output_token = df.iloc[:, 1]
        token_pair = list(zip(input_token, output_token))
        temp_throughputs = []

        for token in token_pair:
            throughput, _ = lpu_throughput.throughput(model, token[0], token[1], batch_size, data_type, arch)
            if throughput == 0:
              print("OOM detected")
              print(f"Current Batch size: {batch_size}")
              #exit(0)
            temp_throughputs.append(throughput)
        avg_throughputs = sum(temp_throughputs) / len(temp_throughputs)
      throughputs_per_batch.append(avg_throughputs)
      tp = sum(throughputs_per_batch) / len(throughputs_per_batch)
      result[f"{ap_type}_batch_{batch_size}_{model_id}"] = tp

  return result
    
if __name__ == "__main__":
  model_id = "mistralai/Mixtral-8x7B-v0.1"
  data_type = "float16"
  throughput(model_id, data_type)