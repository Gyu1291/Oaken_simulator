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

# Model information
from perf_analyzer.models import AutoConfig, AutoModelForCausalLM
# Utilites
from perf_analyzer.utils import kv_cache

##########################################################################
## Class
##########################################################################

class HardwareSpec:
  # Constructor
  def __init__(
    self,
    # Architecture
    mac_per_core    : int   = 32*32*2,     # (Vector Dimension) x (Vector Lane) x 2
    num_core        : int   = 128,      # 8 x 16, 2D-array
    frequency       : float = 1,      # GHz
    max_batch       : int   = 128,       # Maximum support batch size
    # Utilization
    sum_util        : float = 0.8,      # 0 ~ 1
    gen_util        : float = 0.8,      # 0 ~ 1
    mem_util        : float = 0.8,      # 0 ~ 1
    # Specification
    area            : int   = 190,      # mm^2
    power           : int   = 120,      # W
    # Memory
    memory          : str   = "LPDDR5X",
    capacity        : int   = 48,      # GB
    bandwidth       : float = 696       # GB/s
  ):

    self.mac_per_core = mac_per_core
    self.num_core = num_core
    self.frequency = frequency
    self.max_batch = max_batch
    self.sum_util = sum_util
    self.gen_util = gen_util
    self.mem_util = mem_util
    self.area = area
    self.power = power
    self.memory = memory
    self.capacity = capacity
    self.bandwidth = bandwidth

##########################################################################
## Function
##########################################################################

def oaken_kv_size(model, token_length, data_type: str, batch_size: int):
  if "oaken" in data_type:
    byte_size = 0.5
  elif "w4a16" in data_type:
    byte_size = 2
  else:
    byte_size = get_word_size(data_type)
  kv_size = kv_cache.data_size(model.config, token_length, byte_size, batch_size)
  return kv_size

def check_memory_capacity(model, token_length, data_type: str, batch_size: int, arch: HardwareSpec):
  model_size = model.n_param() * 2 / 1024 / 1024 / 1024
  kv_size = oaken_kv_size(model, token_length, data_type, batch_size)

  if arch.capacity < model_size + kv_size:
    return False, model_size, kv_size
  else:
    return True, model_size, kv_size
  
# Get the word size
def get_word_size(data_type):
  if data_type == "float32":
    return 4
  elif data_type == "w4a16":
    return 0.5
  elif data_type == "float16":
    return 2
  elif data_type == "bfloat16":
    return 2
  elif data_type == "float8":
    return 1
  elif data_type == "int8":
    return 1
  elif data_type == "int4":
    return 0.5
  elif data_type == "oaken":
    return 2
  elif data_type == "oaken_w4":
    return 0.5
  else:
    return 2

# Calculate batch utilization
def batch_util(model, position):
  # return (1 - BATCH_ALPHA_UTIL) + (BATCH_ALPHA_UTIL / batch_size)
  model_size = model.n_param() * 2 / 1024 / 1024 / 1024
  kv_size = kv_cache.data_size(model.config, position=position)
  return model_size / (model_size + kv_size)

# Calculate peak performance
def peak_performance(data_type, arch: HardwareSpec):
  return arch.mac_per_core * arch.num_core * arch.frequency * (2 / get_word_size(data_type)) / 1000

def latency_per_operation(layer_name, operation, num_token, batch_size, data_type, kv_latency, arch: HardwareSpec, stage):
  # Utilization Configuration
  # Amount of data feeded from memory per cycle
  Amount_per_cycle = arch.bandwidth / arch.frequency
  Element_per_cycle_from_Memory = Amount_per_cycle // get_word_size(data_type) #Number of weight elements comes form memory

  # Sharing : number of cores that can share the broadcasted data from memory
  Sharing = 1
  if stage == 0 :
    Sharing = min(num_token, arch.num_core)
  
  elif stage == 1 :
    if (layer_name == "Attention") :
      Sharing = min(1, arch.num_core)
    else :
      Sharing = min(batch_size, arch.num_core)
  
  Maximum_Elements_Core = (arch.mac_per_core // 2) #maximum element per core
  if stage==0:
    Maximum_Elements_MAC = min(num_token, arch.num_core) * Maximum_Elements_Core #maximum element per mac
  else:
    if layer_name == "Attention":
      Maximum_Elements_MAC = Maximum_Elements_Core #maximum element per mac
    else:
      Maximum_Elements_MAC = min(batch_size, arch.num_core) * Maximum_Elements_Core #maximum element per mac
  Element_per_cycle_from_Memory *= Sharing
  Distribute_Utilization = Element_per_cycle_from_Memory/Maximum_Elements_MAC if Element_per_cycle_from_Memory/Maximum_Elements_MAC < 1 else 1
  # Compute_per_cycle : number of elements that can be processed in one cycle in total MACs
  #   Summarization Stage -> total_num_core * MAC/Core
  #   Generation Stage    -> 1 Core * MAC/Core

  if stage == 0:
    Compute_per_cycle = Distribute_Utilization * Maximum_Elements_Core
  else:
    #Token 길이가 얼마나 길든 generation stage에서 1 토큰 생성할 때 1코어 사용
    #operation.n_op() 값은 현재 1토큰을 생성할 때 필요한 operation 수
    #따라서 1코어의 계산능력을 기준으로 나눠주어야 올바른 Cycle이 나옴
    Compute_per_cycle = Distribute_Utilization * Maximum_Elements_Core
  cycles = operation.n_op() / Compute_per_cycle
  Latency = (cycles / arch.frequency) * 1e-9
  return Latency


def latency_of_sorting(vector_size, bitwidth, rw_count, bandwidth, frequency):

  log2 = math.log2(vector_size)
  term = ((log2 ** 2) + log2) / 2
  xfer_size = vector_size * bitwidth * term * rw_count
  cycle = xfer_size // bandwidth
  latency = cycle * (1 / frequency) * 1e-9

  return latency

def latency_per_layer(layer, input_token, batch_size, data_type, kv_latency, arch: HardwareSpec, stage):
  #temporary utilization for each layer
  latency = 0
  for operation in layer.operations:
    # operation = [nn.Embedding, nn.Mean, nn.StdDev, nn.Softmax, nn.Linear, nn.Matmul, nn.Elementwise]
    # Calculate the latency for each operation
    if stage==1 and layer.name == "Attention" :
      # Attention : for batch in batch_size
      latency += latency_per_operation(layer.name, operation, input_token, batch_size, data_type, kv_latency, arch, stage)

    elif operation.name == "Sorting" : 
      latency += latency_of_sorting(operation.vector_size, 2, 2, arch.bandwidth, arch.frequency)

    else :
      latency += latency_per_operation(layer.name, operation, input_token, batch_size, data_type, kv_latency, arch, stage)

  return latency

# Calculate throughput
def throughput(model, input_token, output_token, batch_size, data_type, arch: HardwareSpec):
  has_enough_memory, model_size, kv_size = check_memory_capacity(model, (input_token+output_token-1), data_type,  batch_size, arch)
  # print(f'Weight size: {model_size} GB, Key value_size: {kv_size} GB, Total size: {(model_size+kv_size)*1024} MiB, Ratio: {model_size/(model_size+kv_size)}: {kv_size/(model_size+kv_size)}')
  if has_enough_memory==False:
    return 0, 0
  
  current_position = input_token
  summarization_latency = 0
  generation_latency = 0
  sum_attention_latency = 0
  #summarizatino stage latency
  model.set_position(current_position)
  kv_latency = oaken_kv_size(model, current_position, data_type, 1)/arch.bandwidth/2
  summarization_latency += latency_per_layer(model.tok_embed, 1, batch_size, data_type, kv_latency,  arch, 0)
  for decoder_layer in model.layers:
    for sub_layer in decoder_layer.layers:
      temp_latency = latency_per_layer(sub_layer, current_position, batch_size, data_type, kv_latency,  arch, 0)
      summarization_latency += temp_latency
      if sub_layer.name == "Attention":
        sum_attention_latency += temp_latency
  summarization_latency += latency_per_layer(model.ln_f, current_position, batch_size, data_type, kv_latency,  arch, 0)
  summarization_latency += latency_per_layer(model.lm_head, current_position, batch_size, data_type, kv_latency,  arch, 0)

  #summarizatino stage total latency(token parallel scheduling)
  summarization_latency = summarization_latency * math.ceil(input_token/arch.num_core) * batch_size
  sum_attention_latency = sum_attention_latency * math.ceil(input_token/arch.num_core) * batch_size
  #generation stage latency
  current_position += output_token/2

  #batch parallel scheduling
  gen_attention_latency = 0
  #for _ in range(output_token-1):
  model.set_position(current_position)
  kv_latency = oaken_kv_size(model, current_position, data_type, 1)/arch.bandwidth/2

  generation_latency += latency_per_layer(model.tok_embed, 1, batch_size, data_type, kv_latency, arch, 1)
  #KV IO latency for batch
  generation_latency += kv_latency*(2*min(batch_size, arch.num_core)-1)
  gen_attention_latency += kv_latency*(2*min(batch_size, arch.num_core)-1)
  for decoder_layer in model.layers:
    for sub_layer in decoder_layer.layers:
      temp_latency = latency_per_layer(sub_layer, current_position, batch_size, data_type, kv_latency, arch, 1)
      generation_latency += temp_latency
      if sub_layer.name == "Attention":
        gen_attention_latency += temp_latency
  generation_latency += latency_per_layer(model.ln_f, current_position, batch_size, data_type, kv_latency, arch, 1)
  generation_latency += latency_per_layer(model.lm_head, current_position, batch_size, data_type, kv_latency, arch, 1)
  #current_position += 1

  generation_latency = generation_latency * math.ceil(batch_size/arch.num_core) * (output_token-1)
  gen_attention_latency = gen_attention_latency * math.ceil(batch_size/arch.num_core) * (output_token-1)
  total_latency = summarization_latency + generation_latency
  throughput = (output_token * batch_size) / total_latency
  attention_latency = sum_attention_latency + gen_attention_latency
  #print('Batch Size : {} Output Token : {} Input Token : {}'.format(batch_size, output_token, input_token))
  # print('Summarization Latency : {:.3f} Generation Latency : {:.3f} Total Latency : {:.3f}'.format(summarization_latency, generation_latency, total_latency))
  #print('throughput : {:.2f} token/sec'.format(throughput))
  # print(f'generation_latecny: {generation_latency}, kv_latency : {kv_latency*(2*min(batch_size, arch.num_core)-1)* math.ceil(batch_size/arch.num_core) * (output_token-1)}')
  #print(f'attention_latency: {attention_latency}, non_attention_latency: {total_latency - attention_latency}')
  return throughput, total_latency

##########################################################################
