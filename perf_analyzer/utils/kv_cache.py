##########################################################################
##  Key-Value Calculator
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

from perf_analyzer.models import AutoConfig

##########################################################################
## Defines
##########################################################################

# Data precision
FLOAT16   = 2
# Data size
BYTE      = 1
KILO_BYTE = BYTE * 1024
MEGA_BYTE = KILO_BYTE * 1024
GIGA_BYTE = MEGA_BYTE * 1024

##########################################################################
## Function
##########################################################################

# Calculate Key-Value data size
def data_size(config: AutoConfig, position: int=None, byte_size: int=FLOAT16, batch_size: int=1) -> int:
  # If the specific position is not given
  if position is None:
    position = config.max_length
  # If the model uses sliding window
  if config.sliding_window is not None:
    position = position if position < config.sliding_window else config.sliding_window
    # Dimension for key-value
    kv_width = config.dim_attention_heads * config.num_key_value_heads
    kv_height = position
    kv_layers = config.num_hidden_layers
    kv_dtype = byte_size
  else:
    position = position # if position < config.max_length else config.max_length
    # Dimension for key-value
    kv_width = config.dim_attention_heads * config.num_key_value_heads
    kv_height = position
    kv_layers = config.num_hidden_layers
    kv_dtype = byte_size
  # Calculate data size
  return batch_size * kv_height * kv_width * 2 * kv_layers * kv_dtype / GIGA_BYTE

##########################################################################