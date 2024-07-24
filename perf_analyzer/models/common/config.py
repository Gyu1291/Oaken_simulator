##########################################################################
##  Base Model Configuration
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

from transformers import AutoConfig

##########################################################################
## Class
##########################################################################

class BaseConfig:

  # Constructor
  def __init__(self) -> None:
    
    ''' Model Configuration '''
    self.architecture         = None
    self.bos_token_id         = None
    self.eos_token_id         = None
    self.hidden_act           = None
    self.hidden_size          = None
    self.intermediate_size    = None
    self.max_length           = None
    self.sliding_window       = None
    self.dim_attention_heads  = None
    self.num_attention_heads  = None
    self.num_key_value_heads  = None
    self.num_hidden_layers    = None
    self.norm_eps             = None
    self.vocab_size           = None

    # Convert table
    self.config_table = {
      "architecture"          : "architectures",
      "bos_token_id"          : "bos_token_id",
      "eos_token_id"          : "eos_token_id",
      "hidden_act"            : "hidden_act",
      "hidden_size"           : "hidden_size",
      "intermediate_size"     : "intermediate_size",
      "max_length"            : "max_length",
      "sliding_window"        : "sliding_window",
      "dim_attention_heads"   : "dim_attention_heads",
      "num_attention_heads"   : "num_attention_heads",
      "num_key_value_heads"   : "num_key_value_heads",
      "num_hidden_layers"     : "num_hidden_layers",
      "vocab_size"            : "vocab_size",
      "multi_query"           : "multi_query",
      "num_experts_per_tok"   : "num_experts_per_tok",
      "num_local_experts"     : "num_local_experts"
    }

  @classmethod
  def from_pretrained(cls, ckpt: AutoConfig):
    # Instance config object
    old_config = ckpt
    new_config = cls()
    
    # Load parameters
    new_config.architecture         = getattr(old_config, new_config.config_table["architecture"])[0]
    new_config.bos_token_id         = getattr(old_config, new_config.config_table["bos_token_id"])
    new_config.eos_token_id         = getattr(old_config, new_config.config_table["eos_token_id"])
    new_config.hidden_size          = getattr(old_config, new_config.config_table["hidden_size"])
    new_config.num_attention_heads  = getattr(old_config, new_config.config_table["num_attention_heads"])
    new_config.num_hidden_layers    = getattr(old_config, new_config.config_table["num_hidden_layers"])
    new_config.vocab_size           = getattr(old_config, new_config.config_table["vocab_size"])
    
    # For hidden activation
    if "hidden_act" in new_config.config_table.keys():
      new_config.hidden_act = getattr(old_config, new_config.config_table["hidden_act"])
    else:
      new_config.hidden_act = "gelu"

    # For specific head dimension
    if "dim_attention_heads" in new_config.config_table.keys():
      new_config.dim_attention_heads = getattr(old_config, new_config.config_table["dim_attention_heads"])
    else:
      new_config.dim_attention_heads = new_config.hidden_size // new_config.num_attention_heads
    
    # For specific key-value heads
    if "num_key_value_heads" in new_config.config_table.keys():
      # Multi/Group-query attention
      new_config.num_key_value_heads = getattr(old_config, new_config.config_table["num_key_value_heads"])
    else:
      new_config.num_key_value_heads = new_config.num_attention_heads

    # Or, multi-query flags
    if "multi_query" in new_config.config_table.keys():
      if getattr(old_config, new_config.config_table["multi_query"]):
        new_config.num_key_value_heads = 1

    # For intermediate size
    if "intermediate_size" in new_config.config_table.keys():
      if isinstance(new_config.config_table["intermediate_size"], int):
        new_config.intermediate_size = new_config.config_table["intermediate_size"]
      else:
        new_config.intermediate_size = getattr(old_config, new_config.config_table["intermediate_size"])
    else:
      new_config.intermediate_size = new_config.hidden_size * 4

    # For max length
    if isinstance(new_config.config_table["max_length"], int):
      new_config.max_length = new_config.config_table["max_length"]
    else:
      new_config.max_length = getattr(old_config, new_config.config_table["max_length"])

    # For sliding window
    if "sliding_window" in new_config.config_table.keys():
      new_config.sliding_window = getattr(old_config, new_config.config_table["sliding_window"])
    else:
      new_config.sliding_window = None

    # For Mixure-of-Experts model
    if "num_experts_per_tok" in new_config.config_table.keys():
      new_config.num_experts_per_tok = getattr(old_config, new_config.config_table["num_experts_per_tok"])
    if "num_local_experts" in new_config.config_table.keys():
      new_config.num_local_experts = getattr(old_config, new_config.config_table["num_local_experts"])

    return new_config

##########################################################################