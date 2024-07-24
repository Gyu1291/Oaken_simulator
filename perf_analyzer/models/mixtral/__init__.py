##########################################################################
##  Modeling Mixtral
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

from perf_analyzer.models.common.config import BaseConfig
from perf_analyzer.models.common.modeling import (
  TokenEmbedding,
  PositionalEmbedding,
  LayerNorm,
  RMSNorm,
  Attention,
  MLP,
  MoE,
  LMHead
)

##########################################################################
## Configuration
##########################################################################

class MixtralConfig(BaseConfig):

  # Constructor
  def __init__(self) -> None:
    super().__init__()
    # Configrue for this model
    self.config_table = {
      "architecture"          : "architectures",
      "bos_token_id"          : "bos_token_id",
      "eos_token_id"          : "eos_token_id",
      "hidden_act"            : "hidden_act",
      "hidden_size"           : "hidden_size",
      "intermediate_size"     : "intermediate_size",
      "max_length"            : "max_position_embeddings",
      "num_attention_heads"   : "num_attention_heads",
      "num_key_value_heads"   : "num_key_value_heads",
      "num_hidden_layers"     : "num_hidden_layers",
      "vocab_size"            : "vocab_size",
      "sliding_window"        : "sliding_window",
      "num_experts_per_tok"   : "num_experts_per_tok",
      "num_local_experts"     : "num_local_experts"
    }

##########################################################################
## Model
##########################################################################

class MixtralForCausalLM():
  # Constructor
  def __init__(self, config: MixtralConfig):
    # Config
    self.config = config
    # Construct model
    self.tok_embed = TokenEmbedding(config.hidden_size, config.vocab_size)
    self.layers = list(
      [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
    )
    self.ln_f = RMSNorm(config.hidden_size)
    self.lm_head = LMHead(config.hidden_size, config.vocab_size, use_table=False, use_bias=False)

  def set_position(self, position):
    self.position = position
    for layer in self.layers:
      layer.set_position(position)

  # Get the number of operations
  def n_op(self, position: int=1):
    # Initialize op
    op = 0
    # Accumulate operations
    op = op + self.tok_embed.n_op()       # Token embedding
    op = op + self.config.hidden_size     # Add
    for layer in self.layers:             # Decoder layer
      op = op + layer.n_op(position)
    op = op + self.ln_f.n_op()            # Final LayerNorm
    op = op + self.lm_head.n_op()         # LM-Head
    return op

  # Get the number of parameters
  def n_param(self):
    # Initialize size
    size = 0
    # Accumulate parameters
    size = size + self.tok_embed.n_param()
    for layer in self.layers:
      size = size + layer.n_param()
    size = size + self.ln_f.n_param()
    size = size + self.lm_head.n_param()
    return size

class MixtralDecoderLayer():
  # Constructor
  def __init__(self, config: MixtralConfig, layer_idx: int):
    # Config
    self.config = config
    self.layer_idx = layer_idx
    # Construct model
    self.ln_1 = RMSNorm(config.hidden_size)
    self.attn = Attention(
      config.hidden_size, 
      config.dim_attention_heads,
      config.num_attention_heads,
      config.num_key_value_heads,
      use_bias=False
    )
    self.ln_2 = RMSNorm(config.hidden_size)
    self.moe = MoE(
      config.hidden_size,
      config.intermediate_size,
      config.hidden_act,
      config.num_experts_per_tok,
      config.num_local_experts,
      use_bias=False
    )
    self.layers = []
    self.init_layers()



  def init_layers(self):
    self.layers.append(self.ln_1)
    self.layers.append(self.attn)
    self.layers.append(self.ln_2)
    self.layers.append(self.moe)

  def set_position(self, position):
    self.position = position
    self.attn.set_position(position)

  # Get the number of operations
  def n_op(self, position: int=1):
    # Initialize op
    op = 0
    # Acumulate operations
    op = op + self.ln_1.n_op()          # Pre-LayerNorm
    op = op + self.attn.n_op(position)  # Self-Attention
    op = op + self.config.hidden_size   # Residual
    op = op + self.ln_2.n_op()          # Post-LayerNorm
    op = op + self.moe.n_op()           # MLP
    op = op + self.config.hidden_size   # Residual
    return op

  # Get the number of parameters
  def n_param(self):
    # Initialize size
    size = 0
    # Accumulate parameters
    size = size + self.ln_1.n_param()
    size = size + self.attn.n_param()
    size = size + self.ln_2.n_param()
    size = size + self.moe.n_param()
    return size

##########################################################################