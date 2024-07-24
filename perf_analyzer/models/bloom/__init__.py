##########################################################################
##  Modeling Bloom
##
##  Authors:  Gyubin    Choi  ( gb.choi@hyperaccel.ai       )
##  Version:  1.3.1
##  Date:     2024-03-06      ( v1.3.1, init                )
##
##########################################################################

from perf_analyzer.models.common.config import BaseConfig
from perf_analyzer.models.common.modeling import (
  TokenEmbedding,
  LayerNorm,
  Attention,
  MLP,
  LMHead
)
from anytree import NodeMixin, LevelOrderIter, RenderTree

##########################################################################
## Configuration
##########################################################################

class BloomConfig(BaseConfig):

  # Constructor
  def __init__(self) -> None:
    super().__init__()
    # Configure for this model
    self.config_table = {
      "architecture"          : "architectures",
      "bos_token_id"          : "bos_token_id",
      "eos_token_id"          : "eos_token_id",
      "hidden_size"           : "hidden_size",
      "num_attention_heads"   : "num_attention_heads",
      "num_hidden_layers"     : "n_layer",
      "vocab_size"            : "vocab_size",
      "max_length"            : 2048
    }
        
##########################################################################
## Model
##########################################################################

class BloomForCausalLM(NodeMixin):
  # Constructor
  def __init__(self, config: BloomConfig):
    super(BloomForCausalLM, self).__init__()
    # Config
    self.config = config
    self.name = self.__class__.__name__
    # Construct model
    self.tok_embed = TokenEmbedding(config.hidden_size, config.vocab_size, parent=self)
    self.layers = list(
      [BloomDecoderLayer(config, layer_idx, parent=self) for layer_idx in range(config.num_hidden_layers)]
    )
    self.ln_f = LayerNorm(config.hidden_size, parent=self)
    self.lm_head = LMHead(config.hidden_size, config.vocab_size, use_bias=False, parent=self)
    # Setup token position
    self.set_position(self.config.max_length)

  # Set the token position
  def set_position(self, position):
    self.position = position
    for layer in self.layers:
      layer.set_position(position)

  def n_op(self):
    # Initialize op
    op = 0
    # Accumulate operations
    op = op + self.tok_embed.n_op()       # Token embedding
    op = op + self.config.hidden_size     # Add
    for layer in self.layers:             # Decoder layer
      op = op + layer.n_op()
    op = op + self.ln_f.n_op()            # Final LayerNorm
    op = op + self.lm_head.n_op()         # LM-Head
    return op

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
  
class BloomDecoderLayer(NodeMixin):
  # Constructor
  def __init__(self, config:BloomConfig, layer_idx: int, name=None, parent=None):
    super(BloomDecoderLayer, self).__init__()
    # Config
    self.config = config
    self.layer_idx = layer_idx
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent
    # Construct model
    self.ln_1 = LayerNorm(config.hidden_size, parent=self)
    self.attn = Attention(
      config.hidden_size, 
      config.dim_attention_heads,
      config.num_attention_heads,
      config.num_key_value_heads,
      use_bias=True,
      parent=self
    )
    self.ln_2 = LayerNorm(config.hidden_size)
    self.mlp = MLP(
      config.hidden_size,
      config.intermediate_size,
      config.hidden_act,
      use_bias=True,
      use_gate=False,
      parent=self
    )

  # Set the token position
  def set_position(self, position):
    self.position = position
    self.attn.set_position(position)

    # Get the number of operations
  def n_op(self):
    # Initialize op
    op = 0
    # Acumulate operations
    op = op + self.ln_1.n_op()          # Pre-LayerNorm
    op = op + self.attn.n_op()          # Self-Attention
    op = op + self.config.hidden_size   # Residual
    op = op + self.ln_2.n_op()          # Post-LayerNorm
    op = op + self.mlp.n_op()           # MLP
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
    size = size + self.mlp.n_param()
    return size

##########################################################################