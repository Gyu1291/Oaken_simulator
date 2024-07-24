##########################################################################
##  Modeling HuggingFace CausalLM
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

from perf_analyzer.models.common import nn
from anytree import NodeMixin

##########################################################################
## Class
##########################################################################

''' Embeddings '''
# Token Embedding
class TokenEmbedding(NodeMixin):
  # Constructor
  def __init__(self, hidden_size: int, vocab_size: int, name=None, parent=None):
    super(TokenEmbedding, self).__init__()
    # Option
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent
    # Weight
    self.wte = nn.Embedding(hidden_size, vocab_size, parent=self)
    self.operations = []
    self.init_operations()
    
  def init_operations(self):
    self.operations.append(self.wte)
    return

  # Get the number of operations
  def n_op(self):
    # Initialize op
    op = 0
    # Accumulate operations
    op = op + self.wte.n_op()
    return op
  
  def n_cycle(self):
    cycle = 0
    cycle = cycle + self.wte.n_cycle()
    return cycle
  
  # Get the number of parameters
  def n_param(self):
    # Initialize size
    size = 0
    # Accumulate parameters
    size = size + self.wte.n_param()
    return size

# Positional Embedding
class PositionalEmbedding(NodeMixin):
  # Constructor
  def __init__(self, hidden_size: int, max_length: int, name=None, parent=None):
    super(PositionalEmbedding, self).__init__()
    # Option
    self.hidden_size = hidden_size
    self.max_length = max_length
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent
    # Weight
    self.wpe = nn.Embedding(hidden_size, max_length, parent=self)
    self.operations = []
    self.init_operations()

  def init_operations(self):
    self.operations.append(self.wpe)
    return
  
  # Get the number of operations
  def n_op(self):
    # Initialize op
    op = 0
    # Accumulate operations
    op = op + self.wpe.n_op()
    return op
  
  def n_cycle(self):
    cycle = 0
    cycle = cycle + self.wpe.n_cycle()
    return cycle

  # Get the number of parameters
  def n_param(self):
    # Initialize size
    size = 0
    # Accumulate parameters
    size = size + self.wpe.n_param()
    return size

''' Normalization '''
# Layer Normalization
class LayerNorm(NodeMixin):
  # Constructor
  def __init__(self, hidden_size: int, name=None, parent=None):
    super(LayerNorm, self).__init__()
    # Options
    self.hidden_size = hidden_size
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent
    # Operation
    self.mean = nn.Mean(dim=hidden_size, parent=self)
    self.stddev = nn.StdDev(dim=hidden_size, parent=self)
    self.sub = nn.Elementwise(dim=hidden_size, parent=self)
    self.div = nn.Elementwise(dim=hidden_size, parent=self)
    self.mul = nn.Elementwise(dim=hidden_size, parent=self)
    self.add = nn.Elementwise(dim=hidden_size, parent=self)
    # Weight
    self.g = nn.Parameter(dim=[hidden_size])
    self.b = nn.Parameter(dim=[hidden_size])
    self.operations = []
    self.init_operations()

  def init_operations(self):
    self.operations.append(self.mean)
    self.operations.append(self.stddev)
    self.operations.append(self.sub)
    self.operations.append(self.div)
    self.operations.append(self.mul)
    self.operations.append(self.add)
    return    
  
  # Get the number of operations
  def n_op(self):
    # Initialize op
    op = 0
    # Accumulate operations
    op = op + self.mean.n_op()      # Mean
    op = op + self.stddev.n_op()    # Stddev
    op = op + self.sub.n_op()       # Sub
    op = op + self.div.n_op()       # Div
    op = op + self.mul.n_op()       # Mul
    op = op + self.add.n_op()       # Add
    return op

  def n_cycle(self):
    # Initialize op
    cycle = 0
    # Accumulate operations
    cycle = cycle + self.mean.n_cycle()      # Mean
    cycle = cycle + self.stddev.n_cycle()    # Stddev
    cycle = cycle + self.sub.n_cycle()       # Sub
    cycle = cycle + self.div.n_cycle()       # Div
    cycle = cycle + self.mul.n_cycle()       # Mul
    cycle = cycle + self.add.n_cycle()       # Add
    return cycle


  # Get the number of parameters
  def n_param(self):
    # Initialize size
    size = 0
    # Accumulate parameters
    size = size + self.g.n_param()
    size = size + self.b.n_param()
    return size

# Root-Mean-Square (RMS) Normalization
class RMSNorm(NodeMixin):
  # Constructor
  def __init__(self, hidden_size: int, name=None, parent=None):
    super(RMSNorm, self).__init__()
    # Options
    self.hidden_size = hidden_size
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent
    # Operation
    self.stddev = nn.StdDev(dim=hidden_size, parent=self)
    self.div = nn.Elementwise(dim=hidden_size, parent=self)
    self.mul = nn.Elementwise(dim=hidden_size, parent=self)
    # Weight
    self.w = nn.Parameter(dim=[hidden_size])
    self.operations = []
    self.init_operations()

  def init_operations(self):
    self.operations.append(self.stddev)
    self.operations.append(self.div)
    self.operations.append(self.mul)
    return
    
  # Get the number of operations
  def n_op(self):
    # Initialize op
    op = 0
    # Accumulate operations
    op = op + self.stddev.n_op()    # Stddev
    op = op + self.div.n_op()       # Div
    op = op + self.mul.n_op()       # Mul
    return op
  
  def n_cycle(self):
    # Initialize cycle
    cycle = 0
    # Accumulate cycle
    cycle = cycle + self.stddev.n_cycle()    # Stddev
    cycle = cycle + self.div.n_cycle()       # Div
    cycle = cycle + self.mul.n_cycle()       # Mul
    return cycle

  # Get the number of parameters
  def n_param(self):
    # Initialize size
    size = 0
    # Accumulate parameters
    size = size + self.w.n_param()
    return size

''' Self-Attention '''
class Attention(NodeMixin):
  # Constructor
  def __init__(self, hidden_size: int, dim_attention_heads: int, num_attention_heads: int, num_key_value_heads: int, use_bias: bool=False, name=None, parent=None):
    super(Attention, self).__init__()
    # Options
    self.hidden_size = hidden_size
    self.dim_attention_heads = dim_attention_heads
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.use_bias = use_bias
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent
    # Operation
    self.softmax = [nn.Softmax(dim=1, parent=self) for _ in range(self.num_attention_heads)]
    self.query_key = [nn.Matmul(self.dim_attention_heads, 1, parent=self) for _ in range(self.num_attention_heads)]
    self.score_value = [nn.Matmul(1, self.dim_attention_heads, parent=self) for _ in range(self.num_attention_heads)]
    # Weight
    self.k_proj = nn.Linear(hidden_size, dim_attention_heads * num_key_value_heads, use_bias, parent=self)
    self.q_proj = nn.Linear(hidden_size, dim_attention_heads * num_attention_heads, use_bias, parent=self)
    self.v_proj = nn.Linear(hidden_size, dim_attention_heads * num_key_value_heads, use_bias, parent=self)
    self.o_proj = nn.Linear(dim_attention_heads * num_attention_heads, hidden_size, use_bias, parent=self)
    self.operations = []
    self.init_operations()

  def init_operations(self):
    operations = []
    operations.append(self.q_proj)
    operations.append(self.k_proj)
    operations.append(self.v_proj)
    for head in range(self.num_attention_heads):
      operations.append(self.query_key[head])
      operations.append(self.softmax[head])
      operations.append(self.score_value[head])
    operations.append(self.o_proj)
    self.operations = operations
    return

  # Set the token position
  def set_position(self, position):
    self.position = position
    for head in range(self.num_attention_heads):
      self.softmax[head].dim = position
      self.query_key[head].out_features = position
      self.score_value[head].in_features = position
    self.init_operations()

  # Get the number of operations
  def n_op(self):
    # Initialize op
    op = 0
    # Accumulate operations
    op = op + self.q_proj.n_op()                  # Create query
    op = op + self.k_proj.n_op()                  # Create key
    op = op + self.v_proj.n_op()                  # Create value
    for head in range(self.num_attention_heads):
      op = op + self.query_key[head].n_op()       # Query-Key
      op = op + self.softmax[head].n_op()         # Softmax
      op = op + self.score_value[head].n_op()     # Score-Value
    op = op + self.o_proj.n_op()                  # Output projection
    return op

  def n_cycle(self):
    # Initialize cycle
    cycle = 0
    # Accumulate cycleerations
    cycle = cycle + self.q_proj.n_cycle()                  # Create query
    cycle = cycle + self.k_proj.n_cycle()                  # Create key
    cycle = cycle + self.v_proj.n_cycle()                  # Create value
    for head in range(self.num_attention_heads):
      cycle = cycle + self.query_key[head].n_cycle()       # Query-Key
      cycle = cycle + self.softmax[head].n_cycle()         # Softmax
      cycle = cycle + self.score_value[head].n_cycle()     # Score-Value
    cycle = cycle + self.o_proj.n_cycle()                  # Output projection
    return cycle


  # Get the number of parameters
  def n_param(self):
    # Initialize size
    size = 0
    # Accumulate parameters
    size = size + self.q_proj.n_param()
    size = size + self.k_proj.n_param()
    size = size + self.v_proj.n_param()
    size = size + self.o_proj.n_param()
    return size

''' Multi-Layer Perceptron '''
class MLP(NodeMixin):
  # Constructor
  def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, use_bias: bool=False, use_gate: bool=False, name=None, parent=None):
    super(MLP, self).__init__()
    # Options
    self.hidden_act = hidden_act
    self.use_bias = use_bias
    self.use_gate = use_gate
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent
    # Use gated mlp
    if self.use_gate:
      self.up_proj = nn.Linear(hidden_size, intermediate_size, use_bias, parent=self)
      self.gate_proj = nn.Linear(hidden_size, intermediate_size, use_bias, parent=self)
      self.down_proj = nn.Linear(intermediate_size, hidden_size, use_bias, parent=self)
    # Use vanilla mlp
    else:
      self.up_proj = nn.Linear(hidden_size, intermediate_size, use_bias, parent=self)
      self.down_proj = nn.Linear(intermediate_size, hidden_size, use_bias, parent=self)

    self.operations = []
    self.init_operations()

  def init_operations(self):
    self.operations.append(self.up_proj)
    if self.use_gate:
      self.operations.append(self.gate_proj)
    self.operations.append(self.down_proj)
    return

  # Get the number of operations
  def n_op(self):
    # Initialize op
    op = 0
    # Accumulate operations
    op = op + self.up_proj.n_op()
    if self.use_gate:
      op = op + self.gate_proj.n_op()
    op = op + self.down_proj.n_op()
    return op
  

  def n_cycle(self):
    # Initialize cycle
    cycle = 0
    # Accumulate cycleerations
    cycle = cycle + self.up_proj.n_cycle()
    if self.use_gate:
      cycle = cycle + self.gate_proj.n_cycle()
    cycle = cycle + self.down_proj.n_cycle()
    return cycle



  # Get the number of parameters
  def n_param(self):
    # Initialize size
    size = 0
    # Accumulate parameters
    size = size + self.up_proj.n_param()
    size = size + self.down_proj.n_param()
    if self.use_gate:
      size = size + self.gate_proj.n_param()
    return size

''' Mixture-of-Experts Block '''
class MoE():
  # Constructor
  def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, num_experts_per_tok: int, num_local_experts: int, use_bias: bool=False):
    # Options
    self.name = "MoE"
    self.hidden_size = hidden_size
    self.hidden_act = hidden_act
    self.num_experts_per_tok = num_experts_per_tok
    self.num_local_experts = num_local_experts
    self.use_bias = use_bias
    # Operation
    self.routing = nn.Softmax(dim=num_local_experts)
    # Weights
    self.gate = nn.Linear(hidden_size, num_local_experts, bias=self.use_bias)
    self.w1 = nn.Linear(hidden_size, intermediate_size, bias=self.use_bias)
    self.w2 = nn.Linear(intermediate_size, hidden_size, bias=self.use_bias)
    self.w3 = nn.Linear(hidden_size, intermediate_size, bias=self.use_bias)

    self.operations = []
    self.init_operations()

  def init_operations(self):
    self.operations.append(self.gate)
    self.operations.append(self.routing)
    for expert in range(self.num_experts_per_tok):
      self.operations.append(self.w1)
      self.operations.append(self.w3)
      self.operations.append(self.w1)
    return

  # Get the numboer of operations
  def n_op(self):
    # Initialize op
    op = 0
    # Accumulate operations
    op = op + self.gate.n_op()
    op = op + self.routing.n_op()
    for expert in range(self.num_experts_per_tok):
      op = op + self.w1.n_op()
      op = op + self.w3.n_op()
      op = op + self.w1.n_op()
      op = op + self.hidden_size
    op = op + self.hidden_size
    return op


  def n_cycle(self):
    # Initialize cycle
    cycle = 0
    # Accumulate cycleerations
    cycle = cycle + self.gate.n_cycle()
    cycle = cycle + self.routing.n_cycle()
    for expert in range(self.num_experts_per_tok):
      cycle = cycle + self.w1.n_cycle()
      cycle = cycle + self.w3.n_cycle()
      cycle = cycle + self.w1.n_cycle()
      cycle = cycle + self.hidden_size
    cycle = cycle + self.hidden_size
    return cycle


  # Get the number of parameters
  def n_param(self):
    # Initialize size
    size = 0
    # Accumulate parameters
    size = size + self.gate.n_param()
    for expert in range(self.num_local_experts):
      size = size + self.w1.n_param()
      size = size + self.w2.n_param()
      size = size + self.w3.n_param()
    return size

''' LanguageModel Head '''
class LMHead(NodeMixin):
  # Constructor
  def __init__(self, hidden_size: int, vocab_size: int, use_table: bool=True, use_bias: bool=False, name=None, parent=None):
    super(LMHead, self).__init__()
    # Options
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.use_table = use_table
    self.use_bias = use_bias
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent
    # Weight
    self.lm_head = nn.Linear(hidden_size, vocab_size, use_bias, parent=self)
    self.sorting = nn.Sorting(self.vocab_size, parent=self)
    self.operations = []
    self.init_operations()

  def init_operations(self):
    self.operations.append(self.lm_head)
    self.operations.append(self.sorting)
    return

  # Get the number of operations
  def n_op(self):
    # Initialize op
    op = 0
    # Accumulate parameters
    op = op + self.lm_head.n_op()
    return op
  
  def n_cycle(self):
    # Initialize cycle
    cycle = 0
    # Accumulate parameters
    cycle = cycle + self.lm_head.n_cycle()
    return cycle



  # Get the number of parameters
  def n_param(self):
    # Initialize size
    size = 0
    # Accumulate parameters
    if not self.use_table:
      size = size + self.lm_head.n_param()
    return size

##########################################################################
