##########################################################################
##  Modeling HuggingFace Functionalities
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

import torch
from anytree import Node, NodeMixin, RenderTree

##########################################################################
## Class
##########################################################################

class Parameter():
  # Constructor
  def __init__(self, dim: list):
    self.dim = torch.LongTensor(dim)
  
  # Get the dimension
  def shape(self) -> torch.Tensor:
    return self.dim
  
  # Get the number of parameters
  def n_param(self) -> int:
    return torch.prod(self.dim).item()

##########################################################################
## Class
##########################################################################

class Linear(NodeMixin):
  # Constructor
  def __init__(self, in_features, out_features, bias: bool=True, name=None, parent=None):
    super(Linear, self).__init__()
    # Options
    self.bias = bias
    self.in_features = in_features
    self.out_features = out_features
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent
    
    # Parameter
    self.w = Parameter([self.in_features, self.out_features])
    if self.bias:
      self.b = Parameter([self.out_features])

  # Get the number of operations
  def n_op(self) -> int:
    op = 0
    # Accumulate operations
    op = op + (self.in_features * self.out_features)        # Multiply
    # op = op + ((self.in_features - 1) * self.out_features)  # Accumulate
    # if self.bias:
      # op = op + self.out_features                           # Bias
    return op

  def n_cycle(self)-> int:
    cycle = 0
    return cycle

  # Get the number of parameters
  def n_param(self) -> int:
    # Initialize size
    size = 0
    # Accumulate parameters
    size = size + self.w.n_param()
    if self.bias:
      size = size + self.b.n_param()
    return size

class Matmul(NodeMixin):
  # Constructor
  def __init__(self, in_features, out_features, name=None, parent=None):
    super(Matmul, self).__init__()
    # Options
    self.in_features = in_features
    self.out_features = out_features
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent

  # Get the number of operations
  def n_op(self) -> int:
    op = 0
    # Accumulate operations
    op = op + (self.in_features * self.out_features)        # Multiply
    # op = op + ((self.in_features - 1) * self.out_features)  # Accumulate
    return op
  
  def n_cycle(self)-> int:
    cycle = 0
    return cycle

  # Get the number of parameters
  def n_param(self) -> int:
    return 0

class Mean(NodeMixin):
  # Constructor
  def __init__(self, dim: int=None, name=None, parent=None):
    super(Mean, self).__init__()
    # Options
    self.dim = dim
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent

  # Get the number of operations
  def n_op(self) -> int:
    op = 0
    # Accumulate operations
    # op = op + (self.dim - 1)  # Summation
    # op = op + 1               # Division
    return op

  def n_cycle(self)-> int:
    cycle = 0
    return cycle

  # Get the number of parameters
  def n_param(self) -> int:
    return 0
  
class Elementwise(NodeMixin):
  # Constructor
  def __init__(self, dim: int=None, name=None, parent=None):
    super(Elementwise, self).__init__()
    # Options
    self.dim = dim
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent

  # Get the number of operations
  def n_op(self) -> int:
    op = 0
    # Accumulate operations
    # op = op + self.dim  # Sub, Div,  Mul, Add
    return op
  
  def n_cycle(self)-> int:
    cycle = 0
    return cycle

  # Get the number of parameters
  def n_param(self) -> int:
    return 0

class StdDev(NodeMixin):
  # Constructor
  def __init__(self, dim: int=None, name=None, parent=None):
    super(StdDev, self).__init__()
    # Options
    self.dim = dim
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent

  # Get the number of operations
  def n_op(self) -> int:
    op = 0
    # Accumulate operations
    # op = op + self.dim        # Subtraction
    # op = op + self.dim        # Square
    # op = op + (self.dim - 1)  # Summation
    # op = op + 1               # Division
    # op = op + 1               # Square-root
    return op
  
  def n_cycle(self)-> int:
    cycle = 0
    return cycle

  # Get the number of parameters
  def n_param(self) -> int:
    return 0

class Softmax(NodeMixin):
  # Constructor
  def __init__(self, dim: int=None, name=None, parent=None):
    super(Softmax, self).__init__()
    # Options
    self.dim = dim
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent

  # Get the number of operations
  def n_op(self) -> int:
    op = 0
    # Accumulate operations
    # op = op + self.dim        # Sub
    # op = op + self.dim        # Exponent
    # op = op + (self.dim - 1)  # Summation
    # op = op + 1               # Division
    return op
  

  def n_cycle(self)-> int:
    cycle = 0
    return cycle

  # Get the number of parameters
  def n_param(self) -> int:
    return 0

# Embedding
class Embedding(NodeMixin):
  # Constructor
  def __init__(self, vector_size: int, entry_size: int, name=None, parent=None):
    super(Embedding, self).__init__()
    # Option
    self.vector_size = vector_size
    self.entry_size = entry_size
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent

    # Weight
    self.table = Parameter(dim=[entry_size, vector_size])
    
  # Get the number of operations
  def n_op(self):
    return 0
  
  def n_cycle(self)-> int:
    cycle = 0
    return cycle
  
  # Get the number of parameters
  def n_param(self):
    # Initialize size
    size = 0
    # Accumulate parameters
    size = size + self.table.n_param()
    return size

# Sorting
class Sorting(NodeMixin):
  # Constructor
  def __init__(self, vector_size: int, name=None, parent=None):
    super(Sorting, self).__init__()
    # Option
    self.vector_size = vector_size
    self.name = self.__class__.__name__ if name is None else name
    self.parent = parent

  # Get the number of operations
  def n_op(self):
    return 0

  def n_cycle(self):
    return 0
  
##########################################################################
