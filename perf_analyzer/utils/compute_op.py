##########################################################################
##  Compute Operation
##
##  Authors:  Gyubin    Choi  ( gb.choi@hyperaccel.ai       )
##  Version:  1.3.1
##  Date:     2024-03-06      ( v1.3.1, init                )
##
##########################################################################

from collections import OrderedDict
import json
import numpy as np

from perf_analyzer.models import AutoConfig, AutoModelForCausalLM

##########################################################################
## Function
##########################################################################

def staged_op(model: AutoModelForCausalLM, input_token, output_token, batch_size):
  staged_op_2 = {key: value * batch_size for key, value in model.aggregate_nodes(2, input_token + output_token - 1).items()}
  staged_op_3 = {key: value * batch_size for key, value in model.aggregate_nodes(3, input_token + output_token - 1).items()}
  return staged_op_2, staged_op_3

def aggregate_op_nodes(root, position):
  # Setup the token position
  root.set_position(position)
  # Operation count list
  op_dict = OrderedDict()
  for node in root.leaves:
    # Create new operation type
    if node.name not in op_dict.keys():
      op_dict[node.name] = node.n_op() * position
    # Accmulate the operation count
    else:
      op_dict[node.name] = op_dict[node.name] + (node.n_op() * position)
  return op_dict

def aggregate_module_nodes(root, position):
  # Setup the token position
  root.set_position(position)
  # Get the module nodes
  module_list = []
  for node in root.leaves:
    if node.parent not in module_list:
      module_list.append(node.parent)
  # Operation count list
  op_dict = OrderedDict()
  for node in module_list:
    # Create new operation type
    if node.name not in op_dict.keys():
      op_dict[node.name] = node.n_op() * position
    # Accmulate the operation count
    else:
      op_dict[node.name] = op_dict[node.name] + (node.n_op() * position)
  return op_dict

##########################################################################

if __name__ == "__main__":
  model = AutoModelForCausalLM.from_pretrained("gpt2")
  op = aggregate_module_nodes(model)
  print(op)
