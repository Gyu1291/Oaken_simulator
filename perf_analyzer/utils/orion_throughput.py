##########################################################################
##  Throughput Estimation
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

import json
import numpy as np

from perf_analyzer.models import AutoConfig, AutoModelForCausalLM

##########################################################################
## Defines
##########################################################################

# Architecture Configurations
DATA_BYTE         = 2             # Half-precision
MAC               = 64 * 16 * 2   # (Vector Dimension) x (Vector Lane)
# Memory Specification
MEMROY_BANDWIDTH  = 460           # GB/s
# Frequcency
LOGIC_FREQUENCY   = 220           # MHz
EFFECT_FREQUENCY  = MEMROY_BANDWIDTH / (MAC / 2 * DATA_BYTE) * 1000

##########################################################################
## Class
##########################################################################

class UtilPredictor():

  # Constructor
  def __init__(self, filename=None):
    self.slope = None
    self.intercept = None
    if filename is not None:
      self.read(filename)

  # Read dataset file
  def read(self, filename):
    self.model_op = []
    self.mac_util = []
    # Read JSON file
    with open(filename, "r") as f:
      raw_data = json.load(f)
    for data in raw_data["throughput"]:
      # Calculate the number of operations
      model = AutoModelForCausalLM.from_pretrained(data["model_id"])
      n_op = model.n_op() // data["num_device"]
      # Calculate the estimated throughput
      frequency = min(LOGIC_FREQUENCY, EFFECT_FREQUENCY)
      sec_per_token = (n_op / MAC) * (1 / frequency) / 1e6
      token_per_sec = 1 / sec_per_token
      # Calculate utilization of MAC
      util = data["throughput"] / token_per_sec
      # Store data points
      self.model_op.append(n_op)
      self.mac_util.append(util)
    # Linear regression
    self.linear_regression()

  # Get the linear regression
  def linear_regression(self):
    # X-axis should be logarithm
    x = np.log(self.model_op)
    y = self.mac_util
    # Calculate coefficients
    coefficients = np.polyfit(x, y, 1)
    self.slope = coefficients[0]
    self.intercept = coefficients[1]

  # Predict utilization
  def util(self, n_op):
    return self.slope * (np.log(n_op)) + self.intercept

##########################################################################
## Function
##########################################################################

def estimate(model: AutoModelForCausalLM, tp: int=1):
  n_op = model.n_op() // tp
  # Calculate the estimated throughput
  frequency = min(LOGIC_FREQUENCY, EFFECT_FREQUENCY)
  sec_per_token = (n_op / MAC) * (1 / frequency) / 1e6
  token_per_sec = 1 / sec_per_token
  return token_per_sec

def predict(model: AutoModelForCausalLM, predictor: UtilPredictor, batch_size: int=1, tp: int=1):
  n_op = model.n_op() // tp
  util = predictor.util(n_op)
  return estimate(model, tp) * util

##########################################################################