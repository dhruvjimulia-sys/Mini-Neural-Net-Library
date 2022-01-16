import numpy as np

class Layer:
  def __init__(self, no_prev_layer, no_this_layer):
    self.weights = np.random.rand(no_this_layer, no_prev_layer)
    self.biases = np.random.rand(no_this_layer)
    self.no_of_neurons = no_this_layer
    self.input = []
    self.output = []

  def calculate(self, input):
     self.input = np.array(input)
     multiplied = np.matmul(self.weights, input)
     output = np.add(multiplied, self.biases)
     self.output = self.sigmoid(output)
     return self.output

  def sigmoid(self, matrix):
    return 1 / (1 + np.exp(-matrix))