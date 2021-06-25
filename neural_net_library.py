import numpy as np
import pandas as pd
import random

learning_rate = 10

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

class NNet:
  def __init__(self, *args):  #Assumes at least two layers: input and output
    self.no_of_layers = len(args)
    self.layers = []
    self.no_of_inputs = args[0]

    for index, arg in enumerate(args):
      if index != 0:
        layer = Layer(args[index - 1], arg)
        self.layers.append(layer)

  def fit (self, inputs, outputs): # Expects numpy array of inputs and outputs
    observed_outputs = self.predict(inputs)
    errors = outputs - observed_outputs
    self.backpropogate(errors)
    return

  def predict (self, input): # Input is single numpy array
    curr_input = input
    curr_output = 0
    for layer in self.layers:
      curr_output = layer.calculate(curr_input)
      curr_input = curr_output
    return curr_output

  def backpropogate(self, errors):
    curr_errors = errors
    for index, layer in reversed(list(enumerate(self.layers))):
      matrix_1 = np.atleast_2d(layer.output * (1 - layer.output) * curr_errors).T
      matrix_2 = np.array([np.array(layer.input)])
      
      delta_weights = learning_rate * np.matmul(matrix_1, matrix_2)
      layer.weights += delta_weights
      # print(delta_weights)
      delta_biases = layer.output * (1 - layer.output) * curr_errors
      layer.biases += delta_biases
      prev_layer_errors = np.matmul(np.matrix.transpose(layer.weights), curr_errors)
      curr_errors = prev_layer_errors


nnet = NNet(2, 6, 1)
# nnet.predict(np.linspace(0, 0.001, 100))

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[1], [0], [0], [0]]

for i in range(250):
  index = random.randint(0, 3)
  nnet.fit(inputs[index], outputs[index])
  # nnet.fit([0, 0], [0])

# Calculate accuracy
correct = 0
for i in range(10000):
  index = random.randint(0, 3)
  if round(nnet.predict(inputs[index])[0]) == outputs[index][0]:
    correct = correct + 1
print((correct/10000) * 100)

