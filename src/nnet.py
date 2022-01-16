import numpy as np
import random
from .layer import Layer

class NNet:
  # Pre: harchitecture has atleast 2 layers corresponding to
  #      input and output
  def __init__(self, hyperparams):
    arch = hyperparams.architecture
    assert len(arch) >= 2, 'Neural network must have atleast two layers'

    self.layers = []
    self.no_of_inputs = arch[0]
    for index, arg in enumerate(arch):
      if index != 0:
        layer = Layer(arch[index - 1], arg)
        self.layers.append(layer)
    
    self.learning_rate = hyperparams.learning_rate
    self.epochs = hyperparams.epochs

  def fit_value(self, inputs, outputs):
    observed_outputs = self.predict(inputs)
    errors = outputs - observed_outputs
    self.backpropogate(errors)
    return

  
  # Pre: len(input_data) == len(output_data)
  def fit(self, input_data, output_data):
    
    input_size = len(input_data) - 1
    for i in range(self.epochs):
      for j in range(input_size):
        index = random.randint(0, input_size)
        self.fit_value(input_data[index], output_data[index])

  def predict(self, input):
    curr_input = input
    curr_output = 0
    for layer in self.layers:
      curr_output = layer.calculate(curr_input)
      curr_input = curr_output
    return curr_output

  def backpropogate(self, errors):
    curr_errors = errors
    for layer in reversed(list(self.layers)):
      matrix_1 = np.atleast_2d(layer.output * (1 - layer.output) * curr_errors).T
      matrix_2 = np.array([np.array(layer.input)])
      
      delta_weights = self.learning_rate * np.matmul(matrix_1, matrix_2)
      layer.weights += delta_weights

      delta_biases = layer.output * (1 - layer.output) * curr_errors
      layer.biases += delta_biases
      prev_layer_errors = np.matmul(np.matrix.transpose(layer.weights), curr_errors)
      curr_errors = prev_layer_errors

  def calc_accuracy(self, inputs, outputs, no_of_tests, epsilon):
    no_correct = 0
    for i in range(no_of_tests):
      index = random.randint(0, len(inputs) - 1)
      predicted = self.predict(inputs[index])

      max_predicted = map(lambda x: x + epsilon, predicted)
      min_predicted = map(lambda x: x - epsilon, predicted)

      larger_than_min = all([o > p for o, p in zip(outputs[index], min_predicted)])
      smaller_than_max = all([o < p for o, p in zip(outputs[index], max_predicted)])

      if larger_than_min and smaller_than_max:
        no_correct = no_correct + 1
      
    return (no_correct/no_of_tests) * 100