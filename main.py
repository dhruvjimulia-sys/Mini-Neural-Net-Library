from src.nnet import NNet
from src.hyperparams import HyperParams

params = HyperParams(architecture = [2, 2, 1], learning_rate = 10, epochs = 15)
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[1], [0], [0], [0]]

nnet = NNet(params)
p_acc = nnet.calc_accuracy(inputs, outputs, no_of_tests = 10000, epsilon = 0.49)
nnet.fit(inputs, outputs)
acc = nnet.calc_accuracy(inputs, outputs, no_of_tests = 10000, epsilon = 0.49)

print(f"Accuracy before fitting = {p_acc}")
print(f"Accuracy after fitting = {acc}")