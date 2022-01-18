# Toy Neural Network Library

## Prerequisites

In order to run this program, you need to install Python on your computer. You can do this by following the instructions below:

https://www.python.org/downloads/

Note: This library was created using Python 3.8.10

After installing Python, you also need to install NumPy in the virtual environment using the command `pip install numpy`

## Getting Started
The usage of this library can be demonstrated in the `main.py` file, where the neural network library is used to solve the XOR problem. In order to run `main.py`, you can execute the following command in the project directory:
```
$ python3 main.py
```
## Library Usage

To begin using this library, you need to import two modules from the `src` directory:

```python
from src.nnet import NNet
from src.hyperparams import HyperParams
```

Then, you will be able to construct an instance of the class `HyperParams`, specifying the architecture of the neural network and optionally the learning rate and the number of epochs.

```python
params = HyperParams(architecture = [2, 2, 1], learning_rate = 10, epochs = 15)
```

Then you can create a neural network by passing this `params` object in the `NNet` constructor.
```python
nnet = NNet(params)
```
You can now call the following functions on the neural network object `nnet`:
* `predict(inputs)` : Returns the predicted value of the outputs given particular input
* `fit(inputs, outputs)` : Alter the weights and biases given a list of possible inputs and their corresponding outputs
* `calc_accuracy(inputs, outputs, no_of_tests, epsilon)` : Calculate the accuracy of a neural network given a list of possible inputs and their corresponding outputs

## List of hyperparameters TODO

* [x] Learning Rate
* [x] Architecture
* [x] Epochs
* [ ] More Ways to Initialize Weights
* [ ] More Ways to Initialize Biases
* [ ] More Activation Functions
* [ ] More Loss Functions
* [ ] Batches
* [ ] Regularization

## Credits
The neural network was implemented using the algorithm in the following book:

Tariq Rashid. Make Your Own Neural Network. Createspace Independent Publishing Platform; 2016.
