import numpy as np
from copy import deepcopy
from random import shuffle


class ActivationFunction:
    def __init__(self, func, derivative_func):
        self.function = np.vectorize(func)
        self.derivative = np.vectorize(derivative_func)


sigmoid = ActivationFunction(
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: x * (1 - x)
)


def _list_to_vector(inp):
    return np.array(inp).reshape((-1, 1))


class NeuralNetwork:
    def __init__(self, layer_sizes: list, activation_function=sigmoid):
        """layer_sizes is a list of integers specifying the size of each layer.
        The first number is the input layer, and the last is the output layer.
        """
        # assert len(layer_sizes) == 3, 'Only one hidden layer for now'
        self.activation_function = activation_function.function
        self.activation_derivative = activation_function.derivative

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.num_inputs = layer_sizes[0]
        self.num_outputs = layer_sizes[-1]

        self.weights = [np.random.randn(next_size, size)
                        for next_size, size in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.biases = [np.random.randn(size, 1) for size in layer_sizes[1:]]

    def guess(self, inputs: list):
        if len(inputs) != self.num_inputs:
            raise Exception(f'Wrong number of inputs {inputs}: Expected {self.num_inputs}')

        results = _list_to_vector(inputs)
        for weight, bias in zip(self.weights, self.biases):
            try:
                results = weight @ results
            except TypeError:
                print(f'w:\n{weight},\nr:\n{results}')
            results = results + bias
            results = self.activation_function(results)
        return results.flatten().tolist()

    def __copy__(self):
        result = NeuralNetwork(self.layer_sizes)
        result.activation_function = self.activation_function
        result.activation_derivative = self.activation_derivative
        result.weights = deepcopy(self.weights)
        result.biases = deepcopy(self.biases)
        return result

    def copy(self):
        return self.__copy__()

    def mutate(self, mutate_function):
        mutate_function = np.vectorize(mutate_function)
        self.weights = [mutate_function(weight) for weight in self.weights]
        self.biases = [mutate_function(bias) for bias in self.biases]
