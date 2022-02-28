from abc import ABC, abstractmethod
from re import L
from typing import List

import numpy as np


def weights_initialization(input_size, output_size):
    # USE BETTER INITIALIZATION
    return np.zeros((input_size, output_size))


class Layer(ABC):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, learning_rate):
        raise NotImplementedError



class LinearLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = weights_initialization(input_size, output_size)
        self.biases = np.zeros(output_size)
        self.data = None # Think what you need to store here

    def forward(self, x):
        # Store data what you need and compute x @ W + b (@ - DOT PRODUCT)
        return

    def backward(self, out, learning_rate):
        # dL/dW = x.T @ out
        # dL/db = out
        # dL/dx = out @ W.T
        # W = W - lr * dL/dW
        # b = b - lr * dL/db
        # update weight and biases with respect to gradient and return gradient with respect to input
        return 


class ReLU(Layer):

    def __init__(self) -> None:
        self.data = None # Think what you need to store here

    def forward(self, x):
        # Store data what you need and compute z = x if x >=0 else 0 (look for numpy where)
        return

    def backward(self, out, learning_rate):
        # dL/dz = (1 if x >=0 else 0) * out (think what is x)
        return 


class SoftmaxCE(Layer):

    def __init__(self) -> None:
        self.data = None # Think what you need to store here

    def forward(self, x):
        # Store data what you need and compute S[i] = exp^x_i / sum(exp^x_i)
        return

    def backward(self, out, learning_rate):
        # dL/dz = S[k] - y[k] if your out is a constant (more probable) your just need to do S[k] = S[k] - 1 (k is a class index)
        return 


class Graph:

    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.lr = learning_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y):
        for layer in reversed(self.layers):
            y = layer.backward(y, self.lr)
