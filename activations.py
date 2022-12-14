# -*- coding: utf-8 -*-
'''
inspired by Omar Aflak from 'The Independent Code'
https://www.youtube.com/channel/UC1OLIHvAKBQy3o5LcbbxUSg
https://www.youtube.com/watch?v=Lakz2MoHy6o&list=PLQ4osgQ7WN6PGnvt6tzLAVAEMsL3LBqpm&index=2
'''

import numpy as np
from layer import layer

class activation(layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward_propagate(self, input):
        self.input = input
        return self.activation(self.input)

    def backward_propagate(self, output_gradient, alpha):
        return np.multiply(output_gradient, self.activation_derivative(self.input))

class tanh(activation):
    def __init__(self):
        tanh = lambda z: np.tanh(z)
        tanh_derivative = lambda z: 1 - np.square(np.tanh(z))
        super().__init__(tanh, tanh_derivative)

class sigmoid(activation):
    def __init__(self):
        sigmoid = lambda z: np.where(z < 0, np.exp(z) / (1 + np.exp(z)), 1 / (1 + np.exp(-z)))
        sigmoid_derivative = lambda z: sigmoid(z) * (1 - sigmoid(z))
        super().__init__(sigmoid, sigmoid_derivative)

class relu(activation):
    def __init__(self):
        relu = lambda z: np.maximum(z, 0)
        relu_derivative = lambda z: z > 0
        super().__init__(relu, relu_derivative)

class softplus(activation):
    def __init__(self):
        softplus = lambda z: np.log(1 + np.exp(z))
        softplus_derivative = lambda z: 1 / (1 + np.exp(-z))
        super().__init__(softplus, softplus_derivative)
