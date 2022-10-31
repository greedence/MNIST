# -*- coding: utf-8 -*-
import numpy as np

class layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward_propagate(self, input):
        pass

    def backward_propagate(self, output_gradient, alpha):
        pass
    
class dense(layer):
    def __init__(self, input_size, output_size):
        rng = np.random.default_rng(seed = 1472)
        self.weights = rng.random(size = (output_size, input_size)) - 0.5
        self.bias = np.zeros((output_size, 1))

    def forward_propagate(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward_propagate(self, output_gradient, alpha):
        weights_derivatives = np.dot(output_gradient, self.input.T)
        bias_derivatives = output_gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= alpha * weights_derivatives
        self.bias -= alpha * bias_derivatives
        return input_gradient
