# -*- coding: utf-8 -*-
'''
inspired by Omar Aflak from 'The Independent Code'
https://www.youtube.com/channel/UC1OLIHvAKBQy3o5LcbbxUSg
https://www.youtube.com/watch?v=Lakz2MoHy6o&list=PLQ4osgQ7WN6PGnvt6tzLAVAEMsL3LBqpm&index=2
'''

import numpy as np
from scipy import signal

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
        self.biases = np.zeros((output_size, 1))

    def forward_propagate(self, input):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.biases
        return self.output

    def backward_propagate(self, output_gradient, alpha):
        weights_derivatives = np.dot(output_gradient, self.input.T)
        bias_derivatives = output_gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= alpha * weights_derivatives
        self.biases -= alpha * bias_derivatives
        return input_gradient

class convolutional(layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward_propagate(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], 'valid')
        return self.output

    def backward_propagate(self, output_gradient, alpha):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], 'valid')
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], 'full')
        self.kernels -= alpha * kernels_gradient
        self.biases -= alpha * output_gradient
        return input_gradient

class reshape(layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_propagate(self, input):
        self.output = np.reshape(input, self.output_shape)
        return self.output

    def backward_propagate(self, output_gradient, alpha):
        input_gradient = np.reshape(output_gradient, self.input_shape)
        return input_gradient

class softmax(layer):
    def forward_propagate(self, input):
        exp = np.exp(input)
        self.output = exp / np.sum(exp)
        return self.output

    def backward_propagate(self, output_gradient, alpha):
        n = np.size(self.output)
        tile = np.tile(self.output, n)
        return np.dot(tile * (np.identity(n) - tile.T), output_gradient)
