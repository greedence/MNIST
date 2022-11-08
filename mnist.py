# -*- coding: utf-8 -*-
'''
inspired by Omar Aflak from 'The Independent Code'
https://www.youtube.com/channel/UC1OLIHvAKBQy3o5LcbbxUSg
https://www.youtube.com/watch?v=Lakz2MoHy6o&list=PLQ4osgQ7WN6PGnvt6tzLAVAEMsL3LBqpm&index=2
'''

import numpy as np
import pickle
import mnist_dataset
from layer import dense, convolutional, reshape, softmax
from activations import sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_derivative

def main():
    load_data = True
    n = 100
    if load_data == True:
        (x_training, y_training), (x_test, y_test) = mnist_dataset.load_data(n)
        x_training, y_training = format_data(x_training, y_training)
        x_test, y_test = format_data(x_test, y_test)
        pickle.dump((x_training, y_training), open('./data/training_set.p', 'wb'))
        pickle.dump((x_test, y_test), open('./data/test_set.p', 'wb'))
    else:
        (x_training, y_training) = pickle.load(open('./data/training_set.p', 'rb'))
        (x_test, y_test) = pickle.load(open('./data/test_set.p', 'rb'))

    network = [
        convolutional((1, 28, 28), 3, 5),
        sigmoid(),
        reshape((5, 26, 26), (5 * 26 * 26, 1)),
        dense(5 * 26 * 26, 100),
        sigmoid(),
        dense(100, 10),
        softmax()
        ]
    epochs = 10
    alpha = 0.1
    for e in range(epochs):
        error = 0
        for x, y in zip(x_training, y_training):
            input = x
            for layer in network:
                output = layer.forward_propagate(input)
                input = output
            error = binary_cross_entropy(y, output)
            gradient = binary_cross_entropy_derivative(y, output)
            for layer in reversed(network):
                gradient = layer.backward_propagate(gradient, alpha)
        error /= len(x_training)
        print('epoch =', e, 'error =', error)

    for x, y in zip(x_training, y_training):
        input = x
        for layer in network:
            output = layer.forward_propagate(input)
            input = output
        print('prediction =', np.argmax(output), 'label =', np.argmax(y))

def format_data(x, y):
    m = len(x)
    x = x.reshape(m, 1, 28, 28)
    x = x.astype('float32') / 255
    y = one_hot_encode(y)
    y = y.reshape(m, 10, 1)
    return x, y

def one_hot_encode(y):
    m = y.size
    one_hot_encoded_y = np.zeros((m, 10), dtype = int)
    for i in range(m):
        one_hot_encoded_y[i][y[i]] = 1
    return one_hot_encoded_y

if __name__ == '__main__':
    main()
