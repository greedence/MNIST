# -*- coding: utf-8 -*-
'''
inspired by 'The Independent Code'
https://www.youtube.com/channel/UC1OLIHvAKBQy3o5LcbbxUSg
'''
import numpy as np
from layer import dense
from activations import sigmoid
from losses import mse, mse_derivative

def main():
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
    network = [
        dense(2, 2),
        sigmoid(),
        dense(2, 1),
        sigmoid()
        ]
    epochs = 1000
    alpha = 1
    
    for e in range(epochs):
        error = 0
        for x, y in zip(X, Y):

            # forward propagate
            output = x
            for layer in network:
                output = layer.forward_propagate(output)

            # error
            error = mse(y, output)

            # backward propagate
            gradient = mse_derivative(y, output)
            for layer in reversed(network):
                gradient = layer.backward_propagate(gradient, alpha)

        error /= len(X)
        if (e + 1) % 100 == 0 or e == 0:
            print('epoch %d/%d, error = %f' % (e + 1, epochs, error))
    print()
    for i in range(4):
        output = np.array(X[i])
        for layer in network:
            output = layer.forward_propagate(output)
        print('layer one weights')
        print(network[0].weights)
        print('layer one bias')
        print(network[0].bias)
        print('layer one output')
        print(network[1].input)
        print('layer one activated output')
        print(network[2].input)
        print()
        print('layer two weights')
        print(network[2].weights)
        print('layer two bias')
        print(network[2].bias)
        print('layer two output')
        print(network[3].input)
        print('layer two activated output')
        print(np.where(network[3].input < 0, np.exp(network[3].input) / (1 + np.exp(network[3].input)), 1 / (1 + np.exp(-network[3].input))))
        print('prediction for', X[i][0], X[i][1], 'is', output[0])
        print()


if __name__ == '__main__':
    main()
