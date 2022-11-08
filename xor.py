# -*- coding: utf-8 -*-
'''
inspired by Omar Aflak from 'The Independent Code'
https://www.youtube.com/channel/UC1OLIHvAKBQy3o5LcbbxUSg
https://www.youtube.com/watch?v=Lakz2MoHy6o&list=PLQ4osgQ7WN6PGnvt6tzLAVAEMsL3LBqpm&index=2
'''

import numpy as np

from layer import dense
from activations import sigmoid
from losses import mse, mse_derivative

#from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

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
            input = x
            for layer in network:
                output = layer.forward_propagate(input)
                input = output

            # error
            error = mse(y, output)

            # backward propagate
            gradient = mse_derivative(y, output)
            for layer in reversed(network):
                gradient = layer.backward_propagate(gradient, alpha)

        error /= len(X)
        if (e + 1) % 250 == 0 or e == 0:
            print('epoch %d/%d, error = %f' % (e + 1, epochs, error))

    # test
    nx, ny = (11, 11)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y, sparse = False)
    z = np.empty(shape = (nx, ny))

    ax = plt.axes(projection='3d')
    ax.set_title('xor')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('prediction')
    #ax.view_init(15, 230)

    for i in range(nx):
        for j in range(ny):
            input = np.reshape((xv[i, j], yv[i, j]), (2, 1))
            for layer in network:
                output = layer.forward_propagate(input)
                input = output
            z[i][j] = output

    ax.plot_surface(xv, yv, z)
    plt.savefig('./data/xor_plot.png', dpi = 600)
    plt.show

if __name__ == '__main__':
    main()
