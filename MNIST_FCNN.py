# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

class MNIST_FCNN:
    def __init__(self, alpha, epochs, input_layer, classes, hidden_layer_1, hidden_layer_2, seed):
        self.__alpha = alpha
        self.__epochs = epochs
        self.__input_layer = input_layer
        self.__classes = classes
        self.__hidden_layer_1 = hidden_layer_1
        self.__hidden_layer_2 = hidden_layer_2
        rng = np.random.default_rng(seed)
        self.__W1 = (rng.random(size = (self.__hidden_layer_1, self.__input_layer)) - 0.5)
        self.__W2 = (rng.random(size = (self.__hidden_layer_2, self.__hidden_layer_1)) - 0.5)
        self.__W3 = (rng.random(size = (self.__classes, self.__hidden_layer_2)) - 0.5)
        self.__b1 = np.zeros((self.__hidden_layer_1, 1))
        self.__b2 = np.zeros((self.__hidden_layer_2, 1))
        self.__b3 = np.zeros((self.__classes, 1))
        self.A3 = None
        self.cost = np.empty(self.__epochs)
        self.accuracy = 0
        
    def get_classes(self):
        return self.__classes

    def get_predictions(self):
        return self.A3

    def get_cost(self):
        return self.cost
    
    def train(self, ds):
        for i in range(self.__epochs):
            Z1, Z2, Z3, A1, A2 = self.__forward_propagate(ds)
            self.cost[i] = self.__cross_entropy_cost(ds.get_labels(), self.A3)
            dW1, dW2, dW3, db1, db2, db3 = self.__backward_propagate(ds, Z1, Z2, Z3, A1, A2)
            self.__update_parameters(dW1, dW2, dW3, db1, db2, db3)
    
    def __forward_propagate(self, ds):
        Z1 = self.__W1.dot(ds.get_images()) + self.__b1
        A1 = self.__sigmoid_stable(Z1)
        Z2 = self.__W2.dot(A1) + self.__b2
        A2 = self.__ReLU(Z2)
        Z3 = self.__W3.dot(A2) + self.__b3
        self.A3 = self.__softmax_stable(Z3)
        return Z1, Z2, Z3, A1, A2

    def get_accuracy(self):
        return np.count_nonzero(self.accuracy) / np.size(self.accuracy)
    
    def test(self, ds):
        _, _, _, _, _ = self.__forward_propagate(ds)
        self.accuracy = np.argmax(ds.get_labels(), axis = 0) == np.argmax(self.A3, axis = 0)
    
    def test_random(self, ds, pixels_x, pixels_y):
        _, m = ds.get_labels().shape
        image = np.random.randint(m)
        plt.imshow(np.reshape(ds.get_images().T[image], (pixels_x, pixels_y)))
        plt.show()
        print('prediction: ', self.get_predictions().T[image].argmax())
        print('true label: ', ds.get_labels().T[image].argmax())

    def __sigmoid_stable(self, Z):
        return np.where(Z < 0, np.exp(Z) / (1 + np.exp(Z)), 1 / (1 + np.exp(-Z)))

    def __sigmoid_derivative(self, Z):
        return self.__sigmoid_stable(Z) * (1 - self.__sigmoid_stable(Z))

    def __ReLU(self, Z):
        return np.maximum(Z, 0)

    def __ReLU_derivative(self, Z):
        return Z > 0

    def __softmax_stable(self, Z):
        exps = np.exp(Z - np.max(Z))
        return exps / np.sum(exps, axis = 0)

    def __backward_propagate(self, ds, Z1, Z2, Z3, A1, A2):
        X = ds.get_images()
        Y = ds.get_labels()
        _, m = X.shape
        dZ3 = self.A3 - Y
        dW3 = 1 / m * dZ3.dot(A2.T)
        db3 = 1 / m * np.sum(dZ3, axis = 1, keepdims = True)
        dA2 = self.__W3.T.dot(dZ3)
        
        dZ2 = dA2 * self.__ReLU_derivative(Z2)
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
        dA1 = self.__W2.T.dot(dZ2)

        dZ1 = dA1 * self.__sigmoid_derivative(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)

        return dW1, dW2, dW3, db1, db2, db3

    def __update_parameters(self, dW1, dW2, dW3, db1, db2, db3):
        self.__W1 = self.__W1 - self.__alpha * dW1
        self.__W2 = self.__W2 - self.__alpha * dW2
        self.__W3 = self.__W3 - self.__alpha * dW3
        self.__b1 = self.__b1 - self.__alpha * db1
        self.__b2 = self.__b2 - self.__alpha * db2
        self.__b3 = self.__b3 - self.__alpha * db3

    def __cross_entropy_loss(self, Y, A):
        return - np.sum(Y * np.log(A), axis = 0)

    def __cross_entropy_cost(self, Y, A):
        _, m = Y.shape
        L = self.__cross_entropy_loss(Y, A)
        return 1 / m * np.sum(L)

    def __kronecker_delta(self, i, j):
        return int(i == j)
