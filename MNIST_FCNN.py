# -*- coding: utf-8 -*-
import numpy as np

class MNIST_FCNN:
    def __init__(self, alpha, epoch, input_layer, classes, hidden_layer):
        self.__alpha = alpha
        self.__epoch = epoch
        self.__input_layer = input_layer
        self.__classes = classes
        self.__hidden_layer = hidden_layer

        rng = np.random.default_rng()
        self.__W1 = (rng.random(size = (self.__hidden_layer, self.__input_layer)) - 0.5) * np.sqrt(2 / 10)
        self.__W2 = (rng.random(size = (self.__classes, self.__hidden_layer)) - 0.5) * np.sqrt(2 / 10)
        self.__b1 = np.zeros((self.__hidden_layer, 1))
        self.__b2 = np.zeros((self.__classes, 1))
    
    def get_classes(self):
        return self.__classes

    def train(self, ds):
        cost = np.empty(self.__epoch)
        for i in range(self.__epoch):
            Z1, Z2, A1, A2 = self.__forward_propagate(ds)
            cost[i] = self.__cross_entropy_cost(ds.get_labels(), A2)
            dW1, dW2, db1, db2 = self.__backward_propagate(ds, Z1, Z2, A1, A2)
            self.__update_parameters(dW1, dW2, db1, db2)
        return cost
    
    def __forward_propagate(self, ds):
        Z1 = self.__W1.dot(ds.get_images()) + self.__b1
        A1 = self.__sigmoid_stable(Z1)
        Z2 = self.__W2.dot(A1) + self.__b2
        A2 = self.__stable_softmax(Z2)
        return Z1, Z2, A1, A2

    def test(self, ds):
        _, _, _, A2 = self.__forward_propagate(ds)
        accuracy = np.argmax(ds.get_labels(), axis = 0) == np.argmax(A2, axis = 0)
        return np.count_nonzero(accuracy) / np.size(accuracy)

    def __sigmoid_stable(self, Z):
        return np.where(Z < 0, np.exp(Z) / (1 + np.exp(Z)), 1 / (1 + np.exp(-Z)))

    def __sigmoid_derivative(self, Z):
        return self.__sigmoid_stable(Z) * (1 - self.__sigmoid_stable(Z))

    def __ReLU(self, Z):
        return np.maximum(Z, 0)

    def __ReLU_derivative(self, Z):
        return Z > 0

    def __stable_softmax(self, Z):
        exps = np.exp(Z - np.max(Z))
        return exps / np.sum(exps, axis = 0)

    def __backward_propagate(self, ds, Z1, Z2, A1, A2):
        X = ds.get_images()
        Y = ds.get_labels()
        _, m = X.shape
        dZ2 = A2 - Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
        dA1 = self.__W2.T.dot(dZ2)
        dZ1 = dA1 * self.__sigmoid_derivative(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)
        return dW1, dW2, db1, db2

    def __update_parameters(self, dW1, dW2, db1, db2):
        self.__W1 = self.__W1 - self.__alpha * dW1
        self.__W2 = self.__W2 - self.__alpha * dW2
        self.__b1 = self.__b1 - self.__alpha * db1
        self.__b2 = self.__b2 - self.__alpha * db2

    def __cross_entropy_loss(self, Y, A):
        return - np.sum(Y * np.log(A), axis = 0)

    def __cross_entropy_cost(self, Y, A):
        _, m = Y.shape
        L = self.__cross_entropy_loss(Y, A)
        return 1 / m * np.sum(L)

    def normalize(self, ds):
        X = ds.get_images()
        _, m = X.shape
        mu = 1 / m * np.sum(X, axis = 0)
        X = X - mu
        sigma2 = 1 / m * np.sum(np.square(X), axis = 0)
        ds.set_images(X / sigma2)

    def __kronecker_delta(self, i, j):
        return int(i == j)
