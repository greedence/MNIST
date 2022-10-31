# -*- coding: utf-8 -*-
import numpy as np

def mse(y_label, y_prediction):
    return np.square(y_label - y_prediction)

def mse_derivative(y_label, y_prediction):
    return (-2) * (y_label - y_prediction)

def cross_entropy(y_label, y_prediction):
    m = np.size(y_label)
    l =  (-1) * np.sum(y_label * np.log(y_prediction), axis = 1)
    return 1 / m * np.sum(l)

def cross_entropy_derivative(y_label, y_prediction):
    return y_prediction - y_label
