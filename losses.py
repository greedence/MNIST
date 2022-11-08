# -*- coding: utf-8 -*-
'''
inspired by Omar Aflak from 'The Independent Code'
https://www.youtube.com/channel/UC1OLIHvAKBQy3o5LcbbxUSg
https://www.youtube.com/watch?v=Lakz2MoHy6o&list=PLQ4osgQ7WN6PGnvt6tzLAVAEMsL3LBqpm&index=2
'''

import numpy as np

def mse(y_label, y_prediction):
    return np.square(y_label - y_prediction)

def mse_derivative(y_label, y_prediction):
    return (-2) * (y_label - y_prediction)

def binary_cross_entropy(y_label, y_prediction):
    return -np.mean(y_label * np.log(y_prediction) + (1 - y_label) * np.log(1 - y_prediction))

def binary_cross_entropy_derivative(y_label, y_prediction):
    return ((1 - y_label) / (1 - y_prediction) - y_label / y_prediction) / np.size(y_label)

def cross_entropy(y_label, y_prediction):
    m = np.size(y_label)
    l =  (-1) * np.sum(y_label * np.log(y_prediction), axis = 1)
    return 1 / m * np.sum(l)

def cross_entropy_derivative(y_label, y_prediction):
    return y_prediction - y_label
