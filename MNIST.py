# -*- coding: utf-8 -*-
import struct as st
import numpy as np
from matplotlib import pyplot as plt

'''
inspired by Andrew Ng
https://www.youtube.com/watch?v=qzPQ8cEsVK8&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=38
'''

def main():
    '''
    load and format training data, initialize parameters
    '''
    X = convertIDXToArray('./data/train-images.idx3-ubyte', 'images').astype(int)
    X = X.T
    #X = normalize(X)
    
    Y = convertIDXToArray('./data/train-labels.idx1-ubyte', 'labels').astype(int)
    Y = Y.T
    
    W1, W2, b1, b2, C, alpha, epoch = initialize_parameters()
    Y = one_hot_encode(Y, C)

    '''
    Train network
    '''
    W1, W2, b1, b2 = train_neural_network(X, Y, W1, W2, b1, b2, alpha, epoch)
    
    '''
    load and format test data
    '''
    X = convertIDXToArray('./data/t10k-images.idx3-ubyte', 'images').astype(int)
    X = X.T
    #X = normalize(X)

    Y = convertIDXToArray('./data/t10k-labels.idx1-ubyte', 'labels').astype(int)
    Y = Y.T
    Y = one_hot_encode(Y, C)

    '''
    Test network
    '''
    accuracy = test_neural_network(X, Y, W1, W2, b1, b2)
    print('accuracy: ', accuracy)

def initialize_parameters():
    C = 10 # no of classes
    h = 15 # hidden layer size
    rng = np.random.default_rng(123)
    W1 = (rng.random(size = (h, 28 * 28)) - 0.5) * np.sqrt(2 / 10)
    W2 = (rng.random(size = (C, h)) - 0.5) * np.sqrt(2 / 10)
    b1 = np.zeros((h, 1))
    b2 = np.zeros((C, 1))
    alpha = 0.1
    epoch = 500
    return W1, W2, b1, b2, C, alpha, epoch

def train_neural_network(X, Y, W1, W2, b1, b2, alpha, epoch):

    cost = np.empty(epoch)

    for i in range(epoch):
        Z1, Z2, A1, A2 = forward_propagate(X, W1, W2, b1, b2)
        cost[i] = cross_entropy_cost(Y, A2)
        if ((i + 1) % 100 == 0):
            print(cost[i])
        dW1, dW2, db1, db2 = backward_propagate(X, Y, W1, W2, b1, b2, Z1, Z2, A1, A2)
        W1, W2, b1, b2 = update_parameters(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha)

    return W1, W2, b1, b2

def forward_propagate(X, W1, W2, b1, b2):
    Z1 = W1.dot(X) + b1
    A1 = sigmoid_stable(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = stable_softmax(Z2)
    return Z1, Z2, A1, A2

def test_neural_network(X, Y, W1, W2, b1, b2):
    _, _, _, A2 = forward_propagate(X, W1, W2, b1, b2)
    accuracy = np.argmax(Y, axis = 0) == np.argmax(A2, axis = 0)
    return np.count_nonzero(accuracy) / np.size(accuracy)

def sigmoid_stable(Z):
    return np.where(Z < 0, np.exp(Z) / (1 + np.exp(Z)), 1 / (1 + np.exp(-Z)))

def sigmoid_derivative(Z):
    return sigmoid_stable(Z) * (1 - sigmoid_stable(Z))

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_derivative(Z):
    return Z > 0

def stable_softmax(Z):
    exps = np.exp(Z - np.max(Z))
    return exps / np.sum(exps, axis = 0)

def backward_propagate(X, Y, W1, W2, b1, b2, Z1, Z2, A1, A2):
    _, m = X.shape
    dZ2 = A2 - Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
    dA1 = W2.T.dot(dZ2)
    
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)
    return dW1, dW2, db1, db2

def update_parameters(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha):
    W1 = W1 - alpha * dW1
    W2 = W2 - alpha * dW2
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    return W1, W2, b1, b2

def one_hot_encode(Y, C):
    m = Y.size
    one_hot_encoded_Y = np.zeros((C, m))
    for i in range(m):
        one_hot_encoded_Y[Y[i]][i] = 1
    return one_hot_encoded_Y

def cross_entropy_loss(Y, A):
    L = - np.sum(Y * np.log(A), axis = 0)
    return L

def cross_entropy_cost(Y, A):
    _, m = Y.shape
    L = cross_entropy_loss(Y, A)
    J = 1 / m * np.sum(L)
    return J
    
def normalize(X):
    _, m = X.shape
    mu = 1 / m * np.sum(X, axis = 0)
    X = X - mu
    sigma2 = 1 / m * np.sum(np.square(X), axis = 0)
    X = X / sigma2
    return X
        
def plot_picture(X, Y, n):
    plt.imshow(np.reshape(X.T[n], (28, 28)))
    plt.show()
    print(Y.T[n].argmax())
    return True

def kronecker_delta(i, j):
    return int(i == j)

def convertIDXToArray(filename, file_type):
    '''
    http://yann.lecun.com/exdb/mnist/
    convert MNIST data set to a numpy array
    https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
    https://docs.python.org/3/library/struct.html
    '''
    
    # Open the IDX files in readable binary mode
    data_file = open(filename, 'rb')
    
    # Read the magic number
    # ‘>’: big-endian
    # ‘4B’: 4 bytes unsigned char
    data_file.seek(0)
    _ = st.unpack('>4B', data_file.read(4))
    
    # Read the dimensions of the Image data-set
    # ‘>’: big-endian
    # ‘I’: unsigned int
    nImg = st.unpack('>I', data_file.read(4))[0] # num of images
    nImg = 1000
    if file_type == 'images':
        nR = st.unpack('>I', data_file.read(4))[0] # num of rows
        nC = st.unpack('>I', data_file.read(4))[0] # num of column

    # Declare Image NumPy array
    if file_type == 'images':
        indata_array = np.zeros((nImg, nR, nC))
        outdata_array = np.zeros((nImg, nR * nC), dtype = int)
    elif file_type == 'labels':
        outdata_array = np.zeros((nImg), dtype = int)

    # Reading the Image data
    # ‘>’: big-endian
    # ‘B’: unsigned char
    # each pixel data is 1 byte
    if file_type == 'images':
        indata_array = np.asarray(st.unpack('>' + 'B' * nImg * nR * nC * 1, data_file.read(nImg * nR * nC * 1))).reshape((nImg, nR, nC))
        for i in range(nImg):
            index = 0
            for j in range(nR):
                for k in range(nC):
                    outdata_array[i][index] = indata_array[i][j][k]
                    index = index + 1
    elif file_type == 'labels':
        indata_array = np.asarray(st.unpack('>' + 'B' * nImg * 1, data_file.read(nImg * 1))).reshape(nImg, 1)
        for i in range(nImg):
            outdata_array[i] = indata_array[i][0]
    # Returning the array
    return outdata_array

if __name__ == '__main__':
    main()
