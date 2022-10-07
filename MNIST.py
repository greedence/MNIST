# -*- coding: utf-8 -*-
import struct as st
import numpy as np
from matplotlib import pyplot as plt

def main():
    '''
    load and format training and test data
    '''
    
    training_data = convertIDXToArray('./data/train-images.idx3-ubyte', 'images')
    training_labels = convertIDXToArray('./data/train-labels.idx1-ubyte', 'labels')
    # test_data = convertIDXToArray('./data/t10k-images.idx3-ubyte', 'images')
    # test_labels = convertIDXToArray('./data/t10k-labels.idx1-ubyte', 'labels')
    X_train = training_data.astype(int).T
    Y_train = training_labels.astype(int).T
    # X_test = test_data.astype(int)
    # Y_test = test_labels.astype(int)

def softplus(Z):
    return np.log(1 + np.exp(Z))

def softplus_derivative(Z):
    return np.exp(Z) / (1 + np.exp(Z))
    
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
    nImg = 7
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
