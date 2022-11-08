# -*- coding: utf-8 -*-
import struct as st
import numpy as np

def load_data(n):
    return  (load_training_dataset(n), load_test_dataset(n))

def load_training_dataset(n):
    training_dataset = {'images' : './data/train-images.idx3-ubyte' ,'labels' : './data/train-labels.idx1-ubyte'}
    return (convertIDXToArray(training_dataset['images'], 'images', n), convertIDXToArray(training_dataset['labels'], 'labels', n))
    
def load_test_dataset(n):
    test_dataset = {'images' : './data/t10k-images.idx3-ubyte' ,'labels' : './data/t10k-labels.idx1-ubyte'}
    return (convertIDXToArray(test_dataset['images'], 'images', n), convertIDXToArray(test_dataset['labels'], 'labels', n))

def convertIDXToArray(filename, file_type, n):
    '''
    http://yann.lecun.com/exdb/mnist/
    convert MNIST data set to a numpy array
    https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
    https://docs.python.org/3/library/struct.html
    '''
    data_file = open(filename, 'rb')
    data_file.seek(0)
    _ = st.unpack('>4B', data_file.read(4))
    nImg = st.unpack('>I', data_file.read(4))[0]
    if n != 0:
        nImg = n
    if file_type == 'images':
        nR = st.unpack('>I', data_file.read(4))[0]
        nC = st.unpack('>I', data_file.read(4))[0]
    if file_type == 'images':
        indata_array = np.asarray(st.unpack('>' + 'B' * nImg * nR * nC * 1, data_file.read(nImg * nR * nC * 1))).reshape((nImg, nR, nC))
    elif file_type == 'labels':
        indata_array = np.asarray(st.unpack('>' + 'B' * nImg * 1, data_file.read(nImg * 1))).reshape(nImg, 1)
    return indata_array.astype(int)
