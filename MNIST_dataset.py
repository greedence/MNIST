# -*- coding: utf-8 -*-
import struct as st
import numpy as np

class MNIST_dataset:
    def __init__(self, training_dataset, test_dataset, classes):
        '''
        load and format training data, initialize parameters
        '''
        self.__training_dataset = training_dataset
        self.__test_dataset = test_dataset
        self.__classes = classes
        self.__images = None
        self.__labels = None

    def get_images(self):
        return self.__images

    def set_images(self, X):
        self.__images = X

    def get_labels(self):
        return self.__labels

    def load_training_dataset(self, no_of_pictures):
        self.__images = self.__convertIDXToArray(self.__training_dataset['images'], 'images', no_of_pictures)
        self.__labels = self.__convertIDXToArray(self.__training_dataset['labels'], 'labels', no_of_pictures)
        self.__labels = self.__one_hot_encode()
        
    def load_test_dataset(self, no_of_pictures):
        self.__images = self.__convertIDXToArray(self.__test_dataset['images'], 'images', no_of_pictures)
        self.__labels = self.__convertIDXToArray(self.__test_dataset['labels'], 'labels', no_of_pictures)
        self.__labels = self.__one_hot_encode()

    def __one_hot_encode(self):
        examples = self.__labels.size
        one_hot_encoded_labels = np.zeros((self.__classes, examples))
        for i in range(examples):
            one_hot_encoded_labels[self.__labels[i]][i] = 1
        return one_hot_encoded_labels

    def normalize(self):
        _, m = self.__labels.shape
        mu = 1 / m * np.sum(self.__labels, axis = 0)
        self.__labels = self.__labels - mu
        sigma2 = 1 / m * np.sum(np.square(self.__labels), axis = 0)
        self.__labels = self.__labels / sigma2

    def __convertIDXToArray(self, filename, file_type, no_of_pictures):
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
        if no_of_pictures != 0:
            nImg = no_of_pictures
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
        return outdata_array.astype(int).T
