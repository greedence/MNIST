# -*- coding: utf-8 -*-
#import numpy as np
#from matplotlib import pyplot as plt
import MNIST_dataset
import MNIST_FCNN

'''
inspired by: 
    Andrew Ng, DeepLearning.AI and Coursera
    Yann LeCun, Courant Institute, NYU
    Corinna Cortes, Google Labs, New York
    Christopher J.C. Burges, Microsoft Research, Redmond
links:
    https://www.youtube.com/watch?v=qzPQ8cEsVK8&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=38
    http://yann.lecun.com/exdb/mnist/
'''

def main():
    '''
    Initialize parameters
    '''
    training_dataset = {'images' : './data/train-images.idx3-ubyte' ,'labels' : './data/train-labels.idx1-ubyte'}
    test_dataset = {'images' : './data/t10k-images.idx3-ubyte' ,'labels' : './data/t10k-labels.idx1-ubyte'}
    alpha = 0.1
    epoch = 1000
    input_layer = 28 * 28
    classes = 10
    hidden_layer = 15
    '''
    load and format data
    '''
    nn = MNIST_FCNN.MNIST_FCNN(alpha, epoch, input_layer, classes, hidden_layer)
    ds = MNIST_dataset.MNIST_dataset(training_dataset, test_dataset, nn.get_classes())
    '''
    Train network
    '''
    ds.load_training_dataset()
    cost = nn.train(ds)
    print(cost)
    '''
    Test network
    '''
    ds.load_test_dataset()
    accuracy = nn.test(ds)
    print('accuracy: ', accuracy)
    #plt.imshow(np.reshape(ds.get_images()[0], (28, 28)))
    #plt.show()
    #print(ds.get_labels()[0].argmax())

if __name__ == '__main__':
    main()
