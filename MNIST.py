# -*- coding: utf-8 -*-
import MNIST_dataset
import MNIST_FCNN
import pickle

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
    # set to true to retrain the network, set to false to use pickle.load()
    train_netwok = True
    training_dataset = {'images' : './data/train-images.idx3-ubyte' ,'labels' : './data/train-labels.idx1-ubyte'}
    test_dataset = {'images' : './data/t10k-images.idx3-ubyte' ,'labels' : './data/t10k-labels.idx1-ubyte'}
    alpha = 0.1
    epoch = 1000
    pixels_x = 28
    pixels_y = 28
    classes = 10
    hidden_layer = 32
    seed = 200810071472
    '''
    Load and format data
    '''
    if train_netwok == True:
        nn = MNIST_FCNN.MNIST_FCNN(alpha, epoch, pixels_x * pixels_y, classes, hidden_layer, seed)
        ds = MNIST_dataset.MNIST_dataset(training_dataset, test_dataset, nn.get_classes())
        ds.load_training_dataset()
        pickle.dump(ds, open("ds.p", "wb"))
    else:
        ds = pickle.load(open("ds.p", "rb"))
    '''
    Train network
    '''
    if train_netwok == True:
        nn.train(ds)
        pickle.dump(nn, open("nn.p", "wb"))
    else:
        nn = pickle.load(open("nn.p", "rb"))
    print('cost after last epoch: ', nn.get_cost()[epoch - 1])
    '''
    Test network
    '''
    nn.test(ds)
    print('accuracy on test data: ', nn.get_accuracy())
    
    while input('want to test a random number [q to quit]? ') != 'q':
        nn.test_random(ds, pixels_x, pixels_y)

if __name__ == '__main__':
    main()
