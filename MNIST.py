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
    train_netwok = False
    training_dataset = {'images' : './data/train-images.idx3-ubyte' ,'labels' : './data/train-labels.idx1-ubyte'}
    test_dataset = {'images' : './data/t10k-images.idx3-ubyte' ,'labels' : './data/t10k-labels.idx1-ubyte'}
    alpha = 0.1
    epochs = 1000
    pixels_x = 28
    pixels_y = 28
    classes = 10
    hidden_layer_1 = 15
    hidden_layer_2 = 10
    seed = 4183
    #set to 0 if training on the whole dataset
    no_of_training_pictures = 0
    no_of_test_pictures = 0
    '''
    Load and format data
    '''
    if train_netwok == True:
        nn = MNIST_FCNN.MNIST_FCNN(alpha, epochs, pixels_x * pixels_y, classes, hidden_layer_1, hidden_layer_2, seed)
        ds = MNIST_dataset.MNIST_dataset(training_dataset, test_dataset, nn.get_classes())
        ds.load_training_dataset(no_of_training_pictures)
        pickle.dump(ds, open("ds_training.p", "wb"))
    else:
        ds = pickle.load(open("ds_training.p", "rb"))
    '''
    Train network
    '''
    if train_netwok == True:
        nn.train(ds)
        pickle.dump(nn, open("nn.p", "wb"))
    else:
        nn = pickle.load(open("nn.p", "rb"))
    print('cost after last epoch: ', nn.get_cost()[epochs - 1])
    '''
    Test network
    '''
    nn.test(ds)
    print('accuracy on training data: ', nn.get_accuracy())
    if train_netwok == True:
        ds.load_test_dataset(no_of_test_pictures)
        pickle.dump(ds, open("ds_test.p", "wb"))
    else:
        ds = pickle.load(open("ds_test.p", "rb"))
    nn.test(ds)
    print('accuracy on test data: ', nn.get_accuracy())
    
    while input('want to test a random number [q to quit]? ') != 'q':
        nn.test_random(ds, pixels_x, pixels_y)

if __name__ == '__main__':
    main()
