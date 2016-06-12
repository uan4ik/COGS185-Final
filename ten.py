#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params, regularize_network_params
import sys
import os
import time

import numpy as np
import theano
#theano.config.optimizer='None'
#theano.config.optimizer_excluding='conv_dnn'
import theano.tensor as T
import pickle

import lasagne
import lasagne.layers as ll
from theano import shared

global num_labels
num_labels = 10 
global log_number
log_number = 1 
global max_acc, batch_size
max_acc = 0 
batch_size = 100 # 1000 when there are 10 classes



# load the preprocessed dataset
def load_dataset():
    return pickle.load(open(str(num_labels)+"_classes_ImageNet.p", "rb"))

def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename+".p")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def write_model_logs(epoch, accuracy, number):
    filename = "logs/log_" + str(number)
    if os.path.isfile(filename):
        with open(filename, 'a') as f:
            f.write('Epoch = ' + str(epoch) + 'val accuracy = '+str(accuracy) +' \n')
    else:
        with open(filename, 'w') as f:
            f.write('Epoch = ' + str(epoch) + 'val accuracy = '+str(accuracy) +' \n')



def build_cnn(input_var=None):
    global num_labels
    ###########################SMALLER MODEL################################
    network2 = lasagne.layers.InputLayer(shape=(None, 3, 100, 60),
                                        input_var=input_var)
    network2 = lasagne.layers.BatchNormLayer(network2)
    
    network2 = lasagne.layers.Conv2DLayer(
            network2, num_filters=2,filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network2 = lasagne.layers.MaxPool2DLayer(network2, pool_size=(2, 2))
    network2 = lasagne.layers.BatchNormLayer(network2)
   

    network2 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network2, p=0.5),
            num_units=32,
            nonlinearity=lasagne.nonlinearities.rectify)
    network2 = lasagne.layers.BatchNormLayer(network2)
    
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network2 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network2, p=0.5),
            num_units=num_labels,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    #######################################################################


    ############################THE LARGER MODEL #########################
    
    network = lasagne.layers.InputLayer(shape=(None, 3, 100, 60),
                                        input_var=input_var)
    network = lasagne.layers.BatchNormLayer(network)
   
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32,filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.BatchNormLayer(network)
   
 
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.BatchNormLayer(network)
    
    
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.BatchNormLayer(network)
   
    
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.BatchNormLayer(network)
 

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=num_labels,
            nonlinearity=lasagne.nonlinearities.softmax)

    #######################################################################
    return network, network2


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='cnn', num_epochs=1000):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    #target_var2 = T.fvector('targets2')    
    # Create neural network model (depending on first command line parameter)
    if model == 'cnn':
        network, network2 = build_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)
        return
    
    read_model_data(network, "saved_model_750_67.5000011921.p")    

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    prediction2 = lasagne.layers.get_output(network2)

    #prediction2 = prediction - prediction2
    
    #l2_penalty = regularize_network_params(network, l2) * 0.00001
    #l1_penalty = regularize_network_params(network, l1) * 0.001
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() #+ l2_penalty + l1_penalty 
    # We could add some weight decay as well here, see lasagne.regularization.

    loss2 = lasagne.objectives.categorical_crossentropy(prediction2, prediction)
    loss2 = loss2.mean() #+ l2_penalty + l1_penalty 

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
           loss, params, learning_rate=0.001, momentum=0.99)
    #updates = lasagne.updates.adagrad(loss, params, learning_rate=0.01)

    params2 = lasagne.layers.get_all_params(network2, trainable=True)
    updates2 = lasagne.updates.nesterov_momentum(
           loss2, params2, learning_rate=0.001, momentum=0.99)
    

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_prediction2 = lasagne.layers.get_output(network2, deterministic=True)
    test_loss2 = lasagne.objectives.categorical_crossentropy(test_prediction2,
                                                            target_var)
    test_loss = test_loss.mean()
    test_loss2 = test_loss2.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    test_acc2 = T.mean(T.eq(T.argmax(test_prediction2, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss,
                               updates=updates, allow_input_downcast=True)

    train_fn2 = theano.function([input_var], loss2,
                               updates=updates2, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc],
                                                  allow_input_downcast=True)
     
    val_fn2 = theano.function([input_var, target_var], [test_loss2, test_acc2],
                                                  allow_input_downcast=True)
     
    global batch_size
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_err2 = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size * num_labels, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_err2 += train_fn2(inputs)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_err2 = 0
        val_acc2 = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size * num_labels, shuffle=True):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            err2, acc2 = val_fn2(inputs, targets)
            val_err += err
            val_acc += acc
            val_err2 += err2
            val_acc2 += acc2
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}\t{:.6f}".format(train_err / train_batches,
							 train_err2 / train_batches))
        print("  validation loss:\t\t{:.6f}\t{:.6f}".format(val_err / val_batches, val_err2 / val_batches))
        print("  validation accuracy:\t\t{:.2f} %\t\t{:.2f} %".format(
            val_acc / val_batches * 100, val_acc2 / val_batches * 100))
          
        if epoch % 250 == 0 and epoch != 0:
            print("saving the model")
            write_model_data(network, "saved_model_"+str(epoch)+"_"+str(val_acc/val_batches*100))
        global log_number, max_acc        
        if epoch % 5 == 0 and val_acc/val_batches*100 > max_acc:
            max_acc = val_acc/val_batches*100
            print("saving the logs")
            write_model_logs(epoch, val_acc/val_batches*100, log_number)
        

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_err2 = 0
    test_acc2 = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batch_size * num_labels, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        err2, acc2 = val_fn2(inputs, targets)
        test_err2 += err2
        test_acc2 += acc2
        test_batches += 1
    print("Final results LARGER:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    print("Final results SMALLER:")
    print("  test loss:\t\t\t{:.6f}".format(test_err2 / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc2 / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
