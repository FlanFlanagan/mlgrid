# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:11:35 2020
@author: rf846
"""

import random
import json
import numpy as np

def read_data_set():
    # training data------------------------------
    with open('datasets_and_generators/ANN_trainingdata_MASTER.json', 'r') as openfile:
        mastertrainlist = json.load(openfile)
    #put in list of tuples
    # print(len(mastertrainlist))
    mastertrainlist_tup = []
    for i in range(len(mastertrainlist)):
        temp = (np.array(mastertrainlist[i][0]), np.array(mastertrainlist[i][1]))
        mastertrainlist_tup.append(temp)
    # testing data-------------------------------
    with open('datasets_and_generators/ANN_testdata.json', 'r') as openfile:
        mastertestlist = json.load(openfile)
    #put in list of tuples
    mastertestlist_tup = []
    for i in range(len(mastertestlist)):
        temp = (np.array(mastertestlist[i][0]), np.array(mastertestlist[i][1]))
        mastertestlist_tup.append(temp)
    training_data = mastertrainlist_tup
    test_data = mastertestlist_tup
    return training_data, test_data

def load_data_wrapper(inp, out):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, te_d = read_data_set()
    training_inputs = [np.reshape(x, (inp, 1)) for x in tr_d[0]]
    training_results = [np.reshape(y, (inp, 1)) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (inp, 1)) for x in te_d[0]]
    testing_results = [np.reshape(y, (inp, 1)) for y in te_d[1]]
    test_data = zip(test_inputs, training_results)
    return (training_data, test_data)

def vectorized_result(j, out):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((out, 1))
    e[j] = 1.0
    return e

'''
# ---------------------
# - network.py example:
'''


import ANN_network

training_data, test_data = load_data_wrapper(1521, 1521)
test_data = list(test_data)
# training_data, test_data = read_data_set()
net = ANN_network.Network([1521, 100, 100, 1521])
net.SGD(training_data, 30, 5, -0.4, test_data=test_data)
