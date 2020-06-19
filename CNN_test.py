import random
import json
import numpy as np
import tensorflow as tf
import cv2
import os

def read_data_set():
    with open('datasets_and_generators/CNN_trainingimages.json', 'r') as openfile:
        masterlist = json.load(openfile)
    masterarray = np.array(list(masterlist))
    with open('datasets_and_generators/CNN_traininglabels.json', 'r') as openfile:
        masterlabels = json.load(openfile)
    labels = np.array(list(masterlabels))
    # split data
    n = len(masterlist)
    # print(n)
    train_n = int(n * 0.8)  # determine percentage of the data used in the training set here
    images_train = masterarray[:(train_n)]
    images_test = masterarray[train_n:]
    labels_train = labels[:(train_n)]
    labels_test = labels[train_n:]
    return images_train, labels_train, images_test, labels_test

import CNN_network

images_train, labels_train, images_test, labels_test = read_data_set()
CNN = CNN_network.CNN()
trainedmodel, numpix = CNN.loaddata_and_run(images_train, labels_train, images_test, labels_test)
CNN.testCNN(trainedmodel, numpix)