import random
import json
import numpy as np
import tensorflow as tf
import cv2
import os


def read_data_set():
    """
    Reads in data from MASTER json files and outputs data in lists labeled
    images_train, labels_train, images_test, labels_test
    Parameters
    ----------
    none

    Returns
    -------
    images_train, labels_train, images_test, labels_test: Lists
        lists containing the data used to train and test CNN
    """
    with open('datasets_and_generators/CNN_trainingimages_MASTER.json', 'r') as openfile:
        masterlist = json.load(openfile)
    masterarray = np.array(list(masterlist))
    with open('datasets_and_generators/CNN_traininglabels_MASTER.json', 'r') as openfile:
        masterlabels = json.load(openfile)
    labels = np.array(list(masterlabels))
    # split data
    n = len(masterlist)
    train_n = int(n * 0.7)  # indicate percentage of the data to be used in the training set here
    images_train = masterarray[:(train_n)]
    images_test = masterarray[train_n:]
    labels_train = labels[:(train_n)]
    labels_test = labels[train_n:]
    return images_train, labels_train, images_test, labels_test


def chunks(lst, n):
    """
    Yields successive n-sized chunks from lst.
    Obtained at: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    Parameters
    ----------
    lst: List
        lst to be broken into n sized chunks.
    n: Int
        size of chunks to break lst into.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def prepare(filepath, num_pix):
    """
    This function allows the CNN model to take in any image by
    taking in each image, turning them into black and white
    images and then turning these images into a matrix format
    that can be interpreted by the CNN to make predictions with.
    Parameters
    ----------
    filepath: String
        file path/file name
    num_pix: Int
        number of columns/rows in an image.

    Returns
    -------
    matrix: Array-Like Obj
        lists containing the data used to train and test CNN model
    """
    img = cv2.imread(filepath)  # reads in image
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # turns image to gray scale
    resized = cv2.resize(grayImage, (num_pix, num_pix),
                         interpolation=cv2.INTER_AREA)  # resizes image to proper pixel count
    (thresh, bawimg) = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)  # turns image black and white

    # show image
    # cv2.imshow('b&w', bawimg)
    # print("press enter\n")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # turns data into proper format for CNN
    piclist = bawimg.tolist()
    flat_list = [item for sublist in piclist for item in sublist]
    masterlist = []
    mainlist = []
    for j in flat_list:
        if j == 0:
            mainlist.append(0.)
        else:
            mainlist.append(1.)
    temp = chunks(mainlist, num_pix)
    masterlist.append(list(temp))
    matrix = np.asarray(list(masterlist))
    matrix = np.expand_dims(matrix, axis=3)  # TensorFlow expects a channel dimension
    matrix = tf.cast(matrix, tf.float32)
    return matrix


def testCNN(model, num_pix):
    """
    This is the function used to show how well the model can
    make predictions using real images. It takes in each image
    from  the folder called CNN_testimages which were created using
    ANN_CNN_test_data_generator.py. It loops through each image
    in the folder, preparing them using the prepare() function,
    predicting the poly type with the model, and printing out
    the file names with their relative predictions.
    Parameters
    ----------
    model: Trained CNN Model
        CNN-model.model
    num_pix: Int
        number of pixels in an image(matrix) row/column.
    """
    directory = os.fsencode('datasets_and_generators/CNN_testimages')
    test_imgs = []
    filename = []
    for file in os.listdir(directory):
        filename.append(os.fsdecode(file))
    for i in filename:
        test_imgs.append(prepare('CNN_testimages/' + str(i) + '', num_pix))  # 'CNN_testimages/poly6_1.jpg'

    # report outcomes
    prediction = []
    for j in test_imgs:
        prediction.append(model.predict_classes(j))
    for k in range(len(prediction)):
        print('image ' + str(filename[k]) + '', 'contains poly: ' + str(prediction[k]) + '\n')


# main---------------------------------------------
import CNN_network

images_train, labels_train, images_test, labels_test = read_data_set()  # reads in data
CNN = CNN_network.CNN()

# trainedmodel, numpix = CNN.loaddata_and_run(images_train, labels_train, images_test, labels_test) #trains model: comment out to use already trained model

# use code below to use model that is already trained
numpix = 39
trainedmodel = tf.keras.models.load_model("CNN_model.model")  # uncomment these to use already trained model

# test model by predicting image contents
testCNN(trainedmodel, numpix)
