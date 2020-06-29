import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import time
from keras.regularizers import l2
from keras.constraints import maxnorm
"""
This is a rewrite and extention of the example CNN used initially for demonstration
 in the class Intro to Data Science taught in the spring semester
of 2020 at Colorado School of Mines by Dr. Wendy Fisher
"""
np.random.seed(0)
tf.random.set_seed(0)

import matplotlib.pyplot as plt


# %matplotlib inline

class CNN(object):

    def plot_results(self, history):
        epoch_num = np.arange(1, len(history.history['loss']) + 1)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epoch_num, history.history['loss'], label='training_loss')
        plt.plot(epoch_num, history.history['val_loss'], label='test_loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epoch_num, history.history['accuracy'], label='training_accuracy')
        plt.plot(epoch_num, history.history['val_accuracy'], label='test_accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

    def loaddata_and_run(self, images_tr, labels_tr, images_te, labels_te):
        verbose = 1  # 0==no output, 1=accuracy/loss output, 2=progress bar output

        # Load data - #TODO eventually this should call a database of our own
        # (images_train, labels_train), (images_test, labels_test) = mnist.load_data()

        images_train, labels_train, images_test, labels_test = images_tr, labels_tr, images_te, labels_te

        # Use a subset of the full training and test sets for actual training and testing,
        # to accelerate training, and demonstrate possible pitfalls of smaller training data sets.

        n_train = len(images_train)
        # n_train = 1000  # use a subset and change size here if desired
        images_train = images_train[0:n_train]
        labels_train = labels_train[0:n_train]
        num_pix = len(images_train[0][0])  # assuming a square

        n_test = len(images_test)
        # n_train = 1000  # use a subset and change size here if desired
        images_test = images_test[0:n_test]
        labels_test = labels_test[0:n_test]

        ## You will not need to run this cell more than once, or cut/paste it elsewhere
        plt.figure(figsize=(8 * 2, 2 * 2))
        for i in range(16):
            plt.subplot(2, 8, i + 1)
            plt.imshow(images_train[i], cmap='gray')

        # Create TensorFlow Dataset objects to hold train and test data.
        images_train = images_train / 255
        images_train = np.expand_dims(images_train, axis=3)  # TensorFlow expects a channel dimension
        images_train = tf.cast(images_train, tf.float32)
        labels_train = tf.cast(labels_train, tf.float32)
        dataset_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train))

        images_test = images_test / 255
        images_test = np.expand_dims(images_test, axis=3)  # TensorFlow expects a channel dimension
        images_test = tf.cast(images_test, tf.float32)
        labels_test = tf.cast(labels_test, tf.float32)
        dataset_test = tf.data.Dataset.from_tensor_slices((images_test, labels_test))

        batch_size = 1000

        dataset_train = dataset_train.cache()
        dataset_train = dataset_train.shuffle(n_train)
        dataset_train = dataset_train.batch(batch_size)
        dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

        dataset_test = dataset_test.cache()
        dataset_test = dataset_test.batch(batch_size)
        dataset_test = dataset_test.cache()
        dataset_test = dataset_test.prefetch(tf.data.experimental.AUTOTUNE)

        ## This is the baseline model. Only modify it after copying it to cells further below.
        num_kernels = 3  # originally 3
        dense_layer_neurons = 64  # originally 64
        kernels_size = (3, 3)  # originally 3,3
        model = tf.keras.models.Sequential([

            tf.keras.layers.Conv2D(num_kernels, kernels_size, activation='relu'),  # , kernel_constraint=maxnorm(5)
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),

            tf.keras.layers.Conv2D(num_kernels, kernels_size, activation='relu'),  # , kernel_constraint=maxnorm(5)
            tf.keras.layers.Conv2D(num_kernels, kernels_size, activation='relu'),
            tf.keras.layers.Conv2D(num_kernels, kernels_size, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(dense_layer_neurons, activation='relu', kernel_regularizer=l2(0.1),
                                  bias_regularizer=l2(0.08)),  #
            tf.keras.layers.Dense(13, activation='softmax')
            # TODO: look into this->
            #  https://www.quora.com/What-are-some-useful-tips-for-choosing-and-tweaking-a-convolutional-neural-network-architecture-and-hyperparameters
            #  https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

        ])

        # Do not change any arguments in the call to model.compile()
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(0.001), #TODO: determine appropriate optimizer -> https://keras.io/api/optimizers/
            metrics=['accuracy'],
        )

        # Do not change any arguments in the call to model.fit()
        epochs = 30  # originally 30
        t = time.time()
        history = model.fit(dataset_train,
                            epochs=epochs,
                            validation_data=dataset_test,
                            verbose=verbose)
        print('Training duration: %f seconds.' % (time.time() - t))

        # Plot results
        self.plot_results(history)

        # clear previous model
        tf.keras.backend.clear_session()

        # save model
        model.save('CNN_model.model')
        model = tf.keras.models.load_model("CNN_model.model")

        return model, num_pix
