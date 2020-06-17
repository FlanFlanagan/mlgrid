import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import time
import cv2


np.random.seed(0)
tf.random.set_seed(0)

import matplotlib.pyplot as plt
# %matplotlib inline

class CNN(object):

    def plot_results(self, history):
        epoch_num = np.arange(1, len(history.history['loss'])+1)

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

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def prepare(self, filepath, num_pix):
        img = cv2.imread(filepath)
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # resize image
        resized = cv2.resize(grayImage, (num_pix, num_pix), interpolation=cv2.INTER_AREA)
        #turn b&w
        (thresh, bawimg) = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        #show image
        cv2.imshow('b&w', bawimg)
        print("press enter\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #turn data into proper format
        piclist = bawimg.tolist()
        flat_list = [item for sublist in piclist for item in sublist]
        masterlist = []
        mainlist = []
        for j in flat_list:
            if j == 0:
                mainlist.append(0.)
            else:
                mainlist.append(1.)
        tempa = self.chunks(mainlist, num_pix)
        masterlist.append(list(tempa))
        matrix = np.asarray(list(masterlist))
        matrix = matrix / 255
        matrix = np.expand_dims(matrix, axis=3)  # TensorFlow expects a channel dimension
        matrix = tf.cast(matrix, tf.float32)
        return matrix

    def loaddata_and_run(self, images_tr, labels_tr, images_te, labels_te):

        verbose = 1  # 0==no output, 1=accuracy/loss output, 2=progress bar output

        # Load data - #TODO eventually this should call a database of our own
        # (images_train, labels_train), (images_test, labels_test) = mnist.load_data()

        images_train, labels_train, images_test, labels_test = images_tr, labels_tr, images_te, labels_te

        # Use a subset of the full training and test sets for actual training and testing,
        # to accelerate training, and demonstrate possible pitfalls of smaller training data sets.

        n_train = len(images_train)
        images_train = images_train[0:n_train]
        labels_train = labels_train[0:n_train]
        num_pix = len(images_train[0][0]) #assuming a square

        n_test = len(images_train)
        images_test = images_test[0:n_test]
        labels_test = labels_test[0:n_test]

        ## You will not need to run this cell more than once, or cut/paste it elsewhere
        plt.figure(figsize=(8*2, 2*2))
        for i in range(16):
            plt.subplot(2, 8, i+1)
            plt.imshow(images_train[i], cmap='gray')

        # Create TensorFlow Dataset objects to hold train and test data.
        images_train = images_train/255
        images_train = np.expand_dims(images_train, axis=3) # TensorFlow expects a channel dimension
        images_train = tf.cast(images_train, tf.float32)
        labels_train = tf.cast(labels_train, tf.float32)
        dataset_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train))

        images_test = images_test/255
        # print(images_test)
        # print(type(images_test))
        # print(images_test.shape)
        images_test = np.expand_dims(images_test, axis=3) # TensorFlow expects a channel dimension
        # print(images_test)
        # print(images_test.shape)
        images_test = tf.cast(images_test, tf.float32)
        # print(images_test)
        # print(images_test.shape)
        labels_test = tf.cast(labels_test, tf.float32)
        dataset_test = tf.data.Dataset.from_tensor_slices((images_test, labels_test))

        batch_size = 50

        dataset_train = dataset_train.cache()
        dataset_train = dataset_train.shuffle(n_train)
        dataset_train = dataset_train.batch(batch_size)
        dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

        dataset_test = dataset_test.cache()
        dataset_test = dataset_test.batch(batch_size)
        dataset_test = dataset_test.cache()
        dataset_test = dataset_test.prefetch(tf.data.experimental.AUTOTUNE)

        ## This is the baseline model. Only modify it after copying it to cells further below.
        num_kernels = 4
        dense_layer_neurons = 64
        kernels_size = (3, 3)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_kernels, kernels_size, activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

            tf.keras.layers.Conv2D(num_kernels, kernels_size, activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(dense_layer_neurons, activation='relu'),
            tf.keras.layers.Dense(13)
        ])


        # Do not change any arguments in the call to model.compile()
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy'],
        )



        # Do not change any arguments in the call to model.fit()
        epochs = 30
        t = time.time()
        history = model.fit(dataset_train,
                            epochs=epochs,
                            validation_data=dataset_test,
                            verbose=verbose)
        print('Training duration: %f seconds.' % (time.time() - t))

        # Plot results
        self.plot_results(history)

        #--------------------------------------------------------------My add-ons----------------------------------------------------------
        #clear previous model
        tf.keras.backend.clear_session()

        #save model
        model.save('64x3-CNN.model')

        model = tf.keras.models.load_model("64x3-CNN.model")
        testimg = self.prepare('poly6_real_img.JPG', num_pix)
        prediction = model.predict_classes(testimg)
        print("this picture contains poly:", prediction) #TODO: ensure the correct lables are getting outputted
