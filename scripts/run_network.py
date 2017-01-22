"""
(c) January 2017 by Daniel Seita

Performs the third step in the GitHub documentation. This takes in the numpy
data from the previous step and then runs a neural network on it to classify it
as deformed or not. It uses Keras with the TensorFlow backend.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


def load_dataset():
    """ Loads the dataset and returns it in a dictionary. """
    data = {}
    data['X_train'] = np.load('final_data/X_train.npy')
    data['y_train'] = np.load('final_data/y_train.npy')
    data['X_valid'] = np.load('final_data/X_valid.npy')
    data['y_valid'] = np.load('final_data/y_valid.npy')
    data['X_test'] = np.load('final_data/X_test.npy')
    data['y_test'] = np.load('final_data/y_test.npy')
    return data


def run_cnn(data, 
            batch_size=32, 
            n_classes=2, 
            nb_epoch=100,
            data_augmentation=False):
    """ 
    Now this will finally run the CNN. It's based on the default design from the
    Keras github repository for Cifar-10 data, but it might be better to use a
    smaller network.
    """

    #####################
    ## DATA PROCESSING ##
    #####################

    # Have to add an exra dimension at the end due to dummy channels
    X_train, X_valid, X_test = data['X_train'], data['X_valid'], data['X_test']
    tr0, tr1, tr2 = X_train.shape
    va0, va1, va2 = X_valid.shape
    te0, te1, te2 = X_test.shape
    X_train = X_train.reshape(tr0, tr1, tr2, 1)
    X_valid = X_valid.reshape(va0, va1, va2, 1)
    X_test = X_test.reshape(te0, te1, te2, 1)

    Y_train = np_utils.to_categorical(data['y_train'], n_classes)
    Y_valid = np_utils.to_categorical(data['y_valid'], n_classes)
    Y_test = np_utils.to_categorical(data['y_test'], n_classes)

    print("FYI:\nX_train: {}\nX_valid: {}\nX_test: {}\n".format(
            X_train.shape,
            X_valid.shape,
            X_test.shape))
    print("FYI:\nY_train: {}\nY_valid: {}\nY_test: {}\n".format(
            Y_train.shape,
            Y_valid.shape,
            Y_test.shape))

    # Also get the range in [-1,1] or some small numbers (note tha the data are
    # _already_ zero-centered but magnitudes may still be high).
    X_train /= 255
    X_valid /= 255
    X_test /= 255

    #################
    ## THE NETWORK ##
    #################

    # Following a design mostly from the Keras default.
    model = Sequential()
    in_shape = X_train.shape[1:]

    # First set of convolutions.
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second set of convolutions.
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Actually, the last one will have only two possibilities so we don't really
    # need the softmax but I think it's still OK.
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # Debugging.
    model.summary()

    # Now we can try testing with and without data augmentation.
    if not data_augmentation:
        # This should still be OK I think but let's see..
        print("Not using data augmentation")
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_valid, Y_valid),
                  shuffle=True)
        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
    else:
        # Now data augmentation.
        print("Using real-time data augmentation.")
        print("TODO not implemented yet!")


if __name__ == "__main__":
    data = load_dataset()
    run_cnn(data,
            batch_size=32,
            n_classes=2,
            nb_epoch=50,
            data_augmentation=False)
