
from keras.layers import Input, Conv2D, Dense, BatchNormalization, Flatten, Activation, MaxPool2D, ZeroPadding2D
from keras.models import  Model
from scipy import misc
from keras.backend import argmax
from fr_utils import load_dataset
import matplotlib.pyplot as plt

import numpy as np

from cnn_utils import convert_to_one_hot

def model(input_shape):

    x_input = Input(input_shape)
    # Zero-Padding: pads the border of X_input with zeroes
    x = ZeroPadding2D((3, 3))(x_input)

    # layer group1 32*32*32
    # CONV -> BN -> RELU Block applied to X
    x = Conv2D(32, (7, 7), strides=(1, 1))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2))(x)

    x = ZeroPadding2D((2, 2))(x)

    #layer group2 16*16*64
    x = Conv2D(64, (5, 5), strides=(1, 1))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)

    #layer group3 8*8*128
    x = Conv2D(128, (3, 3), strides=(1, 1))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2))(x)

    #layer group4 8*8*64
    x = Conv2D(64, (1, 1), strides=(1, 1))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)

    #layer group5 4*4*32
    x = Conv2D(32, (3, 3), strides=(1, 1))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2))(x)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    x = Flatten()(x)
    x = Dense(128, activation='sigmoid')(x)
    x = Dense(32, activation='sigmoid')(x)
    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=x_input, outputs=x)

    return model


if __name__ == '__main__':

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    index = 6
    plt.imshow(X_train_orig[index])
    plt.show()
    print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    print(Y_train.shape, X_train.shape)

    mod = model((64, 64, 3))
    mod.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    mod.fit(x = X_train, y = Y_train, epochs=4, batch_size=32)
    pred = mod.evaluate(x = X_test, y=Y_test)

    print("test accuracy: ", pred[1]*100, "%", end = '')
    print()






