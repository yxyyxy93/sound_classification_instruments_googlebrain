import numpy as np
import pandas as pd
import librosa
import os
import glob
import re
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import time
import collections
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import pickle
from keras.utils.vis_utils import plot_model
from pathlib import Path


def cnn_model1():
    # bulid the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def cnn_model2():
    model = Sequential()
    input_shape=(128, 128, 1)

    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


if __name__ == "__main__":
    data_folder = Path("C:/Users/xiayang/OneDrive - UGent/other projects/sound_classification_instruments/dataset")
    # load X
    with open(data_folder / "train.pickle", 'rb') as f:
        train = pickle.load(f)
    with open(data_folder / "test.pickle", 'rb') as f:
        test = pickle.load(f)
    with open(data_folder / "valid.pickle", 'rb') as f:
        valid = pickle.load(f)
    X_train, y_train = zip(*train)
    X_valid, y_valid = zip(*valid)
    X_test, y_test = zip(*test)
    # Reshape for CNN input
    X_train = np.array([x.reshape((128, 128, 1)) for x in X_train])
    X_test = np.array([x.reshape((128, 128, 1)) for x in X_test])
    X_valid = np.array([x.reshape((128, 128, 1)) for x in X_valid])
    # One-Hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, 11))
    y_test = np.array(keras.utils.to_categorical(y_test, 11))
    y_valid = np.array(keras.utils.to_categorical(y_valid, 11))
    # batch size
    batch_size = 500
    # number of classes
    num_classes = y_train.shape[1]
    print("The number of classes is {}".format(num_classes))
    # training epochs
    epochs = 20
    # where to save file
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model = cnn_model1()
    # initiate ADAM optimizer
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # train the model
    model.compile(optimizer="Adam",
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_valid, y_valid),
                        shuffle=True)
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.show()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, 'model')
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
