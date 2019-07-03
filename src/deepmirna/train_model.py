###############################################################################################################
# This file contains the basic utilities to be used to train the MPL neural network
# and save the best result to file in order to make it available for the testing phase
###############################################################################################################
import datetime

#import matplotlib.pyplot as plt
import logging


# keras utilities
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# sklearn utilities
from sklearn.model_selection import train_test_split

# silence warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from deepmirna.globs import TRAIN_MODEL_DIR

_logger = logging.getLogger(__name__)

def create_mlp_model(input_shape, keep_prob=.7):
    """
    creates the MLP model to be used for training. Architecture and parameters
    have been defined through cross validation
    :param input_shape: the input dimension
    :param keep_prob: Represents the probability to keep a neuron during training
    :return: the model built and compiled ready for the training
    """

    # build network
    model = Sequential()
    model.add(Dense(400, input_dim=input_shape, activation='sigmoid'))
    model.add(Dropout(rate=keep_prob))

    model.add(Dense(150, activation='relu'))
    model.add(Dropout(rate=keep_prob))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate=keep_prob))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(rate=keep_prob))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(rate=keep_prob))
    model.add(Dense(1, activation='sigmoid'))

    # compile it
    adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def train_eval(NN_model, model_name, xtrain, ytrain, batch_size=128, n_epochs=15):
    """
    This function is used to validate network's parameters. 80% of data is used for training 
    the other 20% is used for validation purposes
    :param NN_model: the compiled model to use
    :param model_name: a string to be use to save the best model configuration (for example "model1")
    :param xtrain: the training set data
    :param ytrain: the true labels
    :param batch_size: size of mini-batch to use for the training
    :param n_epochs: number of epochs
    :return: the keras history object containing training performances
    """
    # get output location from configuration file
    output_location = os.path.join(TRAIN_MODEL_DIR, model_name + '.h5')

    ###### CALLBACKS #######

    #define the model checkpoint callback -> this will keep on saving the best model
    model_checkpoint = ModelCheckpoint(output_location, verbose=1, save_best_only=True)

    # fix random seed for reproducibility
    seed = 7

    # split into 80% for train and 20% for validation
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=seed)

    ####### TRAIN THE MODEL #########
    results = NN_model.fit(xtrain, ytrain,
                 validation_data=(xval, yval),
                 callbacks=[model_checkpoint],
                 batch_size = batch_size,
                 epochs=n_epochs,
                 verbose=1)

    return results

#def train(NN_model, ):



"""
def plot_model_history(history):

    # summarize history for accuracy
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show() 
    """