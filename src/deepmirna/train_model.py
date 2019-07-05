###############################################################################################################
# This file contains the basic utilities to be used to train the MPL neural network
# and save the best result to file in order to make it available for the testing phase
###############################################################################################################
import datetime

import matplotlib.pyplot as plt
import logging
import pandas as pd


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

import deepmirna.globs as gv
from deepmirna.vectorizer import encode_data

_logger = logging.getLogger(__name__)

def _create_mlp_model(input_shape, keep_prob):
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

def train_eval(model, model_name, xtrain, ytrain, batch_size=128, n_epochs=15):
    """
    This is a helper function  used to validate network's parameters. 80% of data is used for training 
    the other 20% is used for validation purposes
    :param model: the compiled model to use
    :param model_name: a string representing model's name
    :param xtrain: the encoded training set data
    :param ytrain: the true labels
    :param batch_size: size of mini-batch to use for the training
    :param n_epochs: number of epochs
    :return: the keras history object containing training performances
    """
    # create output location
    output_location = os.path.join(gv.TRAIN_MODEL_DIR, model_name)

    ###### CALLBACKS #######

    #define the model checkpoint callback -> this will keep on saving the best model
    model_checkpoint = ModelCheckpoint(output_location, verbose=1, save_best_only=True)

    # fix random seed for reproducibility
    seed = 7

    # split into 80% for train and 20% for validation
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=seed)

    ####### TRAIN THE MODEL #########
    results = model.fit(xtrain, ytrain,
                        validation_data=(xval, yval),
                        callbacks=[model_checkpoint],
                        batch_size = batch_size,
                        epochs=n_epochs,
                        verbose=1)

    return results

def train(model, model_name, xtrain, ytrain, batch_size=128, n_epochs=15):
    """
    This function must be used to create the final model once parameters has already be defined
    through the train_eval function. The WHOLE training set will be used to train the model to
    be used for the testing stage. Please provide a reasonable name like "final_model" or 
    "best_model" in order to distinguish it from the validation model created previously.
    NO validation or test will be used.
    :param model: the validated and compiled model to use for training
    :param model_name: a string to name the model 
    :param xtrain: the encoded training set data
    :param ytrain: the true labels
    :param batch_size: size of mini-batch to use for the training
    :param n_epochs: number of epochs
    :return: the keras history object containing training performance over the complete dataset
    """

    # create output location
    output_location = os.path.join(gv.TRAIN_MODEL_DIR, model_name)

    ####### TRAIN THE FINAL MODEL #########
    results = model.fit(xtrain, ytrain,
                        batch_size = batch_size,
                        epochs=n_epochs,
                        verbose=1)

    # save model
    _logger.info('Saving model ...')
    model.save(output_location)

    return results

def _plot_model_history(history):
    """
    very basic plot showing accuracy and validation error during training. This is useful
    when used during the validation phase.
    :param history: the history object returned by keras fit call
    :return: show accuracy and validation error during training
    """
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

def evaluate_model():
    """
    Use this function to validate DeepMiRNA model 

    :return: keras history object with the results
    """

    # get training parameters
    model_name = gv.TRAIN_MODEL_NAME
    batch_size = gv.BATCH_SIZE
    n_epochs = gv.N_EPOCHS
    keep_prob = gv.KEEP_PROB

    training_df = pd.read_csv(gv.TRAIN_SET_LOCATION, sep='\t', usecols=gv.TRAIN_SET_COLUMNS)
    _logger.info(' Encoding the training set. This might take some time')
    xtrain, ytrain = encode_data(training_df)

    model = _create_mlp_model(xtrain.shape[1], keep_prob)

    _logger.info(' Training started')
    history = train_eval(model, model_name, xtrain, ytrain, batch_size, n_epochs)

    _plot_model_history(history)

    _logger.info(' model {} saved.'.format(model_name))
    _logger.info(' Best model achieved {:.2f} accuracy and {:.2f} validation loss on the validation set.'
                 .format(max(history.history['acc']), max(history.history['val_loss'])))

    return history

def train_model():
    """
    Train DeepMiRNA model over the whole training set and obtain the final model
    
    :return: keras history object 
    """

    # get training parameters
    model_name = gv.TRAIN_FINAL_MODEL_NAME
    batch_size = gv.BATCH_SIZE
    n_epochs = gv.N_EPOCHS
    keep_prob = gv.KEEP_PROB
    training_df = pd.read_csv(gv.TRAIN_SET_LOCATION, sep='\t', usecols=gv.TRAIN_SET_COLUMNS)

    _logger.info(' Encoding the training set. This might take some time')
    xtrain, ytrain = encode_data(training_df)

    _logger.info(' Building and compiling the model')
    model = _create_mlp_model(xtrain.shape[1], keep_prob)

    _logger.info(' Training started')
    history = train(model, model_name, xtrain, ytrain, batch_size, n_epochs)

    return history