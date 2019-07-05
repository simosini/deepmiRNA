#######################################################################################################################
# This python3 file contains the functions needed to predict the miRNA:mRNA duplexes contained in the
# test dataframe used to measure the accuracy of the Neural Network.
# In order to create the list of candidate binding sites to be fed to the NN the whole 3UTR
# of the gene is scanned and potential subsequences are filtered in this order
# (filters ordered from quickest to slowest):
# 1 - complementarity: at least min_pairs complementary nucleotides in the seed region
#                      according to the candidate site selection method (CSSM) chosen
# 2 - free energy : the free energy must be low (below the chosen threshold) in order
#                   to consider a duplex stable
#######################################################################################################################

import datetime
import time
import logging

from tqdm import tqdm

import pandas as pd
import numpy as np

from keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score, classification_report, matthews_corrcoef

# to avoid tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import deepmirna.vectorizer as vec
import deepmirna.candidate_site_finder as csf
import deepmirna.site_accessibility as sa
import deepmirna.globs as gv

_logger = logging.getLogger(__name__)

def load_test_set(nrows=0, skip=0):
    """
    loads the test set
    :param nrows: specifies how many rows to read from the test set, 
                  if 0 reads the entire dataframe in memory
    :param skip: number of lines to skip from the beginning of the dataframe
    :return: the test set
    """
    header = ['mature_miRNA_transcript', '3UTR_transcript', 'functionality',]
    if nrows:
        df_test = pd.read_csv(gv.TEST_SET_LOCATION, usecols=gv.TEST_SET_COLUMNS, sep='\t',
                              nrows=nrows, skiprows=[i for i in range(1, skip)])
        df_test.columns = header
    else:
        df_test = pd.read_csv(gv.TEST_SET_LOCATION, usecols=gv.TEST_SET_COLUMNS, sep='\t',
                              skiprows=[i for i in range(1, skip)])
        df_test.columns = header

    return df_test

def _compute_score(model, mirna_transcript, candidate_site, label_encoder, one_hot_encoder):
    """
    predict the score of the duplex passed as argument
    :param model: the trained neural network model
    :param label_encoder: the label encoder to use 
    :param one_hot_encoder: one_hot encoder object to use
    :param mirna_transcript: the mirna transcript sequence
    :param candidate_site: the candidate mbs sequence
    :return: the score according to the model
    """

    # reshape data
    nn_input = np.concatenate((vec.one_hot_encode_sequence(mirna_transcript, label_encoder, one_hot_encoder),
                               vec.one_hot_encode_sequence(candidate_site, label_encoder, one_hot_encoder, mirna=False)))
    nn_input = np.expand_dims(nn_input, axis=0)

    # predict output
    score = model.predict(nn_input)
    return score

def predict_single(model, mirna_transcript, three_utr_transcript, label_encoder, one_hot_encoder, use_filter=True):
    """
    compute the list of potential candidate sites for the miRNA:gene pair and predict the scores.
    The output will be one iff for at least one candidate the computed score is above the threshold
    and at the same time no scores are below the negative threshold value.
    :param model: the model to use for prediction
    :param mirna_transcript: the mirna transcript
    :param three_utr_transcript: the full 3UTR transcript sequence
    :param label_encoder: the label_encoder to use
    :param one_hot_encoder: one_hot encoder object to use
    :param use_filter: whether to use the a-posteriori filter or not for classification
    :return: 1 for a functional prediction 0 otherwise
    """

    candidate_dict = csf.find_candidate_sites(mirna_transcript, three_utr_transcript)
    min_site_accessibility = gv.SITE_ACCESSIBILITY_THRESHOLD
    for start_idx, candidate_tuple in candidate_dict.items():
        if _compute_score(model, mirna_transcript, candidate_tuple[0], label_encoder,
                          one_hot_encoder) >= gv.POSITIVE_SCORE_THRESHOLD:
            # check site accessibility if filter is in use
            if not use_filter or sa.site_accessibility_energy(*sa.create_folding_chunk(start_idx, three_utr_transcript)) > min_site_accessibility:
                return 1

    return 0

def predict(model, mirnas, threeutrs, use_filter=True):
    """
    given the lists of pairs of mirnas and threeutrs predict their functionality 
    :param model : the nn to use for prediction
    :param mirnas: numpy array of the mirna to check
    :param threeutrs: numpy array of genes
    :param use_filter: whether to use site accessibility filter or not
    :return: the list of predictions 
    """

    preds = [0]*len(mirnas)
    label_encoder, one_hot_encoder = vec.init_encoders()
    for mirna, threeUTR, idx in zip(tqdm(mirnas), threeutrs, range(len(mirnas))):
        pred = predict_single(model, mirna, threeUTR, label_encoder, one_hot_encoder, use_filter=use_filter)
        preds[idx] = pred

    return preds

def _compute_metrics(preds, true_labels):
    """
    Compute the main metrics to evaluate the model on the test set
    :param preds: network's predictions
    :param true_labels: the ground truth
    :return: void just print the results on standard output
    """
    print('######### Confusion matrix ##########')
    cm = confusion_matrix(true_labels, preds)
    print(cm)
    precision, recall, f1score, support = precision_recall_fscore_support(true_labels, preds)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('F1-score: {}'.format(f1score))
    print('support: {}'.format(support))
    print('Balanced accuracy: {}'.format(balanced_accuracy_score(true_labels, preds)))

def test_model():
    """
    Test model performance on the test set according to config file parameters
    
    :return: void just print the main evaluation metrics on the console
    """

    # get testing parameters
    model = load_model(gv.BEST_MODEL_LOCATION)
    nrows = gv.NROWS
    skiprows = gv.SKIPROWS
    use_filter = gv.USE_FILTER

    _logger.info(' Loading test set ...')
    test_set = load_test_set(nrows=nrows, skip=skiprows)
    c = test_set.functionality.value_counts()
    _logger.info('The set contains {} positive and {} negative examples'.format(c[1], c[0]))

    mirnas = test_set['mature_mirna_transcript'].values
    threeutrs = test_set['3UTR_transcript'].values
    labels = test_set['functionality'].values

    del test_set

    _logger.info('Prediction {} filter started ...'.format('with' if use_filter else 'without'))
    time.sleep(.3) # avoid overlapping output with tqdm

    start = datetime.datetime.now()

    preds = predict(model, mirnas, threeutrs, use_filter=use_filter)
    time.sleep(.3)
    _logger.info('It took {} seconds to predict {} duplexes.'.format((datetime.datetime.now() - start).seconds, len(mirnas)))

    # print results
    _logger.info('Test prediction complete')
    _logger.info('Computing metrics ....')
    _compute_metrics(preds, labels)
