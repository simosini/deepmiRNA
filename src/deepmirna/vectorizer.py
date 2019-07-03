#######################################################################################################################
# This python3 file contains the functions needed to encode genomic sequences in order to be used as
# input for the Neural Network.
# Currently the only available method is :
# 1 - OneHot Encoding: the usual approach to convert categorical data to binary vectors
# PS next version of the tool will also contain a word2vec-like encoding algorithm
#######################################################################################################################

import numpy as np

# encoders
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from deepmirna.globs import MAX_MIRNA_LEN, MBS_LEN, FLANKING_NUCLEOTIDES_SIZE


def init_encoders():
    """
    Initializes label and one hot encoders from sklearn. This is needed to avoid creating 2 encoder
    objects for each encoded sequence.
    :return: a tuple representing label_enoder, ohe_encoder
    """
    integer_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    return integer_encoder, one_hot_encoder


def one_hot_encode_sequence(sequence, label_encoder, one_hot_encoder, mirna=True):
    """
    in order to one hot encode a sequence we first map every nucleotide to an integer
    using the label encoder object. After that we can perform the encoding of the sequence.
    Some miRNA transcripts must be 0-padded in order to have the NN requested size (default 30) 
    hence pass miRNA=True for such sequences. miRNA binding sites have all the same length hence,
    for those sequences just use mirna=False
    :param sequence: the genomic sequence to one hot encode
    :param label_encoder: the label encoder object
    :param one_hot_encoder: the one hot encoder object
    :param mirna: defines if the sequence to encode is a miRNA or not (i.e. is a MBS)
    """
    # convert 'U' to 'T' cause we map them to same value
    sequence = sequence.replace('U','T')

    # transform the sequence (a string) to a list and then to a numpy array
    # we prepend the string 'ACGT' in order to always obtain the same encoding
    sequence = np.array(list('ACGT' + sequence))
    integer_encoded = label_encoder.fit_transform(sequence)

    # reshape the sequence to fit the one hot object shape
    reshaped_seq = integer_encoded.reshape(len(integer_encoded), 1)

    # the encoded sequence (cut the first 4 dummy nts)
    ohe_seq = one_hot_encoder.fit_transform(reshaped_seq)[4:]

    # insert padding if the sequence belongs to a miRNA (if needed)
    if mirna:
        ohe_seq = np.append(ohe_seq, [0.,0.,0.,0.]*(MAX_MIRNA_LEN - len(ohe_seq)))

    # return result
    return np.ndarray.flatten(ohe_seq)

def encode_data(train_df, encode_method='onehot'):
    """
    prepares the dataset for training. It first encode the sequences, according to the chosen encoding
    method, and then reshapes the examples in order to fit the neural network input layer 
    :param train_df: the dataframe representing the training set to be used
    :param encode_method: method to encode duplex: currently only onehot 
    :return: the training set encoded and ready to be fed to the neural network and the vector of true labels
    """
    # check encoding method. Currently 1 method available.
    method = encode_method.lower()
    if method != 'onehot':
        raise AttributeError('Only {} method available'.format('one_hot'))

    # extract values
    y_train = train_df.functionality.values.astype(np.uint32)
    mirnas = train_df.mature_mirna_transcript.values
    binding_sites = train_df.mbs_transcript.values

    sample_num = len(mirnas)
    sample_size = 4*(MBS_LEN + 2 * FLANKING_NUCLEOTIDES_SIZE + MAX_MIRNA_LEN)

    # init matrix and encoders
    x_train = np.zeros((sample_num, sample_size))
    label_enc, ohe_enc = init_encoders()

    # encode sequence
    for mirna, site, idx in zip(mirnas, binding_sites, range(len(mirnas))):
        enc_vec = np.concatenate([one_hot_encode_sequence(mirna, label_enc, ohe_enc),
                                  one_hot_encode_sequence(site, label_enc, ohe_enc, mirna=False)])
        x_train[idx] = enc_vec
    return x_train.astype(np.uint32), y_train