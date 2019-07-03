#######################################################################################################################
# This python3 file contains the functions needed to encode genomic sequences in order to be used as
# input for the Neural Network.
# Currently the only available method is :
# 1 - OneHot Encoding: the usual approach to convert categorical data to binary vectors
# PS next version of the tool will also contain a word2vec-like encoding algorithm
#######################################################################################################################

import pandas as pd
import numpy as np

# encoders
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from deepmirna.globs import MAX_MIRNA_LEN

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