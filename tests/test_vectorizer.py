# -*- coding: utf-8 -*-

import pytest
from deepmirna.vectorizer import one_hot_encode_sequence, encode_data, init_encoders
import pandas as pd

__author__ = "simosini"
__copyright__ = "simosini"
__license__ = "mit"


def test_one_hot_encode_sequence_mirna_len():
    # check mirnas
    enc0, enc1 = init_encoders()
    assert len(one_hot_encode_sequence('', enc0, enc1)) == 120
    assert len(one_hot_encode_sequence('U'*25, enc0, enc1)) == 120
    with pytest.raises(SystemExit):
        one_hot_encode_sequence('U'*31, enc0, enc1)

def test_one_hot_encode_sequence_site_len():
    # check site
    enc0, enc1 = init_encoders()
    assert len(one_hot_encode_sequence('A'*40, enc0, enc1, mirna=False)) == 160
    with pytest.raises(SystemExit):
        one_hot_encode_sequence('A'*41, enc0, enc1, mirna=False)

def test_one_hot_encode_sequence_mirna_wellformed():
    # must not raise exception
    enc0, enc1 = init_encoders()
    raised = False
    try:
        one_hot_encode_sequence('A'*20, enc0, enc1)
    except SystemExit:
        raised = True
    assert raised == False

def test_one_hot_encode_sequence_mirna_badformed():
    enc0, enc1 = init_encoders()
    with pytest.raises(SystemExit):
        one_hot_encode_sequence('H'*20, enc0, enc1)

def test_one_hot_encode_sequence_site_wellformed():
    # must not raise exception
    enc0, enc1 = init_encoders()
    raised = False
    try:
        one_hot_encode_sequence('A'*40, enc0, enc1, mirna=False)
    except SystemExit:
        raised = True
    assert raised == False

def test_one_hot_encode_sequence_site_badformed():
    enc0, enc1 = init_encoders()
    with pytest.raises(SystemExit):
        one_hot_encode_sequence('H'*40, enc0, enc1, mirna=False)

def test_encode_data():
    fp = 'data/train_data.csv'
    nrows = 10
    df = pd.read_csv(fp, sep='\t', nrows=nrows)
    x, y = encode_data(df)
    assert x.shape == (nrows, 280)
    assert len(y) == nrows

def test_encode_data_method():
    m = 'random'
    fp = 'data/train_data.csv'
    nrows = 10
    df = pd.read_csv(fp, sep='\t', nrows=nrows)
    with pytest.raises(AttributeError):
        encode_data(df, m)

def test_encode_data_header():
    d = {'a':[0], 'b':[1], 'c':[2]}
    df = pd.DataFrame(d)
    with pytest.raises(SystemExit):
        encode_data(df)