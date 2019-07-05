# -*- coding: utf-8 -*-

import pytest
from deepmirna.vectorizer import one_hot_encode_sequence, encode_data

__author__ = "simosini"
__copyright__ = "simosini"
__license__ = "mit"


def test_one_hot_encode_sequence():
    # check mirnas
    assert len(one_hot_encode_sequence('')) == 120
    assert len(one_hot_encode_sequence('U'*25)) == 120
    with pytest.raises(SystemExit):
        one_hot_encode_sequence('U'*31)
    # check site
    assert len(one_hot_encode_sequence('A'*40, mirna=False)) == 160
    with pytest.raises(SystemExit):
        one_hot_encode_sequence('A'*41, mirna=False)




