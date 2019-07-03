#################################################################################################
# This file sets the global variables to be used across all files of the package. It reads them
# from config.ini and sets the correct values in globs.py
#################################################################################################

from ast import literal_eval
from configparser import ConfigParser
import logging

import deepmirna.globs as gv

__author__ = "simosini"
__copyright__ = "simosini"
__license__ = "mit"

_logger = logging.getLogger(__name__)

def set_global_variables(config_file, use_default_values=True):
    """
    sets global variables to be used during the testing phase
    :param config_file: the path to the configuration file
    :param use_default_values: whether to keep default parameters values or not
           when this value is set to true nothing is done cause default values are already set
    :return: void
    """
    _logger.info(' Setting up global parameters...')

    if not use_default_values:

        # read config parameters
        config = ConfigParser()
        config.read(config_file)
        candidate_sites_params = config['candidate_sites']
        test_set_params = config['testing']
        train_set_params = config['training']
        comp = config['computation']

        # set test set parameters
        gv.TEST_SET_LOCATION = test_set_params.get('test_set_location')
        gv.TEST_SET_COLUMNS = literal_eval(test_set_params.get('test_set_cols'))

        # set train set parameters
        gv.TRAIN_SET_LOCATION = train_set_params.get('train_set_location')
        gv.TRAIN_SET_COLUMNS = literal_eval(train_set_params.get('train_set_cols'))
        gv.TRAIN_MODEL_DIR = train_set_params.get('train_model_dir')

        # set miRNA parameters
        gv.SEED_START = int(candidate_sites_params.get('seed_start'))
        gv.SEED_END = int(candidate_sites_params.get('seed_end'))
        gv.MAX_MIRNA_LEN = int(candidate_sites_params.get('max_mirna_len'))
        gv.MIN_COMPLEMENTARY_NUCLEOTIDES = int(candidate_sites_params.get('min_complementary_nucleotides'))

        # set mRNA parameters
        gv.MBS_LEN = int(candidate_sites_params.get('mbs_len'))
        gv.FOLDING_CHUNK_LEN = int(candidate_sites_params.get('folding_chunk_len'))
        gv.MIN_CHUNK_LEN = int(candidate_sites_params.get('min_chunk_len'))
        gv.FLANKING_NUCLEOTIDES_SIZE = int(candidate_sites_params.get('flunking_nucleotides_size'))

        # set filters parameters
        gv.FREE_ENERGY_THRESHOLD = float(candidate_sites_params.get('free_energy_threshold'))
        gv.SITE_ACCESSIBILITY_THRESHOLD = float(candidate_sites_params.get('site_accessibility_threshold'))

        # set sliding window stride
        gv.WINDOW_STRIDE = int(candidate_sites_params.get('window_stride'))

        # set number of cores to use
        gv.MAX_PROCESSES = int(comp.get('max_processes'))

    _logger.info(' Setup complete.')
