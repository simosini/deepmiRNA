#################################################################################################
# This file contains global variables to be used across all modules of the package. Parameters are
# initialized with default values, but they can be overwritten using the values contained in
# config.ini. By default all data used for computation are saved in the data folder contained
# in the project root directory.
#################################################################################################

__author__ = "simosini"
__copyright__ = "simosini"
__license__ = "mit"

### DEFAULT VALUES ###

# test settings *****IMPORTANT : all file paths must be relative to the project root directory ******
TEST_SET_COLUMNS = ['mature_miRNA_transcript', '3UTR_transcript', 'functionality'] # test set cols to keep
TEST_SET_LOCATION = 'data/sample_test.csv' # test set file path
BEST_MODEL_LOCATION = 'models/best_model.h5'# the location of the model to test
NROWS = 0 # number of rows to read from the test set. 0 means read the whole dataset
SKIPROWS = 0 # number of rows to skip from the beginning of the dataset
USE_FILTER = True # whether to use site accessibility filter or not

# train settings *****IMPORTANT : all file paths must be relative to the project root directory ******
TRAIN_SET_COLUMNS = ['mature_miRNA_transcript', 'site_transcript', 'functionality'] # train set cols to keep
TRAIN_SET_LOCATION = 'data/train_data.csv' # train set file path
TRAIN_MODEL_DIR = 'models'
TRAIN_MODEL_NAME = 'model0.h5' # model name for validation: ****CHANGE THIS NAME AT EVERY RUN*****
TRAIN_FINAL_MODEL_NAME = 'final_model.h5' # model name after training over the whole training set
BATCH_SIZE = 128 # mini-batch size to use for the training
N_EPOCHS = 15 # number of epochs to ise for training
KEEP_PROB = 0.7 # dropout rate. Identify the probability to keep a neuron

# biological settings
MBS_LEN = 30 # maximum length for a potential binding site in the mRNA. Must be an even number
MAX_MIRNA_LEN = 30 # the maximum length of a miRNA sequence
SEED_START = 0 # index of the start of the extended seed region (usually 0 or 1) (inclusive)
SEED_END = 10 # index of the end of the extended seed region (usually 10) (exclusive)
MIN_COMPLEMENTARY_NUCLEOTIDES = 6 # minimum number of complementary nucleotides
FOLDING_CHUNK_LEN = 200 # length of the mRNA folding chunk to use to compute the needed opening energy
FLANKING_NUCLEOTIDES_SIZE = 5 # number of downstream and upstream nucleotides to consider for a binding sites
MIN_CHUNK_LEN = 200 # the minimum length a 3UTR chunk can have after splitting
WINDOW_STRIDE = 5 # the window step size when scanning the 3UTR of the gene (mRNA)
FREE_ENERGY_THRESHOLD = -10 # threshold to consider when filtering duplexes. Pairs with free energy greater than this threshold are filtered
SITE_ACCESSIBILITY_THRESHOLD = -14 # only folding chunks with an accessibility above this threshold are kept. The rest are filtered
POSITIVE_SCORE_THRESHOLD = .4 # minimum threshold to consider a NN prediction as positive

# general settings
MAX_PROCESSES = 8 # max number of cores to use for multiprocessing
