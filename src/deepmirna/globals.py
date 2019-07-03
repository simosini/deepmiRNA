#################################################################################################
# This file contains global variables to be used across all modules of the package. Parameters are
# initialized with default values, but they can be overwritten using the values contained in
# config.ini. By default all data used for computation are saved in the data folder contained
# in the project root directory.
#################################################################################################
# datasets settings
HEADER = ['gene_name', 'miRNA_name', 'functionality', '3UTR_transcript', 'mature_miRNA_transcript'] # common dataset header
TEST_SET_LOCATION = '...data/test_data.csv' # test set file path
TRAIN_SET_LOCATION = '...data/train_data.csv' # train set file path

# biological settings
MBS_LEN = 30 # maximum length for a potential binding site in the mRNA. Must be an even number
MAX_MIRNA_LEN = 30 # the maximum length of a miRNA sequence
SEED_START = 0 # index of the start of the extended seed region (usually 0 or 1) (inclusive)
SEED_END = 10 # index of the end of the extended seed region (usually 10) (exclusive)
MIN_COMPLEMENTARY_NUCLEOTIDES = 7 # minimum number of complementary nucleotides
FOLDING_CHUNK_LEN = 200 # length of the mRNA folding chunk to use to compute the needed opening energy
FLANKING_NUCLEOTIDES_SIZE = 5 # number of downstream and upstream nucleotides to consider for a binding sites
MIN_CHUNK_LEN = 200 # the minimum length a 3UTR chunk can have after splitting
WINDOW_STRIDE = 5 # the window step size when scanning the 3UTR of the gene (mRNA)
FREE_ENERGY_THRESHOLD = -10 # threshold to consider when filtering duplexes. Pairs with free energy greater than this threshold are filtered
SITE_ACCESSIBILITY_THRESHOLD = -14 # only folding chunks with an accessibility above this threshold are kept. The rest are filtered
MAX_PROCESSES = 8 # max number of cores to use
SCORE_THRESHOLD = .4 # minimum threshold to consider a NN prediction as positive

