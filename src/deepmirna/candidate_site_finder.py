#################################################################################################
# This file scans the 3'UTR of a given gene to find potential candidate sites to be passed to the
# neural network for evaluation. When the 3'UTR is particularly big the sequence is split in chunks
# and the candidates are searched using multiprocessing. The max number of cores to use is defined
# in the configuration file. A site is a candidate if has at least MIN_COMPLEMENTARY_NUCLEOTIDES
# and if the free energy released by the reaction with the given miRNA is below the
# FREE_ENERGY_THRESHOLD.
#################################################################################################

from multiprocessing import Process, Queue

from math import floor
import RNA
from Bio.Seq import Seq

from deepmirna.cython_lcs import lcs
import deepmirna.globs as gv

__author__ = "simosini"
__copyright__ = "simosini"
__license__ = "mit"

# filter 1 fast : cythonized lcs for complementarity
def is_complement(mirna_seed, site_transcript):
    """
    a site_transcript is a complement if at least min_pairs pairs are complementary to the seed region of the miRNA
    :param mirna_seed: the extended seed region of the miRNA 
    :param site_transcript: the potential binding site to check  
    :return: True if at least MIN_COMPLEMENTARY_NUCLEOTIDES complementary miRNA seed nucleotides 
             are found in the site_transcript, False otherwise 
    """

    seed_len = len(mirna_seed)
    min_nucleotides = gv.MIN_COMPLEMENTARY_NUCLEOTIDES

    # reverse complement the miRNA seed for complementarity
    mirna_seed = str(Seq(mirna_seed.replace('U', 'T')).reverse_complement())


    for start in range(len(site_transcript) - seed_len + 1):
        if lcs(mirna_seed, site_transcript[start:start+seed_len]) >= min_nucleotides:
            return True
    return False

# filter 2 : duplex free energy
def free_energy(mirna_transcript, site_transcript):
    """
    compute free energy of the miRNA:MBS duplex
    :param mirna_transcript: the whole miRNA sequence
    :param site_transcript: the potential MBS sequence whose len is equal to the window size used
    :return: the free energy of the duplex computed using RNA Cofold software
    """

    input_str = '&'.join([site_transcript, mirna_transcript])
    _, mfe = RNA.cofold(input_str)
    return mfe

def find_candidates_from_chunk(mirna_transcript, chunk_transcript, chunk_start_idx, shared_queue):
    """
    This is the function executed by every worker in a multiprocess computation.
    given a chunk of the 3UTR finds all the candidate sites. If the 3'UTR is short the chunk can be 
    the whole 3'UTR sequence. Start index is important for site accessibility computation.
    :param mirna_transcript: the transcript of the miRNA
    :param chunk_transcript: the transcript of the chunk to analyze
    :param chunk_start_idx : the starting index of the chunk needed to compute the absolute starting index of the MBS
    :param shared_queue: the queue where to save the results
    :return: void, just save the resulting list to the shared queue
    """
    candidates_dict = dict()

    # extract seed region
    seed = mirna_transcript[gv.SEED_START:gv.SEED_END]

    # extract constants
    mbs_len = gv.MBS_LEN
    add_nucleotides = gv.FLANKING_NUCLEOTIDES_SIZE
    stride = gv.WINDOW_STRIDE
    mfe = gv.FREE_ENERGY_THRESHOLD

    # size of the MBS with both side additional flunking nucleotides
    total_site_len = mbs_len + 2 * add_nucleotides

    for i in range(0, len(chunk_transcript) - mbs_len + 1, stride):
        # find mbs to check
        mbs = chunk_transcript[i:i + mbs_len]
        # check complementarity and free energy for stability of bond
        if is_complement(seed, mbs):
            fe = free_energy(mirna_transcript, mbs)
            if fe  <  mfe:
                # add upstream and downstream nucleotides to create the binding site to add
                start_idx = max(0, i - add_nucleotides)
                end_idx = min(start_idx + total_site_len, len(chunk_transcript))
                # update start_idx in case we are at the end of the transcript
                start_idx = end_idx - total_site_len
                candidates_dict[(chunk_start_idx + i)] = (chunk_transcript[start_idx:end_idx], fe)
    # save result to the shared queue
    shared_queue.put(candidates_dict)

def find_candidate_sites(mirna_transcript, three_utr_transcript):
    """
    finds all candidate binding sites according to the CSSM provided by the config file.
    The threeUTR is split between a certain number of processes according to its length. 
    The chunks created are overlapping to allow binding sites to be found 
    across 2 consecutive chunks.
    :param mirna_transcript: the miRNA transcript
    :param three_utr_transcript: the whole transcript of the 3UTR of the gene
    :return: a dict with all candidate sites found 
    """

    # shared queue to collect results from processes
    out_q = Queue()

    # keys indicate the starting index of the MBS inside the 3UTR, values are tuples
    # containing the MBS transcript and free energy computed
    # this configuration is needed to compute the site accessibility energy after the NN prediction
    candidate_sites_dict = dict()

    # prepare chunks for processes
    min_nucleotides = gv.MIN_COMPLEMENTARY_NUCLEOTIDES
    overlapping_nts =  min_nucleotides - 1
    min_chunk_len = gv.MIN_CHUNK_LEN
    gap =  min_chunk_len - overlapping_nts
    tot_site_len = len(three_utr_transcript)
    n_proc = floor((tot_site_len - 2 * min_nucleotides) / gap)

    # list to collect processes
    proc_pool = []

    # start processes
    for proc_id in range(n_proc):
        chunk_start = proc_id*gap
        chunk_end = tot_site_len if tot_site_len - chunk_start < 2 * min_chunk_len else chunk_start + min_chunk_len
        proc_chunk = three_utr_transcript[chunk_start : chunk_end]
        p = Process(target=find_candidates_from_chunk, args=(mirna_transcript, proc_chunk, chunk_start, out_q))
        p.start()
        proc_pool.append(p)

    # collect results from processes
    # we do know how many resulting dicts to expect
    for proc_id in range(n_proc):
        candidate_sites_dict.update(out_q.get())

    # wait for all processes to finish
    for p in proc_pool:
        p.join()

    # return result
    return candidate_sites_dict
