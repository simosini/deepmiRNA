#################################################################################################
# This file contains the functions to be used to compute the site accessibility of the mRNA.
# In order to compute this value a region of size equal to the FOLDING_CHUNK_LEN parameter is
# extracted from the 3'UTR and passed to RNA to compute the opening energy. Site accessibility is
# then calculated as the difference between the free energy released by the bond and the energy
# needed to unfold the mRNA (opening energy). The bigger the value the more accessible the site.
# Only folds with high site accessibility (i.e. above the given threshold) are kept.
#################################################################################################

from subprocess import check_output
import re
import shlex
import RNA

import deepmirna.globs as gv

__author__ = "simosini"
__copyright__ = "simosini"
__license__ = "mit"

def site_accessibility_energy(folding_chunk, site_start, site_end):
    """
    compute the site accessibility of the folded mRNA
    :param folding_chunk: the folding chunk surrounding the potential binding site 
    :param site_start: the starting idx inside the folding chunk of the potential binding site
    :param site_end: the ending idx inside the folding chunk of the potential binding site
    :return: the site accessibility energy of the folding chunk
    """

    # compute site regular free energy
    reg_nrg = RNA.cofold(folding_chunk)[1]

    # build constraint: a constraint has the form .....xxxxx.....
    constraint =  '.' * site_start + 'x' * (site_end - site_start) + '.' * (len(folding_chunk) - site_end)
    input_str = '\n'.join([folding_chunk, constraint, '@\n'])

    # call subprocess to compute opening free energy
    cmd = 'RNAsubopt -C --noconv --temp=37 -e 0'
    output = check_output(shlex.split(cmd), input=input_str, encoding='ascii')

    # return result
    opening_nrg = float(re.search("[+-]?\d+\.\d+", output).group(0))
    return  reg_nrg - opening_nrg

def create_folding_chunk(chunk_start_idx, threeutr_transcript):
    """
    extracts the region surrounding the MBS to compute its site accessibility 
    :param chunk_start_idx: the starting index of the chunk wrt the beginning of the 3UTR 
    :param threeutr_transcript: the whole transcript of the 3UTR of the gene
    :return: the folding chunk and the coordinates of the mbs inside it wrt the beginning of the chunk
    """

    # additional nucleotides to consider when computing opening energy
    # i.e if the length of the folding chunk needed to compute the opening
    # energy is 200 and the binding site length is 30 (as with default values)
    # we need to consider 85 nucleotides before and 85 nucleotides after
    # the binding site itself. Remember that the
    # folding chunk is the sequence surrounding the potential binding site.
    # Whenever there are not enough nucleotides (i.e. near the beginning and the end of the 3'UTR)
    # the folding chunk computed will be shorter

    chunk_len = gv.FOLDING_CHUNK_LEN
    mbs_len = gv.MBS_LEN
    fc_additional_nts = (chunk_len - mbs_len) // 2

    # prepare indexes to check for opening energy
    fc_start = max(0, chunk_start_idx - fc_additional_nts)
    fc_end = min(len(threeutr_transcript), chunk_start_idx + mbs_len + fc_additional_nts)
    fc = threeutr_transcript[fc_start:fc_end]
    mbs_start = min(fc_additional_nts, chunk_start_idx)
    mbs_end = mbs_start + mbs_len
    return fc, mbs_start, mbs_end