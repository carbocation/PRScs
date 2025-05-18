#!/usr/bin/env python3

"""
Parse the reference panel, summary statistics, and validation set.

"""


import os
import numpy as np
from scipy.stats import norm
from scipy import linalg
import h5py


def parse_ref(ref_file, chrom):
    print('... parse reference file: %s ...' % ref_file)

    ref_dict = {'CHR':[], 'SNP':[], 'BP':[], 'A1':[], 'A2':[], 'MAF':[]}
    with open(ref_file) as ff:
        header = next(ff)
        for line in ff:
            ll = (line.strip()).split()
            if int(ll[0]) == chrom:
                ref_dict['CHR'].append(chrom)
                ref_dict['SNP'].append(ll[1])
                ref_dict['BP'].append(int(ll[2]))
                ref_dict['A1'].append(ll[3])
                ref_dict['A2'].append(ll[4])
                ref_dict['MAF'].append(float(ll[5]))

    print('... %d SNPs on chromosome %d read from %s ...' % (len(ref_dict['SNP']), chrom, ref_file))
    return ref_dict


def parse_bim(bim_file, chrom):
    print('... parse bim file: %s ...' % (bim_file + '.bim'))

    vld_dict = {'SNP':[], 'A1':[], 'A2':[]}
    with open(bim_file + '.bim') as ff:
        for line in ff:
            ll = (line.strip()).split()
            if int(ll[0]) == chrom:
                vld_dict['SNP'].append(ll[1])
                vld_dict['A1'].append(ll[4])
                vld_dict['A2'].append(ll[5])

    print('... %d SNPs on chromosome %d read from %s ...' % (len(vld_dict['SNP']), chrom, bim_file + '.bim'))
    return vld_dict


def parse_sumstats(ref_dict, vld_dict, sst_file, n_subj):
    print('... parse sumstats file: %s ...' % sst_file)

    ATGC = ['A', 'T', 'G', 'C']
    sst_dict = {'SNP':[], 'A1':[], 'A2':[]}
    with open(sst_file) as ff:
        header = next(ff)
        for line in ff:
            ll = (line.strip()).split()
            # if ll[1] in ATGC and ll[2] in ATGC: # By removing this check, we keep every variant including indels
            sst_dict['SNP'].append(ll[0])
            sst_dict['A1'].append(ll[1])
            sst_dict['A2'].append(ll[2])

    print('... %d SNPs read from %s ...' % (len(sst_dict['SNP']), sst_file))


    mapping = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    def comp(allele):
        """Return Watson–Crick complement for A/T/G/C; leave any longer allele unchanged."""
        return mapping.get(allele, allele)

    vld_snp = set(zip(vld_dict['SNP'], vld_dict['A1'], vld_dict['A2']))

    # ----- for the reference panel -----
    ref_snp = (
        set(zip(ref_dict['SNP'], ref_dict['A1'], ref_dict['A2'])) |
        set(zip(ref_dict['SNP'], ref_dict['A2'], ref_dict['A1'])) |
        set(zip(ref_dict['SNP'], [comp(a) for a in ref_dict['A1']],
                            [comp(a) for a in ref_dict['A2']])) |
        set(zip(ref_dict['SNP'], [comp(a) for a in ref_dict['A2']],
                            [comp(a) for a in ref_dict['A1']]))
    )

    # ----- for the summary-statistics file -----
    sst_snp = (
        set(zip(sst_dict['SNP'], sst_dict['A1'], sst_dict['A2'])) |
        set(zip(sst_dict['SNP'], sst_dict['A2'], sst_dict['A1'])) |
        set(zip(sst_dict['SNP'], [comp(a) for a in sst_dict['A1']],
                            [comp(a) for a in sst_dict['A2']])) |
        set(zip(sst_dict['SNP'], [comp(a) for a in sst_dict['A2']],
                            [comp(a) for a in sst_dict['A1']]))
    )

    comm_snp = vld_snp & ref_snp & sst_snp

    print('... %d common SNPs in the reference, sumstats, and validation set ...' % len(comm_snp))


    n_sqrt = np.sqrt(n_subj)
    sst_eff = {}
    with open(sst_file) as ff:
        header = (next(ff).strip()).split()
        header = [col.upper() for col in header]
        for line in ff:
            ll = (line.strip()).split()
            snp = ll[0]; a1 = ll[1]; a2 = ll[2]
            # Commented out to permit indels
            # if a1 not in ATGC or a2 not in ATGC:
            #     continue
            if (snp, a1, a2) in comm_snp or (snp, comp(a1), comp(a2)) in comm_snp:
                if 'BETA' in header:
                    beta = float(ll[3])
                elif 'OR' in header:
                    beta = np.log(float(ll[3]))

                if 'SE' in header:
                    se = float(ll[4])
                    beta_std = beta/se/n_sqrt
                elif 'P' in header:
                    p = max(float(ll[4]), 1e-323)
                    beta_std = np.sign(beta)*abs(norm.ppf(p/2.0))/n_sqrt

                sst_eff.update({snp: beta_std})

            elif (snp, a2, a1) in comm_snp or (snp, comp(a2), comp(a1)) in comm_snp:
                if 'BETA' in header:
                    beta = float(ll[3])
                elif 'OR' in header:
                    beta = np.log(float(ll[3]))

                if 'SE' in header:
                    se = float(ll[4])
                    beta_std = -1*beta/se/n_sqrt
                elif 'P' in header:
                    p = max(float(ll[4]), 1e-323)
                    beta_std = -1*np.sign(beta)*abs(norm.ppf(p/2.0))/n_sqrt

                sst_eff.update({snp: beta_std})


    sst_dict = {'CHR':[], 'SNP':[], 'BP':[], 'A1':[], 'A2':[], 'MAF':[], 'BETA':[], 'FLP':[]}
    for (ii, snp) in enumerate(ref_dict['SNP']):
        if snp in sst_eff:
            sst_dict['SNP'].append(snp)
            sst_dict['CHR'].append(ref_dict['CHR'][ii])
            sst_dict['BP'].append(ref_dict['BP'][ii])
            sst_dict['BETA'].append(sst_eff[snp])

            a1 = ref_dict['A1'][ii]; a2 = ref_dict['A2'][ii]
            if (snp, a1, a2) in comm_snp:
                sst_dict['A1'].append(a1)
                sst_dict['A2'].append(a2)
                sst_dict['MAF'].append(ref_dict['MAF'][ii])
                sst_dict['FLP'].append(1)
            elif (snp, a2, a1) in comm_snp:
                sst_dict['A1'].append(a2)
                sst_dict['A2'].append(a1)
                sst_dict['MAF'].append(1-ref_dict['MAF'][ii])
                sst_dict['FLP'].append(-1)
            elif (snp, comp(a1), comp(a2)) in comm_snp:
                sst_dict['A1'].append(comp(a1))
                sst_dict['A2'].append(comp(a2))
                sst_dict['MAF'].append(ref_dict['MAF'][ii])
                sst_dict['FLP'].append(1)
            elif (snp, comp(a2), comp(a1)) in comm_snp:
                sst_dict['A1'].append(comp(a2))
                sst_dict['A2'].append(comp(a1))
                sst_dict['MAF'].append(1-ref_dict['MAF'][ii])
                sst_dict['FLP'].append(-1)

    return sst_dict


def parse_ldblk(ldblk_dir, sst_dict, chrom):
    print('... parse reference LD on chromosome %d ...' % chrom)

    if '1kg' in os.path.basename(ldblk_dir):
        chr_name = ldblk_dir + '/ldblk_1kg_chr' + str(chrom) + '.hdf5'
    elif 'ukbb' in os.path.basename(ldblk_dir):
        chr_name = ldblk_dir + '/ldblk_ukbb_chr' + str(chrom) + '.hdf5'
    elif 'ld_matrix' in os.path.basename(ldblk_dir):
        chr_name = ldblk_dir + '/chr' + str(chrom) + '.hdf5'

    hdf_chr = h5py.File(chr_name, 'r')
    n_blk = len(hdf_chr)
    ld_blk = [np.array(hdf_chr['blk_'+str(blk)]['ldblk']) for blk in range(1,n_blk+1)]

    snp_blk = []
    for blk in range(1,n_blk+1):
        snp_blk.append([bb.decode("UTF-8") for bb in list(hdf_chr['blk_'+str(blk)]['snplist'])])

    blk_size = []
    mm = 0
    for blk in range(n_blk):
        idx = [ii for (ii, snp) in enumerate(snp_blk[blk]) if snp in sst_dict['SNP']]
        blk_size.append(len(idx))
        if idx != []:
            idx_blk = range(mm,mm+len(idx))
            flip = [sst_dict['FLP'][jj] for jj in idx_blk]
            ld_blk[blk] = ld_blk[blk][np.ix_(idx,idx)]*np.outer(flip,flip)

            _, s, v = linalg.svd(ld_blk[blk])
            h = np.dot(v.T, np.dot(np.diag(s), v))
            ld_blk[blk] = (ld_blk[blk]+h)/2            

            mm += len(idx)
        else:
            ld_blk[blk] = np.array([])

    return ld_blk, blk_size


