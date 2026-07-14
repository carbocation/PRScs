#!/usr/bin/env python3

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""


import time

import numpy as np
import gigrnd
from beta_backend import make_beta_backend


def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin, chrom, out_dir, beta_std, write_psi, write_pst, seed,
         backend='cpu', cuda_device=0, cuda_bucket_size=32, profile='FALSE'):
    print('... MCMC ...')

    # seed
    if seed is not None:
        np.random.seed(seed)
        gigrnd.seed_rng(seed)

    # derived stats
    beta_mrg = np.array(sst_dict['BETA'], ndmin=2).T
    maf = np.array(sst_dict['MAF'], ndmin=2).T
    n_pst = int((n_iter-n_burnin)/thin)
    p = len(sst_dict['SNP'])

    # initialization
    beta = np.zeros((p,1))
    psi = np.ones((p,1))
    sigma = 1.0
    
    if phi == None:
        phi = 1.0; phi_updt = True
    else:
        phi_updt = False

    if write_pst == 'TRUE':
        beta_pst = np.zeros((p,n_pst))

    beta_est = np.zeros((p,1))
    psi_est = np.zeros((p,1))
    sigma_est = 0.0
    phi_est = 0.0

    beta_backend = make_beta_backend(
        backend, ld_blk, blk_size, beta_mrg, n,
        seed=seed,
        cuda_device=cuda_device,
        cuda_bucket_size=cuda_bucket_size,
    )
    print('... beta backend: %s ...' % beta_backend.describe())
    profile = str(profile).upper() == 'TRUE'
    profile_beta = 0.0
    profile_psi = 0.0
    profile_total = 0.0
    profile_iterations = 0

    # MCMC
    pp = 0
    for itr in range(1,n_iter+1):
        iteration_start = time.perf_counter()
        if itr % 100 == 0:
            print('--- iter-' + str(itr) + ' ---')

        beta_start = time.perf_counter()
        beta, quad = beta_backend.sample(psi, sigma)
        beta_elapsed = time.perf_counter() - beta_start

        s1 = float((beta * beta_mrg).sum())
        s2 = float((beta**2 / psi).sum())
        e1 = float(n/2.0*(1.0 - 2.0*s1 + quad))
        e2 = float(n/2.0*s2)
        err = max(e1, e2)

        # force sigma to be a Python float (not a 0-d array)
        sigma = float(1.0/np.random.gamma((n+p)/2.0, 1.0/err))

        delta = np.random.gamma(a+b, 1.0/(psi+phi))

        psi_start = time.perf_counter()
        gigrnd.gig_rvs_vec(
            psi[:, 0],
            float(a - 0.5),
            delta[:, 0],
            beta[:, 0],
            float(sigma),
            int(n),
        )
        psi_elapsed = time.perf_counter() - psi_start
        
        psi[psi>1] = 1.0

        if phi_updt == True:
            w = np.random.gamma(1.0, 1.0/(phi+1.0))
            phi = np.random.gamma(p*b+0.5, 1.0/(float(delta.sum())+w))

        # posterior
        if (itr>n_burnin) and (itr % thin == 0):
            beta_est = beta_est + beta/n_pst
            psi_est = psi_est + psi/n_pst
            sigma_est = sigma_est + sigma/n_pst
            phi_est = phi_est + phi/n_pst

            if write_pst == 'TRUE':
                beta_pst[:,[pp]] = beta
                pp += 1

        iteration_elapsed = time.perf_counter() - iteration_start
        if profile:
            if itr > 1:
                profile_beta += beta_elapsed
                profile_psi += psi_elapsed
                profile_total += iteration_elapsed
                profile_iterations += 1
            if itr == 1:
                print('[PROFILE chr%d] iter 1 warm-up: beta %.4fs, psi %.4fs, total %.4fs' %
                      (chrom, beta_elapsed, psi_elapsed, iteration_elapsed))
            elif itr % 10 == 0 or itr == n_iter:
                other = profile_total - profile_beta - profile_psi
                print('[PROFILE chr%d] steady-state mean over %d iter: beta %.4fs, psi %.4fs, other %.4fs, total %.4fs' %
                      (chrom, profile_iterations,
                       profile_beta/profile_iterations,
                       profile_psi/profile_iterations,
                       other/profile_iterations,
                       profile_total/profile_iterations))

    # convert standardized beta to per-allele beta
    if beta_std == 'FALSE':
        beta_est /= np.sqrt(2.0*maf*(1.0-maf))

        if write_pst == 'TRUE':
            beta_pst /= np.sqrt(2.0*maf*(1.0-maf))


    # write posterior effect sizes
    if phi_updt == True:
        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
    else:
        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)

    with open(eff_file, 'w') as ff:
        if write_pst == 'TRUE':
            for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_pst):
                ff.write(('%d\t%s\t%d\t%s\t%s' + '\t%.6e'*n_pst + '\n') % (chrom, snp, bp, a1, a2, *beta))
        else:
            for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_est):
                ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' %
                         (chrom, snp, bp, a1, a2, beta.item()))

    # write posterior estimates of psi
    if write_psi == 'TRUE':
        if phi_updt == True:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
        else:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)

        with open(psi_file, 'w') as ff:
            for snp, psi in zip(sst_dict['SNP'], psi_est):
                ff.write('%s\t%.6e\n' % (snp, psi.item()))

    # print estimated phi
    if phi_updt == True:
        print('... Estimated global shrinkage parameter: %1.2e ...' % phi_est )

    print('... Done ...')
