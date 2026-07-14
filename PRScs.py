#!/usr/bin/env python3

"""
PRS-CS: a polygenic prediction method that infers posterior SNP effect sizes under continuous shrinkage (CS) priors
using GWAS summary statistics and an external LD reference panel.

Reference: T Ge, CY Chen, Y Ni, YCA Feng, JW Smoller. Polygenic Prediction via Bayesian Regression and Continuous Shrinkage Priors.
           Nature Communications, 10:1776, 2019.


Usage:
python PRScs.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX --sst_file=SUM_STATS_FILE --n_gwas=GWAS_SAMPLE_SIZE --out_dir=OUTPUT_DIR
                [--a=PARAM_A --b=PARAM_B --phi=PARAM_PHI --n_iter=MCMC_ITERATIONS --n_burnin=MCMC_BURNIN --thin=MCMC_THINNING_FACTOR
                 --chrom=CHROM --write_psi=WRITE_PSI --write_pst=WRITE_POSTERIOR_SAMPLES --seed=SEED
                 --backend=cpu|cuda|cuda-pcg --cuda_device=DEVICE --cuda_bucket_size=SIZE
                 --pcg_tol=TOL --pcg_maxiter=ITERATIONS --pcg_check_interval=ITERATIONS
                 --ld_diagnostics=TRUE|FALSE --ld_rank_tol=TOL
                 --psi_backend=cpu|cuda|cuda-raw|cuda-fused
                 --cuda_gig_max_rounds=ROUNDS
                 --profile=TRUE|FALSE]

"""


import os
import sys
import getopt

import parse_genet
import mcmc_gtb


def parse_param():
    long_opts_list = ['ref_dir=', 'bim_prefix=', 'sst_file=', 'a=', 'b=', 'phi=', 'n_gwas=',
                      'n_iter=', 'n_burnin=', 'thin=', 'out_dir=', 'chrom=', 'beta_std=', 'write_psi=', 'write_pst=', 'seed=', 'help']

    long_opts_list += [
        'backend=', 'cuda_device=', 'cuda_bucket_size=', 'profile=',
        'pcg_tol=', 'pcg_maxiter=', 'pcg_check_interval=',
        'ld_diagnostics=', 'ld_rank_tol=', 'psi_backend=',
        'cuda_gig_max_rounds=',
    ]

    param_dict = {'ref_dir': None, 'bim_prefix': None, 'sst_file': None, 'a': 1, 'b': 0.5, 'phi': None, 'n_gwas': None,
                  'n_iter': 1000, 'n_burnin': 500, 'thin': 5, 'out_dir': None, 'chrom': range(1,23),
                  'beta_std': 'FALSE', 'write_psi': 'FALSE', 'write_pst': 'FALSE', 'seed': None,
                  'backend': 'cpu', 'cuda_device': 0, 'cuda_bucket_size': 32, 'profile': 'FALSE',
                  'pcg_tol': 1e-10, 'pcg_maxiter': 100,
                  'pcg_check_interval': 4, 'ld_diagnostics': 'FALSE',
                  'ld_rank_tol': 1e-8, 'psi_backend': 'cpu',
                  'cuda_gig_max_rounds': 1000}

    print('\n')

    if len(sys.argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "h", long_opts_list)          
        except:
            print('Option not recognized.')
            print('Use --help for usage information.\n')
            sys.exit(2)

        for opt, arg in opts:
            if opt == "-h" or opt == "--help":
                print(__doc__)
                sys.exit(0)
            elif opt == "--ref_dir": param_dict['ref_dir'] = arg
            elif opt == "--bim_prefix": param_dict['bim_prefix'] = arg
            elif opt == "--sst_file": param_dict['sst_file'] = arg
            elif opt == "--a": param_dict['a'] = float(arg)
            elif opt == "--b": param_dict['b'] = float(arg)
            elif opt == "--phi": param_dict['phi'] = float(arg)
            elif opt == "--n_gwas": param_dict['n_gwas'] = int(arg)
            elif opt == "--n_iter": param_dict['n_iter'] = int(arg)
            elif opt == "--n_burnin": param_dict['n_burnin'] = int(arg)
            elif opt == "--thin": param_dict['thin'] = int(arg)
            elif opt == "--out_dir": param_dict['out_dir'] = arg
            elif opt == "--chrom": param_dict['chrom'] = arg.split(',')
            elif opt == "--beta_std": param_dict['beta_std'] = arg.upper()
            elif opt == "--write_psi": param_dict['write_psi'] = arg.upper()
            elif opt == "--write_pst": param_dict['write_pst'] = arg.upper()
            elif opt == "--seed": param_dict['seed'] = int(arg)
            elif opt == "--backend": param_dict['backend'] = arg.lower()
            elif opt == "--cuda_device": param_dict['cuda_device'] = int(arg)
            elif opt == "--cuda_bucket_size": param_dict['cuda_bucket_size'] = int(arg)
            elif opt == "--profile": param_dict['profile'] = arg.upper()
            elif opt == "--pcg_tol": param_dict['pcg_tol'] = float(arg)
            elif opt == "--pcg_maxiter": param_dict['pcg_maxiter'] = int(arg)
            elif opt == "--pcg_check_interval": param_dict['pcg_check_interval'] = int(arg)
            elif opt == "--ld_diagnostics": param_dict['ld_diagnostics'] = arg.upper()
            elif opt == "--ld_rank_tol": param_dict['ld_rank_tol'] = float(arg)
            elif opt == "--psi_backend": param_dict['psi_backend'] = arg.lower()
            elif opt == "--cuda_gig_max_rounds": param_dict['cuda_gig_max_rounds'] = int(arg)
    else:
        print(__doc__)
        sys.exit(0)

    if param_dict['ref_dir'] == None:
        print('* Please specify the directory to the reference panel using --ref_dir\n')
        sys.exit(2)
    elif param_dict['bim_prefix'] == None:
        print('* Please specify the directory and prefix of the bim file for the target dataset using --bim_prefix\n')
        sys.exit(2)
    elif param_dict['sst_file'] == None:
        print('* Please specify the summary statistics file using --sst_file\n')
        sys.exit(2)
    elif param_dict['n_gwas'] == None:
        print('* Please specify the sample size of the GWAS using --n_gwas\n')
        sys.exit(2)
    elif param_dict['out_dir'] == None:
        print('* Please specify the output directory using --out_dir\n')
        sys.exit(2)
    elif param_dict['backend'] not in ('cpu', 'cuda', 'cuda-pcg'):
        print('* --backend must be cpu, cuda or cuda-pcg\n')
        sys.exit(2)
    elif param_dict['cuda_device'] < 0:
        print('* --cuda_device must be non-negative\n')
        sys.exit(2)
    elif param_dict['cuda_bucket_size'] < 1:
        print('* --cuda_bucket_size must be at least 1\n')
        sys.exit(2)
    elif param_dict['profile'] not in ('TRUE', 'FALSE'):
        print('* --profile must be True or False\n')
        sys.exit(2)
    elif not 0 < param_dict['pcg_tol'] < 1:
        print('* --pcg_tol must be between 0 and 1\n')
        sys.exit(2)
    elif param_dict['pcg_maxiter'] < 1:
        print('* --pcg_maxiter must be at least 1\n')
        sys.exit(2)
    elif param_dict['pcg_check_interval'] < 1:
        print('* --pcg_check_interval must be at least 1\n')
        sys.exit(2)
    elif param_dict['ld_diagnostics'] not in ('TRUE', 'FALSE'):
        print('* --ld_diagnostics must be True or False\n')
        sys.exit(2)
    elif not 0 <= param_dict['ld_rank_tol'] < 1:
        print('* --ld_rank_tol must be in [0, 1)\n')
        sys.exit(2)
    elif param_dict['psi_backend'] not in (
            'cpu', 'cuda', 'cuda-raw', 'cuda-fused'):
        print(
            '* --psi_backend must be cpu, cuda, cuda-raw or cuda-fused\n'
        )
        sys.exit(2)
    elif param_dict['cuda_gig_max_rounds'] < 1:
        print('* --cuda_gig_max_rounds must be at least 1\n')
        sys.exit(2)

    for key in param_dict:
        print('--%s=%s' % (key, param_dict[key]))

    print('\n')
    return param_dict


def main():
    param_dict = parse_param()

    for chrom in param_dict['chrom']:
        print('##### process chromosome %d #####' % int(chrom))

        if '1kg' in os.path.basename(param_dict['ref_dir']):
            ref_dict = parse_genet.parse_ref(param_dict['ref_dir'] + '/snpinfo_1kg_hm3', int(chrom))
        elif 'ukbb' in os.path.basename(param_dict['ref_dir']):
            ref_dict = parse_genet.parse_ref(param_dict['ref_dir'] + '/snpinfo_ukbb_hm3', int(chrom))

        vld_dict = parse_genet.parse_bim(param_dict['bim_prefix'], int(chrom))

        sst_dict = parse_genet.parse_sumstats(ref_dict, vld_dict, param_dict['sst_file'], param_dict['n_gwas'])

        need_ld_factors = param_dict['backend'] == 'cuda-pcg'
        if need_ld_factors:
            ld_blk, blk_size, ld_factors, ld_eigenvalues = \
                parse_genet.parse_ldblk(
                    param_dict['ref_dir'], sst_dict, int(chrom),
                    return_factors=True,
                )
        else:
            ld_blk, blk_size = parse_genet.parse_ldblk(
                param_dict['ref_dir'], sst_dict, int(chrom)
            )
            ld_factors = None
            ld_eigenvalues = None

        if param_dict['ld_diagnostics'] == 'TRUE':
            from beta_backend import diagnose_ld_blocks, format_ld_diagnostics
            diagnostics = diagnose_ld_blocks(
                ld_blk, blk_size,
                bucket_size=param_dict['cuda_bucket_size'],
                rank_rtol=param_dict['ld_rank_tol'],
                ld_eigenvalues=ld_eigenvalues,
            )
            print(format_ld_diagnostics(diagnostics))

        mcmc_gtb.mcmc(
            param_dict['a'], param_dict['b'], param_dict['phi'], sst_dict,
            param_dict['n_gwas'], ld_blk, blk_size, param_dict['n_iter'],
            param_dict['n_burnin'], param_dict['thin'], int(chrom),
            param_dict['out_dir'], param_dict['beta_std'],
            param_dict['write_psi'], param_dict['write_pst'],
            param_dict['seed'], backend=param_dict['backend'],
            cuda_device=param_dict['cuda_device'],
            cuda_bucket_size=param_dict['cuda_bucket_size'],
            profile=param_dict['profile'], pcg_tol=param_dict['pcg_tol'],
            pcg_maxiter=param_dict['pcg_maxiter'],
            pcg_check_interval=param_dict['pcg_check_interval'],
            ld_rank_tol=param_dict['ld_rank_tol'],
            ld_factors=ld_factors, ld_eigenvalues=ld_eigenvalues,
            psi_backend=param_dict['psi_backend'],
            cuda_gig_max_rounds=param_dict['cuda_gig_max_rounds'],
        )

        print('\n')


if __name__ == '__main__':
    main()
