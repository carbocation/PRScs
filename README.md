# PRS-CS

**PRS-CS** is a Python based command line tool that infers posterior SNP effect sizes under continuous shrinkage (CS) priors
using GWAS summary statistics and an external LD reference panel.

- Details of the development and evaluation of PRS-CS are described in: \
  T Ge, CY Chen, Y Ni, YCA Feng, JW Smoller. Polygenic Prediction via Bayesian Regression and Continuous Shrinkage Priors. *Nature Communications*, 10:1776, 2019.

- An extension of PRS-CS to PRS-CSx for cross-population polygenic prediction is available at https://github.com/getian107/PRScsx and described in: \
  Y Ruan, YF Lin, YCA Feng, CY Chen, M Lam, Z Guo, Stanley Global Asia Initiatives, L He, A Sawa, AR Martin, S Qin, H Huang, T Ge. Improving polygenic prediction in ancestrally diverse populations. *Nature Genetics*, 54:573-580, 2022.

- A review of the methods and best practices for cross-ancestry polygenic prediction is available at: \
  L Kachuri, N Chatterjee, J Hirbo, DJ Schaid, I Martin, IJ Kullo, EE Kenny, B Pasaniuc, JS Witte, T Ge. Principles and methods for transferring polygenic risk scores across global populations. *Nature Reviews Genetics*, 25:8-25, 2024.


## Version History

**May 14, 2024**: Replaced some scipy functions with numpy due to changes in the latest scipy version.

**Apr 9, 2024**: Allowed for the output of all posterior samples, which can be used to estimate the uncertainty of individualized PRS.

🔴
**Aug 10, 2023**: Added BETA/OR + SE as a new input format (see the format of GWAS summary statistics below), which is now the recommended input data. When using BETA/OR + P as the input, p-values smaller than 1e-323 are truncated, which may reduce the prediction accuracy for traits that have highly significant loci.

**Aug 10, 2023**: Allowed for the output of variant-specific shrinkage estimates.

**Nov 3, 2022**: Import random module from numpy instead of scipy.

**Jun 4, 2021**: Expanded reference panels to five populations.

**May 26, 2021**: Added suggestions for limiting the number of threads in scipy when running PRS-CS (see Computational Efficiency section below).

**Apr 6, 2021**: Added projection of the LD matrix to its nearest non-negative definite matrix.

**Mar 4, 2021**: LD reference panels constructed using the UK Biobank data are now available. 

**Jan 4, 2021**: Improved the accuracy and robustness of random sampling from the generalized inverse Gaussian distribution. Prediction accuracy will probably slightly improve over previous versions.

**Sept 10, 2020**: Fixed a bug in strand flip when there are non-ATGC alleles (e.g., indels) in the GWAS summary statistics. Previous versions erroneously remove variants that can be matched across GWAS summary statistics, the reference panel and the validation bim file via strand flip, which reduces the number of SNPs used in prediction and may slightly affect prediction accuracy. 

**Apr 24, 2020**: Accounted for a rare ZeroDivisionError in MCMC sampling.

**Apr 20, 2020**: Added non-ATGC allele check.

**Apr 11, 2020**: Added strand flip check.

**Mar 25, 2020**: Minor changes to make the software Python 2 and 3 compatible.

**Oct 20, 2019**: Added `--seed`, which can be used to seed the random number generator using a non-negative integer.

**Jun 6, 2019**: Fixed a bug in `--beta_std`. If you explicitly specified `--beta_std=False`, the output was actually standardized beta (in contrast to desired per-allele beta) and we recommend rerunning the analysis. If you left `--beta_std` as default or used `--beta_std=True`, the results were not affected.


## Getting Started

- Clone this repository using the following git command:
   
    `git clone https://github.com/getian107/PRScs.git`

    Alternatively, download the source files from the github website (`https://github.com/getian107/PRScs`)

- Download the LD reference panels and extract files:

    LD reference panels constructed using the 1000 Genomes Project phase 3 samples:
    
     [AFR reference](https://www.dropbox.com/s/mq94h1q9uuhun1h/ldblk_1kg_afr.tar.gz?dl=0 "AFR reference") (~4.44G);
     `tar -zxvf ldblk_1kg_afr.tar.gz`
     
     [AMR reference](https://www.dropbox.com/s/uv5ydr4uv528lca/ldblk_1kg_amr.tar.gz?dl=0 "AMR reference") (~3.84G);
     `tar -zxvf ldblk_1kg_amr.tar.gz`
        
     [EAS reference](https://www.dropbox.com/s/7ek4lwwf2b7f749/ldblk_1kg_eas.tar.gz?dl=0 "EAS reference") (~4.33G);
     `tar -zxvf ldblk_1kg_eas.tar.gz`
        
     [EUR reference](https://www.dropbox.com/s/mt6var0z96vb6fv/ldblk_1kg_eur.tar.gz?dl=0 "EUR reference") (~4.56G);
     `tar -zxvf ldblk_1kg_eur.tar.gz`
     
     [SAS reference](https://www.dropbox.com/s/hsm0qwgyixswdcv/ldblk_1kg_sas.tar.gz?dl=0 "SAS reference") (~5.60G);
     `tar -zxvf ldblk_1kg_sas.tar.gz`
    
    LD reference panels constructed using the UK Biobank data ([Notes](https://www.dropbox.com/s/y3hsc15kwjxwjtd/UKBB_ref.txt?dl=0 "Notes")):
    
     [AFR reference](https://www.dropbox.com/s/dtccsidwlb6pbtv/ldblk_ukbb_afr.tar.gz?dl=0 "AFR reference") (~4.93G);
     `tar -zxvf ldblk_ukbb_afr.tar.gz`
     
     [AMR reference](https://www.dropbox.com/s/y7ruj364buprkl6/ldblk_ukbb_amr.tar.gz?dl=0 "AMR reference") (~4.10G);
     `tar -zxvf ldblk_ukbb_amr.tar.gz`
    
     [EAS reference](https://www.dropbox.com/s/fz0y3tb9kayw8oq/ldblk_ukbb_eas.tar.gz?dl=0 "EAS reference") (~5.80G);
     `tar -zxvf ldblk_ukbb_eas.tar.gz`
    
     [EUR reference](https://www.dropbox.com/s/t9opx2ty6ucrpib/ldblk_ukbb_eur.tar.gz?dl=0 "EUR reference") (~6.25G);
     `tar -zxvf ldblk_ukbb_eur.tar.gz`
    
     [SAS reference](https://www.dropbox.com/s/nto6gdajq8qfhh0/ldblk_ukbb_sas.tar.gz?dl=0 "SAS reference") (~7.37G);
     `tar -zxvf ldblk_ukbb_sas.tar.gz`
     
    For regions that don't have access to Dropbox, reference panels can be downloaded from the
    [alternative download site](https://personal.broadinstitute.org/hhuang//public//PRS-CSx/Reference).

- PRScs requires Python packages **scipy** (https://www.scipy.org/), **h5py** (https://www.h5py.org/) and **numba** (https://numba.pydata.org/) installed.
 
- Once Python and its dependencies have been installed, running

    `./PRScs.py --help` or `./PRScs.py -h`

    will print a list of command-line options.


## Using PRS-CS

`
python PRScs.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX --sst_file=SUM_STATS_FILE --n_gwas=GWAS_SAMPLE_SIZE --out_dir=OUTPUT_DIR [--a=PARAM_A --b=PARAM_B --phi=PARAM_PHI --n_iter=MCMC_ITERATIONS --n_burnin=MCMC_BURNIN --thin=MCMC_THINNING_FACTOR --chrom=CHROM --beta_std=BETA_STD --write_psi=WRITE_PSI --write_pst=WRITE_POSTERIOR_SAMPLES --seed=SEED --backend=cpu|cuda|cuda-direct|cuda-fused-solve|cuda-pcg --cuda_device=DEVICE --cuda_bucket_size=SIZE --pcg_tol=TOL --pcg_maxiter=ITERATIONS --pcg_check_interval=ITERATIONS --ld_diagnostics=TRUE|FALSE --ld_rank_tol=TOL --psi_backend=cpu|cuda|cuda-raw|cuda-fused --cuda_gig_max_rounds=ROUNDS --profile=TRUE|FALSE]
`
 - PATH_TO_REFERENCE (required): Full path (including folder name) to the directory that contains information on the LD reference panel (the snpinfo file and hdf5 files). If the 1000 Genomes reference panel is used, folder name would be `ldblk_1kg_afr`, `ldblk_1kg_amr`, `ldblk_1kg_eas`, `ldblk_1kg_eur` or `ldblk_1kg_sas`; if the UK Biobank reference panel is used, folder name would be `ldblk_ukbb_afr`, `ldblk_ukbb_amr`, `ldblk_ukbb_eas`, `ldblk_ukbb_eur` or `ldblk_ukbb_sas`. Note that the reference panel should match the ancestry of the GWAS sample (not the target sample).

 - VALIDATION_BIM_PREFIX (required): Full path and the prefix of the bim file for the target (validation/testing) dataset. This file is used to provide a list of SNPs that are available in the target dataset.

 - SUM_STATS_FILE (required): Full path and the file name of the GWAS summary statistics. The summary statistics file must include either BETA/OR + SE or BETA/OR + P. When using BETA/OR + SE as the input, the file must have the following format (including the header line):

```
    SNP          A1   A2   BETA      SE
    rs4970383    C    A    -0.0064   0.0090
    rs4475691    C    T    -0.0145   0.0094
    rs13302982   A    G    -0.0232   0.0199
    ...
```
Or:
```
    SNP          A1   A2   OR        SE
    rs4970383    A    C    0.9825    0.0314                 
    rs4475691    T    C    0.9436    0.0319
    rs13302982   A    G    1.1337    0.0543
    ...
```
where SNP is the rs ID, A1 is the effect allele, A2 is the alternative allele, BETA/OR is the effect/odds ratio of the A1 allele, SE is the standard error of the effect. Note that when OR is used, SE corresponds to the standard error of logOR.

When using BETA/OR + P as the input, the file must have the following format (including the header line):

```
    SNP          A1   A2   BETA      P
    rs4970383    C    A    -0.0064   0.4778
    rs4475691    C    T    -0.0145   0.1245
    rs13302982   A    G    -0.0232   0.2429
    ...
```
Or:
```
    SNP          A1   A2   OR        P
    rs4970383    A    C    0.9825    0.5737                 
    rs4475691    T    C    0.9436    0.0691
    rs13302982   A    G    1.1337    0.0209
    ...
```
where SNP is the rs ID, A1 is the effect allele, A2 is the alternative allele, BETA/OR is the effect/odds ratio of the A1 allele, P is the p-value of the effect. Here, a standardized effect size is calculated using the p-value while BETA/OR is only used to determine the direction of an association. Therefore if z-scores or even +1/-1 indicating effect directions are presented in the BETA column, the algorithm should still work properly.

 - GWAS_SAMPLE_SIZE (required): Sample size of the GWAS.

 - OUTPUT_DIR (required): Output directory and output filename prefix of the posterior effect size estimates.

 - PARAM_A (optional): Parameter a in the gamma-gamma prior. Default is 1.

 - PARAM_B (optional): Parameter b in the gamma-gamma prior. Default is 0.5.

 - PARAM_PHI (optional): Global shrinkage parameter phi. If phi is not specified, it will be learnt from the data using a fully Bayesian approach. This usually works well for polygenic traits with large GWAS sample sizes (hundreds of thousands of subjects). For GWAS with limited sample sizes (including most of the current disease GWAS), fixing phi to 1e-2 (for highly polygenic traits) or 1e-4 (for less polygenic traits), or doing a small-scale grid search (e.g., phi=1e-6, 1e-4, 1e-2, 1) to find the optimal phi value in the validation dataset often improves perdictive performance.

 - MCMC_ITERATIONS (optional): Total number of MCMC iterations. Default is 1,000.

 - MCMC_BURNIN (optional): Number of burnin iterations. Default is 500.

 - MCMC_THINNING_FACTOR (optional): Thinning factor of the Markov chain. Default is 5.

 - CHROM (optional): The chromosome(s) on which the model is fitted, separated by comma, e.g., `--chrom=1,3,5`. Parallel computation for the 22 autosomes is recommended. Default is iterating through 22 autosomes (can be time-consuming).

- BETA_STD (optional): If True, return standardized posterior SNP effect sizes (i.e., effect sizes corresponding to standardized genotypes with zero mean and unit variance across subjects). If False, return per-allele posterior SNP effect sizes, calculated by properly weighting the posterior standardized effect sizes using allele frequencies estimated from the reference panel. Default is False.

- WRITE_PSI (optional): If True, write variant-specific shrinkage estimates. Default is False.

- WRITE_POSTERIOR_SAMPLES (optional): If True, write all posterior samples of SNP effect sizes after thinning. Default is False.

- SEED (optional): Non-negative integer which seeds the random number generator.

- BACKEND (optional): Backend for the within-chromosome beta block update. `cpu` uses SciPy and is the default. `cuda` uses batched FP64 Cholesky through CuPy. `cuda-direct` invokes the same FP64 cuSOLVER and cuBLAS routines with preallocated workspaces. `cuda-pcg` uses experimental FP64 perturb-and-solve with batched preconditioned conjugate gradients. This option does not change how chromosomes are scheduled.

- DEVICE (optional): Zero-based CUDA device number. Default is 0.

- SIZE (optional): CUDA block-size bucketing quantum. LD blocks are padded to the next multiple of this value so similarly sized blocks can use batched factorizations. Default is 32. Smaller values reduce padding and GPU memory use; larger values may create fewer batches at the cost of additional cubic work.

- TOL (optional): Relative residual tolerance for `cuda-pcg`. Default is `1e-10`. A draw fails explicitly if its true residual exceeds this tolerance; inaccurate solves are not silently accepted.

- ITERATIONS (optional): Maximum PCG iterations, default 100, and the interval between device-to-host convergence checks, default 4.

- LD_DIAGNOSTICS (optional): If True, report real block sizes, padding overhead, eigenvalue bounds and numerical rank before MCMC. Default is False because the rank calculation is not free for non-PCG backends.

- LD_RANK_TOL (optional): Relative eigenvalue threshold used only to report effective LD rank. Default is `1e-8`; the FP64 PCG backend retains every non-negative eigencomponent and does not truncate at this threshold.

- PSI_BACKEND (optional): `cpu` uses the existing fused Numba GIG sampler and is the default. `cuda` uses a vectorized Devroye rejection sampler driven by CuPy's device RNG. Seeded runs are reproducible within a fixed backend but CPU and CUDA streams differ.

- ROUNDS (optional): Maximum vector rejection rounds for the CUDA GIG sampler. Default is 1000. The sampler fails explicitly instead of returning incomplete draws if this bound is reached.

- PROFILE (optional): If True, report warm-up and steady-state mean time spent in the beta update, `psi` update and remaining within-chromosome work. The first iteration is excluded from steady-state means. Default is False.


## Output

PRS-CS writes posterior SNP effect size estimates for each chromosome to the user-specified directory. The output file contains chromosome, rs ID, base position, A1, A2 and posterior effect size estimate for each SNP. An individual-level polygenic score can be produced by concatenating output files from all chromosomes and then using `PLINK`'s `--score` command (https://www.cog-genomics.org/plink/1.9/score). If polygenic scores are generated by chromosome, use the 'sum' modifier so that they can be combined into a genome-wide score.


## Computational Efficiency

PRS-CS relies on numpy packages, which automatically use all available cores on a compute node. This can be problematic when running PRS-CS on a compute cluster; PRS-CS jobs may interfere with other jobs running on the same node, reducing computational efficiency. To resolve this issue, including the following code in the script to specify the number of threads in scipy:

```
export MKL_NUM_THREADS=$N_THREADS
export NUMEXPR_NUM_THREADS=$N_THREADS
export OMP_NUM_THREADS=$N_THREADS
```
For example, to use a single thread for the computation, set `N_THREADS=1`.

### Experimental within-chromosome CUDA backends

The CUDA backends accelerate the beta block update within one chromosome. They deliberately do not provide chromosome-level scheduling: chromosomes can already be submitted as independent jobs on separate machines.

Install CuPy 14.1 or newer using the package that matches the machine's CUDA runtime. For example, a CUDA 12 installation typically uses:

```
pip install "cupy-cuda12x>=14.1"
```

Then add the backend options to an otherwise normal command:

```
python PRScs.py ... --chrom=22 --backend=cuda --cuda_device=0 --profile=True
```

`cuda` groups padded LD blocks and uses batched CuPy Cholesky and triangular solves. `cuda-direct` is an experimental equivalent path that preallocates the factorization and right-hand-side workspaces, builds device pointer arrays once, and invokes FP64 `potrfBatched` and `trsmBatched` directly. It avoids generic-wrapper copies while retaining the same dense Cholesky calculation:

```
python PRScs.py ... --chrom=22 --backend=cuda-direct --psi_backend=cuda-fused --profile=True
```

With `cuda-direct`, `--profile=True` also reports CUDA-event timings for precision-matrix assembly, Cholesky, both triangular solves, perturbation/scatter work, and host or synchronization overhead. These diagnostic events add a small amount of overhead and are intended for profiling rather than final production timing.

`cuda-fused-solve` retains the same FP64 batched Cholesky, then replaces both one-right-hand-side triangular-solve calls, Gaussian perturbation, quadratic-form reduction, and result scatter with one CUDA kernel:

```
python PRScs.py ... --chrom=22 --backend=cuda-fused-solve --psi_backend=cuda-fused --profile=True
```

`cuda-pcg` draws an exact Gaussian perturbation and solves the resulting precision systems with diagonally preconditioned FP64 conjugate gradients:

```
python PRScs.py ... --chrom=22 --backend=cuda-pcg --pcg_tol=1e-10 --pcg_maxiter=100 --ld_diagnostics=True --profile=True
```

The PCG formulation preserves the same conditional Gaussian target to the configured linear-solve tolerance. The PSD eigendecomposition already needed while parsing LD supplies the fixed square-root factor used to draw each perturbation. Exactly zero PSD components are omitted and factors are bucketed by both matrix size and rank. The reported numerical-rank threshold is diagnostic only; no positive eigenvalues are discarded.

LD matrices and marginal effects are transferred to the selected device once. Each MCMC iteration transfers one `psi` vector to CUDA and returns one beta vector plus the quadratic form. Unequal LD blocks are grouped by padded size. All calculations remain in double precision.

The independent CUDA GIG backend can be combined with either CUDA beta backend:

```
python PRScs.py ... --chrom=22 --backend=cuda-pcg --psi_backend=cuda --profile=True
```

`--psi_backend=cuda` uses vectorized CuPy operations and is the correctness fallback. The experimental `--psi_backend=cuda-raw` backend executes each draw's complete rejection loop in one CUDA kernel using independent counter-based Philox streams. It removes the repeated launches and host synchronizations of the vectorized version:

```
python PRScs.py ... --chrom=22 --backend=cuda --psi_backend=cuda-raw --profile=True
```

Both versions still transfer the beta and local-scale vectors once per iteration; they are intermediate steps toward keeping the complete MCMC state resident on the device.

`--psi_backend=cuda-fused` additionally generates the gamma-distributed `delta` update inside the same kernel before drawing `psi`. This removes the remaining CPU-sized random-vector generation while retaining the same conditional update sequence:

```
python PRScs.py ... --chrom=22 --backend=cuda --psi_backend=cuda-fused --profile=True
```

The CPU and CUDA backends use different random-number generators, so seeded runs are reproducible within a fixed backend configuration but are not expected to produce identical draws across backends. Validate posterior summaries rather than individual samples. CuPy does not guarantee an identical random stream across major CuPy versions.

GPU benefit depends strongly on the real LD block-size distribution and hardware FP64 throughput. Compare elapsed time for the same single chromosome after excluding the first iteration, which includes CUDA library initialization. The backend reports its number of size buckets and static resident GPU memory when the sampler starts; CUDA solver workspaces and the CuPy memory pool require additional memory. For PCG, also inspect the reported mean/max iteration count and maximum true residual. Cholesky remains the correctness and performance fallback when PCG needs too many iterations.

Run the self-contained 40,000-variant benchmark with:

```
python3 benchmark_gpu.py
```

The default comparison runs `cpu`, `cuda`, `cuda-direct`, `cuda-fused-solve`, and `cuda-pcg`. Use, for example, `--backends=cuda-direct,cuda-fused-solve --n-iter=100` for a longer dense-GPU comparison. Add `--psi-backend=cuda` to benchmark the vectorized CUDA GIG update, `--psi-backend=cuda-raw` for the single-kernel GIG implementation, or `--psi-backend=cuda-fused` to include `delta` generation in that kernel. The benchmark does not install or modify CUDA packages or drivers.


## Test Data

The test data contains GWAS summary statistics and a bim file for 1,000 SNPs on chromosome 22.
An example to use the test data:

`
python PRScs.py --ref_dir=path_to_ref/ldblk_1kg_eur --bim_prefix=path_to_bim/test --sst_file=path_to_sumstats/sumstats_se.txt --n_gwas=200000 --chrom=22 --phi=1e-2 --out_dir=path_to_output/eur
`


## Support

Please direct questions or bug reports to Tian Ge (tge1@mgh.harvard.edu).
