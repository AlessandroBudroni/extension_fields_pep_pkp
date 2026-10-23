# Section 5 experiments

## PKP-to-PSD corank experiment

`pkp_permutation_isd_corank_experiment.py` compares the binary matrix derived
from one PKP parameter set with a uniformly random binary matrix of the same
dimensions. It measures

- the invertibility probability `Pr[d(Q)=0]`;
- the average enumeration cost `E[2^d(Q)]`;
- the mean and maximum corank `d(Q)`.

Run it with SageMath:

```sh
sage pkp_permutation_isd_corank_experiment.py 64 32 1 13 \
  --r 8 \
  --instances 100 \
  --partitions 200 \
  --groupings adaptive \
  --skip-compatible \
  --seed 1
```

The positional arguments are `N`, `K`, `ell`, and `nu`, with `q=2^nu`.

## Corank scaling at fixed field size

`pkp_permutation_isd_corank_scaling.py` repeats the adaptive experiment for
growing `N` at a fixed `nu` and code rate. Infeasible lengths, for which no
admissible divisor `r` exists, are omitted automatically. Independent parameter
sets are run in parallel.

For the two experiments reported in the paper:

```sh
python3 pkp_permutation_isd_corank_scaling.py \
  --nu 13 \
  --instances 20 \
  --partitions 100 \
  --jobs 4 \
  --out-dir rank_scaling_nu13

python3 pkp_permutation_isd_corank_scaling.py \
  --nu 23 \
  --instances 20 \
  --partitions 100 \
  --jobs 4 \
  --out-dir rank_scaling_nu23
```

Each run produces:

- a full aggregate CSV, including the selected `K` and `r` and the fraction of
  instances for which an adaptive grouping was found;
- a whitespace-separated `.dat` table ready for the paper's PGFPlots figure;
- PNG previews of `Pr[d(Q)=0]` and `E[2^d(Q)]`;
- per-parameter raw data, summaries, rank diagnostics, and logs under `runs/`.

The defaults are `K/N=1/2`, `ell=1`, and `20 <= N <= 128`. Use `--n-values` to
request an explicit list of lengths.

## Complexity comparison with KMP

`plot_pkp_attacks_vs_nu.py` compares KMP, Esser--Santini, and the
permutation-based PSD attack for fixed PKP parameters and growing field size.
The Esser--Santini baseline applies regular ISD directly to the PSD instance
obtained from the PKP-to-PSD transformation:

```sh
python3 plot_pkp_attacks_vs_nu.py 64 37 1 2 60 \
  --out-csv data/comparison_N64_K37_L1_nu2to60.csv \
  --out-png data/comparison_N64_K37_L1_nu2to60.png
```

`plot_pkp_attacks_vs_n.py` fixes `q=2^11` and `ell=1`, lets `N` grow from 20 to
160, and selects the largest `K` for which the expected number of PKP solutions
is below one:

```sh
python3 plot_pkp_attacks_vs_n.py \
  --n-min 20 \
  --n-max 160 \
  --n-step 1 \
  --nu 11 \
  --ell 1 \
  --out-csv data/pkp_attacks_vs_n_N20to160_step1_q2pow11_ell1.csv \
  --out-png data/pkp_attacks_vs_n_N20to160_step1_q2pow11_ell1.png
```

Both commands output a CSV file and a PNG preview. They require Python 3 and
Matplotlib; `kmp.py` and `psd_attack_complexities.py` are shared helper modules.

### Multiplicity-aware KMP comparison (SPECK)

Both plots show **KMP (the SPECK variant), ES and PSD**. The old KMP estimator
and random-guessing estimates remain available in the API and exported data for
diagnostics, but are not plotted. ES and PSD values are unchanged. The scalar-PKP estimator
implements SPECK (ePrint 2025/923), Proposition 2 and Section 6.2, Lemma 2:

- For input symbol counts `m`, count distinct partial arrangements `A_t(m)`
  exactly by integer dynamic programming.
- Estimate the number of distinct solution vectors as
  `S = 1 + (A_n(m)-1)/q^(n-k)`.
- For each matching length `t=1,...,n-k`, split `k+t` positions into two
  balanced parts, reduce both arrangement lists by `sqrt(S)`, and estimate
  their creation and collision costs. The plotted exponent is the minimum
  over `t` of the maximum of these three terms. The API also returns the
  log of their sum, as in Lemma 2. Reduced list sizes are bounded below by one
  at exponent precision. These are heuristic estimates, not measured timings.
- Random guessing costs `A_n(m)/S` trials in the same model. In a dense
  solution regime this reproduces `q^(n-k)`, e.g. `2^54` for `(64,37,4)`.

Since the plots specify parameters rather than an input vector, the KMP curve
shows the **arithmetic mean estimated work** over 100 uniformly sampled nonzero
vectors, with seed 1. This is `log2(mean(T))`, not `mean(log2(T))`. Every vector
is treated as a separate fixed input: no favorable-codeword selection or
resampling attack is included. Adjust with `--profile-samples` and
`--profile-seed`. CSV files also record the minimum and maximum sampled SPECK
exponents and the sampling settings. The ES/PSD curves keep their original
solution-count models; this addition does not validate their independence
assumptions in the repeated-entry regime.

The SPECK formula currently supports `ell=1`; other `ell` values produce
`nan` for this estimate and no KMP curve, rather than an unsupported
multidimensional extrapolation. The legacy KMP estimates are still exported.

For an exact supplied profile, use the Python API (no averaging):

```python
from kmp import speck_complexity
estimate = speck_complexity(n=64, k=37, q=4, multiplicities=(16, 16, 16, 16))
print(estimate)
```

`k` here is the code dimension. The legacy `kmp_complexity(n,r,l,q)` API uses
`r=n-k`; its historical CLI adds one to its positional `r`, so that CLI expects
`r=n-k-1`. Existing behavior is retained. For example:

```sh
python3 kmp.py 64 26 1 4 --speck --multiplicities 16 16 16 16
```

To regenerate the data and previews with the SPECK variant of KMP, and export tables
for the separately formatted TikZ figure, run from this directory:

```sh
python3 plot_pkp_attacks_vs_nu.py 64 37 1 2 60 \
  --profile-samples 100 --profile-seed 1 \
  --out-csv data/comparison_N64_K37_L1_nu2to60.csv \
  --out-png data/comparison_N64_K37_L1_nu2to60.png \
  --out-dat data/comparison_N64_K37_L1_nu2to60.dat

python3 plot_pkp_attacks_vs_n.py \
  --profile-samples 100 --profile-seed 1 \
  --out-csv data/pkp_attacks_vs_n_N20to160_step1_q2pow11_ell1.csv \
  --out-png data/pkp_attacks_vs_n_N20to160_step1_q2pow11_ell1.png \
  --out-dat data/pkp_attacks_vs_n_N20to160_step1_q2pow11_ell1.dat
```

The `.dat` columns are `nu` (or `N`), `kmp`, `es`, `psd`, `speck`, and
`random_guess`. Copy the first table to
`figures/data/pkp_complexity_N64_K37_L1.txt` and the second to
`figures/data/pkp_complexity_q11_vs_n.txt` in the LaTeX project. The new
PGFPlots KMP curve uses `speck`; `kmp` and `random_guess` are retained only as diagnostics.

Run the exact small-instance checks with:

```sh
python3 -m unittest discover -s . -p test_kmp.py -v
```
