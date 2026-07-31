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
