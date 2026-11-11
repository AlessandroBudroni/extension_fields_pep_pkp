# Section 5 experiments

Requires Python 3 and Matplotlib. Rank experiments use a C compiler by default;
pass `--backend python` to use the Python implementation.

## Section 5.3: rank statistics (Figure 1)

```sh
python3 counted_rank_scaling.py --nu 13 \
  --n-values 24 28 32 40 48 56 64 72 80 84 88 96 104 112 120 128 \
  --instances 20 --partitions 100 --jobs 4 --seed 1 \
  --out-dir data/counted_rank_scaling_nu13

python3 counted_rank_scaling.py --nu 23 \
  --n-values 24 36 48 52 60 72 84 96 104 108 120 \
  --instances 20 --partitions 100 --jobs 4 --seed 1 \
  --out-dir data/counted_rank_scaling_nu23
```

Outputs include CSV summaries, PGFPlots tables, PNG plots, raw ranks, and
records of whether independent source groups were found.

## Section 5.3: attack comparison (Figure 2)

```sh
python3 plot_counted_grouping_comparison.py \
  --nu-min 11 --nu-max 60 --n-min 20 --n-max 160 --surplus 3 --out-dir data
```

This compares KMP, Esser–Santini, and the construction of Section 5.2, implemented
in `counted_grouping.py`. It writes CSV and PGFPlots tables and PNG plots.

Run checks with `python3 -m unittest discover -v`.
