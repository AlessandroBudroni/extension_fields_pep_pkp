# Section 4 experiments

The experiments require SageMath with Matplotlib available (the reported data
were generated with SageMath 10.3). From the project root, enter this directory
before running the commands below:

```sh
cd section_4
sage --version
```

## Verification of Assumption 1

`verify_assumption_1.py` tests the rank of the binary matrix obtained from a
random PKP instance after adjoining the permutation equations. For each
instance it records

- the deficiency from the rank predicted in Assumption 1;
- the rank deficiency already present in the PKP-derived matrix;
- the missing rank contribution, if any, from the permutation equations.

By default, it fixes `k/n=1/2` and `ell=1` and, for every `n`, chooses

```text
nu = ceil((n-1)^2 / ((n-k)ell)),
```

which is the boundary of the polynomial-time regime in Section 4.

To reproduce the experiments, run the following

```sh
sage verify_assumption_1.py \
  --n-values 20 24 32 48 64 \
  --rate 1/2 \
  --ell 1 \
  --instances 100 \
  --jobs 4 \
  --seed 1 \
  --out-dir assumption_1_rate0.5

sage verify_assumption_1.py \
  --n-values 24 30 36 48 60 \
  --rate 1/3 \
  --ell 1 \
  --instances 100 \
  --jobs 4 \
  --seed 1 \
  --out-dir assumption_1_rate_0.3
```

The output directory contains raw and summary CSV files, a whitespace-separated
`.dat` table, a rank-deficiency scaling/tail plot, and a diagnostic plot. Use
`--nu` to fix one extension degree for every length or `--nu-offset` to test
points immediately above or below the boundary.

## Verification of Assumption 2

`verify_assumption_2.py` measures the deficiency

```text
D = min((n-k)k nu, n^2-d) - rank(Hbar),
```

for PEP-derived binary matrices, with `d=n+1` in the strictly
self-orthogonal case and `d=2n` in the self-dual case. The two modes are
selected by the required flags `--self-orthogonal` and `--self-dual`.

Unless `--nu` is supplied, each length is tested at the boundary

```text
nu = ceil((n^2-d) / ((n-k)k)).
```

To reproduce the experiments, run the following.

```sh
sage verify_assumption_2.py \
  --self-orthogonal \
  --n-values 24 30 36 48 60 \
  --rate 1/3 \
  --instances 100 \
  --jobs 4 \
  --seed 1 \
  --out-dir assumption_2_self_orthogonal_rate0.33

sage verify_assumption_2.py \
  --self-orthogonal \
  --n-values 20 24 32 48 64 \
  --rate 1/4 \
  --instances 100 \
  --jobs 4 \
  --seed 1 \
  --out-dir assumption_2_self_orthogonal_rate0.25

sage verify_assumption_2.py \
  --self-dual \
  --n-values 20 24 32 48 64 \
  --nu 5 \
  --instances 100 \
  --jobs 4 \
  --seed 1 \
  --out-dir assumption_2_self_dual_nu5
```

Each output directory contains raw and summary CSV files, a
whitespace-separated `.dat` table, a rank-deficiency scaling/tail plot, and a
kernel-dimension diagnostic plot. Use `--nu-offset` to move every point above
or below its boundary value.

## Polynomial-regime solvers

`solve_pkp_polynomial.py` implements the attack from the polynomial-time PKP
lemma. It descends the public tensor system to `GF(2)`, appends the permutation
equations, and enumerates the resulting affine solution space.

```sh
sage solve_pkp_polynomial.py 8 4 1 13 --instances 3 --seed 1
```

The positional parameters are `n k ell nu`. They must satisfy
`(n-k) ell nu >= (n-1)^2`.

`solve_pce_polynomial.py` implements the analogous attack against PCE/PEP with
strictly self-orthogonal or self-dual codes. It uses the fixed-row guessing
step from the proof to remove the known structural kernel before enumerating
the remaining affine solutions.

```sh
sage solve_pce_polynomial.py --self-orthogonal 12 4 5 \
  --instances 3 --seed 1

sage solve_pce_polynomial.py --self-dual 8 4 5 \
  --instances 3 --seed 1
```

The positional parameters are `n k nu`. They must satisfy
`(n-k) k nu > n^2`; self-dual mode additionally requires `k=n/2`.

Both scripts generate random instances and use the sampled secret only to
construct them. Recovery uses only the corresponding public matrices, and
every output permutation is verified against the original public instance.
Use `--max-kernel-dimension` to cap the affine space that the scripts will
enumerate.
