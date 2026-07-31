#!/usr/bin/env python3
"""Compare PKP attacks while the PKP length grows.

For every N, choose the largest dimension K (equivalently, the smallest
codimension N-K) for which the expected number of PKP solutions is below one:

    N! / q^((N-K)*ell) < 1.

The attack complexities use the same formulas as
``plot_pkp_attacks_vs_nu.py``.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_pkp_attacks_vs_nu import compute_point
from psd_attack_complexities import log2_gamma_factorial


def largest_k_below_one(n: int, nu: int, ell: int) -> int:
    """Return the largest K such that N!/2^(nu*(N-K)*ell) is below one."""
    factorial_n = math.factorial(n)
    q = 1 << nu

    for codimension in range(n + 1):
        if factorial_n < q ** (codimension * ell):
            return n - codimension

    raise ValueError(
        f"no K in [0,{n}] gives an expected solution count below one"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare KMP, Esser--Santini, and permutation-based PSD as N "
            "grows, choosing the largest K with expected solution count < 1."
        )
    )
    parser.add_argument("--n-min", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=160)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--nu", type=int, default=11, help="q=2^nu")
    parser.add_argument("--ell", type=int, default=1)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="output CSV (default: derived from the parameters)",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="output plot (default: derived from the parameters)",
    )
    args = parser.parse_args()

    if args.n_min <= 0 or args.n_min > args.n_max:
        parser.error("require 0 < n_min <= n_max")
    if args.n_step <= 0 or args.nu <= 0 or args.ell <= 0:
        parser.error("n_step, nu, and ell must be positive")

    out_csv = args.out_csv or Path(
        f"pkp_attacks_vs_n_N{args.n_min}to{args.n_max}_"
        f"step{args.n_step}_q2pow{args.nu}_ell{args.ell}.csv"
    )
    out_png = args.out_png or Path(
        f"pkp_attacks_vs_n_N{args.n_min}to{args.n_max}_"
        f"step{args.n_step}_q2pow{args.nu}_ell{args.ell}.png"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "N",
        "K",
        "codimension",
        "nu",
        "ell",
        "log2_expected_solutions",
        "kmp_log2",
        "esser_santini_log2_exact_two_size",
        "esser_santini_log2_smooth",
        "psd_log2_exact_padded",
        "psd_log2_asymptotic",
    ]
    rows = []

    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for n in range(args.n_min, args.n_max + 1, args.n_step):
            k = largest_k_below_one(n, args.nu, args.ell)
            codimension = n - k
            log2_expected = (
                log2_gamma_factorial(n)
                - codimension * args.ell * args.nu
            )
            if not log2_expected < 0:
                raise AssertionError("selected parameters must have E[S] < 1")
            if k < n and (
                log2_gamma_factorial(n)
                - (codimension - 1) * args.ell * args.nu
                < 0
            ):
                raise AssertionError("selected K is not maximal")

            point = compute_point(n, k, args.ell, args.nu)
            row = {
                "N": n,
                "K": k,
                "codimension": codimension,
                "nu": args.nu,
                "ell": args.ell,
                "log2_expected_solutions": log2_expected,
                "kmp_log2": point.kmp_log2,
                "esser_santini_log2_exact_two_size": point.es_log2_exact,
                "esser_santini_log2_smooth": point.es_log2_smooth,
                "psd_log2_exact_padded": point.psd_log2_exact,
                "psd_log2_asymptotic": point.psd_log2_asymptotic,
            }
            writer.writerow(row)
            rows.append(row)

            print(
                f"N={n:3d} K={k:3d} N-K={codimension:3d} "
                f"log2(E[S])={log2_expected:8.4f} | "
                f"KMP={point.kmp_log2:10.4f} "
                f"ES={point.es_log2_exact:10.4f} "
                f"PSD_exact={point.psd_log2_exact:10.4f} "
                f"PSD_asym={point.psd_log2_asymptotic:10.4f}"
            )

    ns = [row["N"] for row in rows]
    plt.figure(figsize=(10, 6))
    plt.plot(
        ns,
        [row["kmp_log2"] for row in rows],
        color="#6F9CC5",
        linewidth=2.2,
        label="KMP",
    )
    plt.plot(
        ns,
        [row["esser_santini_log2_exact_two_size"] for row in rows],
        color="#EBA469",
        linewidth=2.2,
        linestyle="--",
        label="Esser--Santini",
    )
    plt.plot(
        ns,
        [row["psd_log2_asymptotic"] for row in rows],
        color="#69B199",
        linewidth=2.2,
        label="Permutation-based PSD",
    )
    plt.xlabel(r"PKP length $N$")
    plt.ylabel(r"$\log_2 T$")
    plt.title(
        rf"$q=2^{{{args.nu}}}$, $\ell={args.ell}$; "
        rf"maximal $K$ with $\mathbb{{E}}[S]<1$"
    )
    plt.grid(axis="y", color="0.88", linewidth=0.8)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()

    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
