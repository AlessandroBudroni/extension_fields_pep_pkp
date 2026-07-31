#!/usr/bin/env python3
"""Compare PKP attacks using the padded two-size PSD partition formula."""

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from kmp import kmp_complexity  # noqa: E402
from psd_attack_complexities import (  # noqa: E402
    esser_santini_psd_log2_complexity,
    log2_gamma_factorial,
)


@dataclass(frozen=True)
class Point:
    nu: int
    k_psd: int
    k_prime: int
    es_k_eff: int
    es_a: int
    es_b: int
    es_r0: int
    es_r1: int
    a: int
    b: int
    r0: int
    r1: int
    q0: int
    s0: int
    q1: int
    s1: int
    log2_s: float
    log2_partition_numerator: float
    log2_partition_numerator_asymptotic: float
    kmp_log2: float
    es_log2_exact: float
    es_log2_smooth: float
    psd_log2_exact: float
    psd_log2_asymptotic: float


def padded_partition_psd_log2(
    original_n: int, codimension: int, ell: int, nu: int
) -> tuple[
    float,
    float,
    float,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
]:
    """
    Return the modeled PSD work factor from the two-size construction.

    The binary PSD instance has dimension
        k_psd = N^2 - (N-K)*ell*nu.
    After adjoining the 2N-1 independent permutation equations, the
    information-set size is k_prime=k_psd-2N+1.  Writing k_prime=aN+b
    gives N-b blocks retaining r0=N-a positions and b blocks retaining
    r1=N-a-1 positions.

    Complete groups share a retained-position cell.  A remainder group of
    size s_i shares its s_i-cell and pads it deterministically to r_i
    positions.  Therefore
        Pr[success] >=
        (r0!)^q0 s0! (r1!)^q1 s1! / N!.
    """
    n_psd = original_n * original_n
    original_rows = codimension * ell * nu
    k_psd = n_psd - original_rows
    k_prime = k_psd - 2 * original_n + 1
    if not 0 <= k_prime < n_psd:
        raise ValueError("augmented PSD information-set size is outside [0,N^2)")

    a, b = divmod(k_prime, original_n)
    r0 = original_n - a
    r1 = r0 - 1
    if r0 <= 0 or (b > 0 and r1 <= 0):
        raise ValueError("the retained sizes must be positive")

    q0, s0 = divmod(original_n - b, r0)
    if b:
        q1, s1 = divmod(b, r1)
    else:
        q1, s1 = 0, 0

    retained = (original_n - b) * r0 + b * r1
    if retained != n_psd - k_prime:
        raise AssertionError("two-size retained-column count is inconsistent")

    log2_partition_numerator = (
        q0 * log2_gamma_factorial(r0)
        + log2_gamma_factorial(s0)
        + q1 * log2_gamma_factorial(r1)
        + log2_gamma_factorial(s1)
    )
    log2_partition_numerator_asymptotic = (
        ((original_n - b) / r0) * log2_gamma_factorial(r0)
        + ((b / r1) * log2_gamma_factorial(r1) if b else 0.0)
    )
    log2_s = max(
        log2_gamma_factorial(original_n) - original_rows, 0.0
    )
    psd_log2_exact = (
        log2_gamma_factorial(original_n)
        - log2_partition_numerator
        - log2_s
    )
    psd_log2_asymptotic = (
        log2_gamma_factorial(original_n)
        - log2_partition_numerator_asymptotic
        - log2_s
    )
    return (
        psd_log2_exact,
        psd_log2_asymptotic,
        log2_s,
        k_psd,
        k_prime,
        a,
        b,
        r0,
        r1,
        q0,
        s0,
        q1,
        s1,
        log2_partition_numerator,
        log2_partition_numerator_asymptotic,
    )


def compute_point(
    original_n: int, original_k: int, ell: int, nu: int
) -> Point:
    codimension = original_n - original_k
    (
        psd_log2_exact,
        psd_log2_asymptotic,
        log2_s,
        k_psd,
        k_prime,
        a,
        b,
        r0,
        r1,
        q0,
        s0,
        q1,
        s1,
        log2_partition_numerator,
        log2_partition_numerator_asymptotic,
    ) = padded_partition_psd_log2(original_n, codimension, ell, nu)

    _, kmp_log2 = kmp_complexity(
        original_n, codimension, ell, 1 << nu, debug=False
    )
    es_log2_smooth, _, es_k_eff = esser_santini_psd_log2_complexity(
        original_n * original_n, original_n, k_psd
    )
    es_a, es_b = divmod(es_k_eff, original_n)
    es_r0 = original_n - es_a
    es_r1 = es_r0 - 1
    if es_r0 <= 0 or (es_b > 0 and es_r1 <= 0):
        raise ValueError("the Esser--Santini retained sizes must be positive")
    es_log2_exact = (
        (original_n - es_b) * math.log2(original_n / es_r0)
        + (
            es_b * math.log2(original_n / es_r1)
            if es_b
            else 0.0
        )
        - log2_s
    )
    if es_log2_exact + 1e-12 < es_log2_smooth:
        raise AssertionError(
            "exact Esser--Santini cost must dominate its smooth interpolation"
        )
    return Point(
        nu=nu,
        k_psd=k_psd,
        k_prime=k_prime,
        es_k_eff=es_k_eff,
        es_a=es_a,
        es_b=es_b,
        es_r0=es_r0,
        es_r1=es_r1,
        a=a,
        b=b,
        r0=r0,
        r1=r1,
        q0=q0,
        s0=s0,
        q1=q1,
        s1=s1,
        log2_s=log2_s,
        log2_partition_numerator=log2_partition_numerator,
        log2_partition_numerator_asymptotic=(
            log2_partition_numerator_asymptotic
        ),
        kmp_log2=kmp_log2,
        es_log2_exact=es_log2_exact,
        es_log2_smooth=es_log2_smooth,
        psd_log2_exact=psd_log2_exact,
        psd_log2_asymptotic=psd_log2_asymptotic,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare KMP, Esser--Santini, and the permutation-based PSD "
            "attack using the padded two-size construction."
        )
    )
    parser.add_argument("N", type=int, help="original PKP length")
    parser.add_argument("K", type=int, help="original PKP dimension")
    parser.add_argument("L", type=int, help="PKP ell parameter")
    parser.add_argument("nu_min", type=int)
    parser.add_argument("nu_max", type=int)
    parser.add_argument("--out-png", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    if args.N <= 0 or not 0 <= args.K < args.N or args.L <= 0:
        parser.error("require N>0, 0<=K<N, and L>0")
    if args.nu_min > args.nu_max:
        parser.error("nu_min must not exceed nu_max")

    stem = (
        f"pkp_complexity_padded_N{args.N}_K{args.K}_L{args.L}_"
        f"nu{args.nu_min}to{args.nu_max}"
    )
    out_png = args.out_png or Path(f"{stem}.png")
    out_csv = args.out_csv or Path(f"{stem}.csv")

    points = [
        compute_point(args.N, args.K, args.L, nu)
        for nu in range(args.nu_min, args.nu_max + 1)
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "nu",
                "k_psd",
                "k_prime",
                "esser_santini_k_eff",
                "esser_santini_a",
                "esser_santini_b",
                "esser_santini_r0",
                "esser_santini_r1",
                "a",
                "b",
                "r0",
                "r1",
                "q0",
                "s0",
                "q1",
                "s1",
                "log2_S",
                "log2_partition_numerator",
                "log2_partition_numerator_asymptotic",
                "kmp_log2",
                "esser_santini_log2_exact_two_size",
                "esser_santini_log2_smooth",
                "psd_log2_exact_padded",
                "psd_log2_asymptotic",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.nu,
                    point.k_psd,
                    point.k_prime,
                    point.es_k_eff,
                    point.es_a,
                    point.es_b,
                    point.es_r0,
                    point.es_r1,
                    point.a,
                    point.b,
                    point.r0,
                    point.r1,
                    point.q0,
                    point.s0,
                    point.q1,
                    point.s1,
                    point.log2_s,
                    point.log2_partition_numerator,
                    point.log2_partition_numerator_asymptotic,
                    point.kmp_log2,
                    point.es_log2_exact,
                    point.es_log2_smooth,
                    point.psd_log2_exact,
                    point.psd_log2_asymptotic,
                ]
            )

    nus = [point.nu for point in points]
    plt.figure(figsize=(10, 6))
    plt.plot(
        nus,
        [point.kmp_log2 for point in points],
        color="#6F9CC5",
        linewidth=2.2,
        label="KMP",
    )
    plt.plot(
        nus,
        [point.es_log2_exact for point in points],
        color="#EBA469",
        linewidth=2.2,
        linestyle="--",
        label="Esser--Santini",
    )
    plt.plot(
        nus,
        [point.psd_log2_asymptotic for point in points],
        color="#69B199",
        linewidth=2.2,
        label="Permutation-based PSD",
    )
    plt.xlabel(r"$\nu$ ($q=2^\nu$)")
    plt.ylabel(r"$\log_2 T$")
    plt.title(
        f"PKP attack comparison: N={args.N}, K={args.K}, L={args.L}"
    )
    plt.grid(axis="y", color="0.88", linewidth=0.8)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()

    for point in points:
        print(
            f"nu={point.nu:3d} r0={point.r0:3d} r1={point.r1:3d} "
            f"ES_r0={point.es_r0:3d} ES_r1={point.es_r1:3d} "
            f"(q0,s0)=({point.q0},{point.s0}) "
            f"(q1,s1)=({point.q1},{point.s1}) "
            f"KMP={point.kmp_log2:10.4f} "
            f"ES_exact={point.es_log2_exact:10.4f} "
            f"ES_smooth={point.es_log2_smooth:10.4f} "
            f"PSD_exact={point.psd_log2_exact:10.4f} "
            f"PSD_asym={point.psd_log2_asymptotic:10.4f}"
        )
    print(f"\nSaved plot: {out_png}")
    print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()
