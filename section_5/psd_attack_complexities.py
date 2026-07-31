#!/usr/bin/env python3
import argparse
import math

LN2 = math.log(2.0)


def log2_gamma_factorial(x: float) -> float:
    """
    Return log2(x!) using Gamma(x+1), valid for real x > -1.
    """
    if x <= -1:
        raise ValueError("factorial argument must be greater than -1")
    return math.lgamma(x + 1.0) / LN2


def solution_count_log2(n: int, w: int, k: int) -> float:
    """
    Compute log2(S) for
        S = max(1, w! * 2^{-(n-k)}).

    This uses the original instance dimension k, as requested.
    """
    return max(log2_gamma_factorial(w) - (n - k), 0.0)


def validate_psd_parameters(n: int, w: int, k: int) -> None:
    if n <= 0:
        raise ValueError("n must be positive")
    if w <= 0:
        raise ValueError("w must be positive")
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if w * w != n:
        raise ValueError("PSD parameters must satisfy w^2 = n")


def regular_effective_dimension(k: int, w: int) -> int:
    """Return the effective dimension used by regular ISD."""
    return k - w


def permutation_effective_dimension(k: int, w: int) -> int:
    """Return the effective dimension after enforcing permutation structure."""
    return k - (2 * w - 1)


def esser_santini_psd_log2_complexity(n: int, w: int, k: int) -> tuple[float, float, int]:
    """
    Esser-Santini regular-ISD complexity for a PSD instance viewed as RSD.

    Its effective dimension is
        k_eff = k - w.

    A uniformly sampled regular information set avoids the w support
    positions with smooth asymptotic probability
        p = (1 - k_eff/n)^w.

    Hence the expected running time, up to polynomial factors, is
        T = (1 - k_eff/n)^(-w) / S,

    where S is the estimated number of permutation-matrix solutions.
    """
    validate_psd_parameters(n, w, k)

    k_eff = regular_effective_dimension(k, w)
    if k_eff < 0:
        raise ValueError("k - w must be non-negative")

    log2_s = solution_count_log2(n, w, k)
    p = 1.0 - ((k_eff) / n)
    if p <= 0:
        raise ValueError("invalid parameters: 1 - k_eff/n must be positive")

    log2_t = -w * math.log2(p) - log2_s
    return log2_t, log2_s, k_eff


def psd_perm_isd_log2_complexity(n: int, w: int, k: int) -> tuple[float, float, int, float]:
    """
    Our permutation-based PSD complexity after adding 2w-1 permutation equations.

        T = O~( w! / (((w-v)!)^(w/(w-v)) * S) )

    with
        k_eff = k - (2w-1)
        v = k_eff / w.

    When v is not integral, we extend the factorial term using Gamma, which is
    the natural analogue of how the Esser-Santini expression uses the effective
    dimension directly without rounding.
    """
    validate_psd_parameters(n, w, k)

    k_eff = permutation_effective_dimension(k, w)
    if k_eff < 0:
        raise ValueError("k - (2w-1) must be non-negative")

    v = k_eff / w
    if v < 0 or v >= w:
        raise ValueError("v = k_eff / w must satisfy 0 <= v < w")

    log2_s = solution_count_log2(n, w, k)
    remaining = w - v
    log2_t = log2_gamma_factorial(w) - (w / remaining) * log2_gamma_factorial(remaining) - log2_s
    return log2_t, log2_s, k_eff, v


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Esser-Santini and permutation-based PSD attack complexities on a PSD instance."
    )
    parser.add_argument("n", type=int, help="PSD length parameter n")
    parser.add_argument("w", type=int, help="PSD weight parameter w, with w^2 = n")
    parser.add_argument("k", type=int, help="Original PSD dimension parameter k")
    args = parser.parse_args()

    es_log2_t, log2_s, es_k_eff = esser_santini_psd_log2_complexity(
        args.n, args.w, args.k
    )
    psd_log2_t, _, psd_k_eff, v = psd_perm_isd_log2_complexity(
        args.n, args.w, args.k
    )

    print(f"n                    = {args.n}")
    print(f"w                    = {args.w}")
    print(f"k                    = {args.k}")
    print(f"Esser-Santini k_eff = {es_k_eff}")
    print(f"PSD attack k_eff    = {psd_k_eff}")
    print(f"log2(S)              = {log2_s:.12f}")
    print(f"v                    = {v:.12f}")
    print()
    print(f"Esser-Santini log2(T)= {es_log2_t:.12f}")
    print(f"PSD attack log2(T)   = {psd_log2_t:.12f}")
    print()
    print(f"Esser-Santini        ≈ 2^({es_log2_t:.6f})")
    print(f"PSD attack           ≈ 2^({psd_log2_t:.6f})")


if __name__ == "__main__":
    main()
