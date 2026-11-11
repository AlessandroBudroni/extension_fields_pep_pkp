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

    The parameter k is the dimension before adjoining permutation equations.
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


def padded_partition_log2_numerator(w: int, k_prime: int) -> float:
    """Log2 of the factorial product for a two-size partition.

    Empty block families contribute one. Padding can create extra successes,
    so this product gives a lower bound on the success-probability numerator.
    """
    if w <= 0 or not 0 <= k_prime < w * w:
        raise ValueError("require w > 0 and 0 <= k_prime < w^2")
    a, b = divmod(k_prime, w)
    total = 0.0
    for count, retained in ((w - b, w - a), (b, w - a - 1)):
        if count:
            groups, remainder = divmod(count, retained)
            total += (groups * log2_gamma_factorial(retained)
                      + log2_gamma_factorial(remainder))
    return total


def psd_perm_isd_log2_complexity(n: int, w: int, k: int) -> tuple[float, float, int, float]:
    """Auxiliary two-size PSD partition trial estimate.

    Uses integer group sizes, remainder factorials, and the solution-count
    estimate S. The optimized construction in Section 5.2 is implemented in
    counted_grouping.py. The returned v=k_eff/w is the average information-set size
    per block; no factorial is evaluated at a fractional retained size.
    """
    validate_psd_parameters(n, w, k)

    k_eff = permutation_effective_dimension(k, w)
    if k_eff < 0:
        raise ValueError("k - (2w-1) must be non-negative")

    log2_s = solution_count_log2(n, w, k)
    log2_t = (log2_gamma_factorial(w)
              - padded_partition_log2_numerator(w, k_eff) - log2_s)
    return log2_t, log2_s, k_eff, k_eff / w


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute regular-ISD and two-size PSD partition estimates."
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
    print(f"PSD partition log2(T) estimate = {psd_log2_t:.12f}")
    print()
    print(f"Esser-Santini        ≈ 2^({es_log2_t:.6f})")
    print(f"PSD partition estimate   ≈ 2^({psd_log2_t:.6f})")


if __name__ == "__main__":
    main()
