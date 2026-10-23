#!/usr/bin/env python3
import math
import random
from collections import Counter
from functools import lru_cache


@lru_cache(maxsize=512)
def arrangement_counts(multiplicities):
    """Exact counts of distinct ordered partial arrangements, for every length.

    Appending j copies of a new symbol to a length-t word gives
    binom(t+j,j) interleavings. Zero multiplicities may be omitted.
    """
    if not multiplicities or any(not isinstance(m, int) or m <= 0
                                 for m in multiplicities):
        raise ValueError("multiplicities must be positive integers")
    counts = [1]
    for m in multiplicities:
        updated = [0] * (len(counts) + m)
        for t, count in enumerate(counts):
            for j in range(m + 1):
                updated[t + j] += math.comb(t + j, j) * count
        counts = updated
    return tuple(counts)


def speck_complexity(n, k, q, multiplicities):
    """SPECK (2025/923), Prop. 2 and Sec. 6.2, Lemma 2; scalar PKP only.

    k is the CODE DIMENSION, unlike the legacy plot calls to kmp_complexity.
    Counts distinct permuted vectors, not permutations of repeated symbols.
    No favorable-codeword sampling. The solution count and collision model
    are heuristic. Reduced lists are floored at one at exponent precision.
    Both the sum in Lemma 2 and its polynomial-factor-suppressed maximum
    are returned; optimization uses the latter to match the paper plots.
    """
    multiplicities = tuple(sorted(multiplicities))
    if n <= 0 or not 0 <= k < n or q < 2 or sum(multiplicities) != n:
        raise ValueError("require n>0, 0<=k<n, q>=2 and multiplicities summing to n")
    if len(multiplicities) > q:
        raise ValueError("more distinct symbols than field elements")
    counts = arrangement_counts(multiplicities)
    log_m = math.log2(counts[n])
    checks = (n - k) * math.log2(q)
    # 1 + (M-1)/q^(n-k), evaluated without constructing tiny floats.
    log_s = (log2sumexp([0.0, math.log2(counts[n] - 1) - checks])
             if counts[n] > 1 else 0.0)
    best = None
    for t in range(1, n - k + 1):
        u1 = (k + t) // 2
        u2 = k + t - u1
        a = max(0.0, math.log2(counts[u1]) - log_s / 2)
        b = max(0.0, math.log2(counts[u2]) - log_s / 2)
        collision = a + b - t * math.log2(q)
        exponent = max(a, b, collision)
        result = dict(best_log2_T=exponent, total_log2_T=log2sumexp([a, b, collision]),
                      matching_symbols=t, u1=u1, u2=u2,
                      L1_log2=a, L2_log2=b, collisions_log2=collision)
        if best is None or exponent < best['best_log2_T']:
            best = result
    best.update(log2_solutions=log_s, log2_arrangements=log_m,
                random_guess_log2=max(0.0, log_m - log_s),
                multiplicities=multiplicities)
    return best


def speck_random_profile_complexity(n, k, q, samples=100, seed=1):
    """Arithmetic mean work over uniform nonzero input vectors, in log2.

    Samples describe an ensemble of fixed inputs; they are NOT attacker
    retries and the cheapest profile is never selected. Fixed seeds make
    the plot reproducible. No finite-field arithmetic is needed for counts.
    """
    if samples < 1 or n < 1 or q < 2 or not 0 <= k < n:
        raise ValueError("invalid parameters or sample count")
    rng = random.Random(f"{seed}:{n}:{k}:{q}")
    results = []
    for _ in range(samples):
        while True:
            vector = [rng.randrange(q) for _ in range(n)]
            if any(vector):
                break
        profile = tuple(sorted(Counter(vector).values()))
        results.append(speck_complexity(n, k, q, profile))
    costs = [r['best_log2_T'] for r in results]
    guesses = [r['random_guess_log2'] for r in results]
    return dict(best_log2_T=log2sumexp(costs) - math.log2(samples),
                random_guess_log2=log2sumexp(guesses) - math.log2(samples),
                min_log2_T=min(costs), max_log2_T=max(costs), samples=samples,
                seed=seed)

# =========================
# Helpers: primes & powers
# =========================

def is_prime(p: int) -> bool:
    if p < 2:
        return False
    if p % 2 == 0:
        return p == 2
    i = 3
    r = int(p ** 0.5)
    while i <= r:
        if p % i == 0:
            return False
        i += 2
    return True

def next_prime(n: int) -> int:
    """Smallest prime >= n."""
    if n <= 2:
        return 2
    p = n if n % 2 else n + 1
    while not is_prime(p):
        p += 2
    return p

def next_prime_power_ge(x: float) -> int:
    """
    Smallest prime power p^k >= x, k>=1.
    Strategy:
      - candidate 1: next prime >= x  (k=1)
      - candidates with k>=2: for primes p <= x, take smallest p^k >= x
      - return the minimum among candidates
    """
    x_ceil = max(2, int(math.ceil(x)))
    best = next_prime(x_ceil)

    max_p = x_ceil
    for p in range(2, max_p + 1):
        if not is_prime(p):
            continue
        val = p * p
        while val < x and val <= best:
            val *= p
            if val > best:
                break
        if val >= x and val < best:
            best = val
    return best

# ======================
# Base-2 log utilities
# ======================

_LN2 = math.log(2.0)

def log2_factorial(n: int) -> float:
    """log2(n!) via lgamma for stability."""
    return math.lgamma(n + 1.0) / _LN2

def log2sumexp(vals):
    """Return log2(sum_i 2^{vals[i]}). vals are log2 numbers."""
    m = max(vals)
    if m == float("-inf"):
        return m
    s = 0.0
    for v in vals:
        # 2^{v} = 2^{m} * 2^{v-m}; pull out 2^{m}
        s += 2.0 ** (v - m)
    return m + math.log2(s)

# ===========================
# Complexity terms (log2)
# ===========================

def log2_Li(n: int, ui: int) -> float:
    """log2 |L_i| = log2( n! / (n-ui)! )."""
    if ui < 0 or ui > n:
        return float("-inf")
    return log2_factorial(n) - log2_factorial(n - ui)

def log2_N_join(n: int, l: int, r: int, u1: int, u2: int, log2q: float) -> float:
    """
    log2 N_{L1 ⋈ L2} =
      2*log2(n!) - log2((n-u1)!) - log2((n-u2)!) + l*(n-r-u1-u2)*log2(q)
    """
    exp_q = l * (n - r - u1 - u2)  # can be negative
    return (2 * log2_factorial(n)
            - log2_factorial(n - u1)
            - log2_factorial(n - u2)
            + exp_q * log2q)

def log2_L(n: int, l: int, r: int, u1: int, u2: int, log2q: float) -> float:
    """log2 |L| = log2( n! / (n-u1-u2)! ) + l*(n-r-u1-u2)*log2(q)."""
    s = u1 + u2
    if s > n:
        return float("-inf")
    exp_q = l * (n - r - s)
    return log2_factorial(n) - log2_factorial(n - s) + exp_q * log2q

def log2_T(n: int, l: int, r: int, u1: int, u2: int, log2q: float) -> float:
    """log2 total T = log2( |L1| + |L2| + N_join + |L| )."""
    return log2sumexp([
        log2_Li(n, u1),
        log2_Li(n, u2),
        log2_N_join(n, l, r, u1, u2, log2q),
        log2_L(n, l, r, u1, u2, log2q),
    ])

# =========================
# Public API (log2-based)
# =========================

def pick_q_min(n: int, l: int, r: int) -> int:
    """
    Choose q as the smallest prime power >= ceil( (n/e)^(1/(l*(1-R))) ),
    where R = r/n. (Same prescription, independent of log base.)
    """
    if r <= 0 or r >= n:
        R = max(1e-12, min(1 - 1e-12, r / n))
    else:
        R = r / n
    base = (n / math.e) ** (1.0 / (l * (1.0 - R)))
    q0 = math.ceil(base)
    return next_prime_power_ge(q0)

def optimize_T(n: int, l: int, r: int, q: int, return_all_min=False, tol_log2=1e-12):
    """
    Enumerate u1,u2 with n - r + 1 <= u1 + u2 <= n and 0<=ui<=n.
    Find argmin of T_KMP using base-2 logs.
    If return_all_min=True, return all pairs within tol_log2 of the minimum (in log2 scale).
    """
    log2q = math.log2(q)
    best_log2T = float("inf")
    best_pairs = []
    lo_sum = max(0, n - r + 1)
    hi_sum = n

    for u1 in range(0, n + 1):
        # u2 constrained by sum window and by nonnegativity in factorials:
        u2_min = max(0, lo_sum - u1)
        u2_max = min(n - u1, hi_sum - u1)
        if u2_min > u2_max:
            continue
        for u2 in range(u2_min, u2_max + 1):
            lt = log2_T(n, l, r, u1, u2, log2q)
            if lt < best_log2T - 1e-18:
                best_log2T = lt
                best_pairs = [(u1, u2)]
            elif return_all_min and abs(lt - best_log2T) <= tol_log2:
                best_pairs.append((u1, u2))

    if not best_pairs:
        return float("inf"), []

    if not return_all_min:
        # deterministic tie-break
        best_pairs = [min(best_pairs)]
    return best_log2T, best_pairs

def components_log2(n: int, l: int, r: int, u1: int, u2: int, q: int):
    """
    Return the four components **in log2**:
      L1, L2, N_join, L, and their sum T (all in base-2 logs).
    """
    log2q = math.log2(q)
    L1 = log2_Li(n, u1)
    L2 = log2_Li(n, u2)
    Njoin = log2_N_join(n, l, r, u1, u2, log2q)
    L = log2_L(n, l, r, u1, u2, log2q)
    T = log2sumexp([L1, L2, Njoin, L])

    return {
        "L1_log2": L1,
        "L2_log2": L2,
        "N_join_log2": Njoin,
        "L_log2": L,
        "T_log2": T,
    }

def kmp_complexity(n: int, r: int, l: int, q: int, list_all_min=False, debug=False):

    best_log2T, best_pairs = optimize_T(n, l, r, q, return_all_min=list_all_min)
    out = {
        "n": n, "l": l, "r": r,
        "q": q,
        "best_log2_T": best_log2T,
        "best_pairs": [],
    }
    for (u1, u2) in best_pairs:
        comps = components_log2(n, l, r, u1, u2, q)
        out["best_pairs"].append({
            "u1": u1, "u2": u2,
            "components_log2": comps
        })
    if debug:
        print(
            f"\nKMP Result for n: {out['n']}, r: {out['r']}, ell: {out['l']}, q: {out['q']}\n"
            f"Minimizing (u1,u2) pairs : {out.get('pairs', 'N/A')}\n"
            f"T_total   (log2)         : {out['best_log2_T']:.6f}\n"
            f"===================================================="
        )

    return out, best_log2T

# ---------- CLI ----------

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Compute the complexity of KMP.")
    parser.add_argument("n", type=int, help="Code length n")
    parser.add_argument("r", type=int, help="legacy CLI parameter n-k-1 (the API uses n-k)")
    parser.add_argument("l", type=int, help="Parameter l")
    parser.add_argument("q", type=int, help="Field size q")
    parser.add_argument("--all-min", action="store_true",
                        help="List all (u1,u2) that are tied (within ~1e-6 in log2 T) for minimum.")
    parser.add_argument("--speck", action="store_true", help="also estimate scalar PKP using SPECK")
    parser.add_argument("--multiplicities", type=int, nargs="+", help="exact positive symbol counts")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    res = kmp_complexity(args.n, args.r+1, args.l, args.q, list_all_min=args.all_min, debug=True)
    if args.speck:
        if args.l != 1:
            parser.error("SPECK estimation currently supports l=1 only")
        # Legacy CLI uses n-k+1, and adds one above; keep that behavior intact.
        k = args.n - args.r - 1
        estimate = (speck_complexity(args.n, k, args.q, args.multiplicities)
                    if args.multiplicities else
                    speck_random_profile_complexity(args.n, k, args.q, args.samples, args.seed))
        print("SPECK (dimension k=%d):" % k)
        print(json.dumps(estimate, indent=2))
