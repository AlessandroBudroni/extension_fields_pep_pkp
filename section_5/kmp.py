#!/usr/bin/env python3
import math

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
    parser.add_argument("r", type=int, help="Dimension r")
    parser.add_argument("l", type=int, help="Parameter l")
    parser.add_argument("q", type=int, help="Field size q")
    parser.add_argument("--all-min", action="store_true",
                        help="List all (u1,u2) that are tied (within ~1e-6 in log2 T) for minimum.")
    args = parser.parse_args()

    res = kmp_complexity(args.n, args.r+1, args.l, args.q, list_all_min=args.all_min, debug=True)
