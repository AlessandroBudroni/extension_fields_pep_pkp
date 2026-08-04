#!/usr/bin/env sage
"""Sample and solve PKP instances in the polynomial-time parameter regime.

The solver uses only the public matrices H and V.  It descends the tensor
system to GF(2), appends the row/column permutation equations, enumerates the
resulting affine solution space, and verifies permutation candidates.
"""

import argparse
import random
import time

from sage.all import GF, set_random_seed, vector

from polynomial_solver_utils import (
    affine_solution_space,
    column_vector_to_matrix,
    enumerate_affine_space,
    is_permutation_matrix,
)
from verify_assumption_1 import (
    binary_expansion_of_tensor,
    permutation_equation_matrix,
    sample_random_pkp,
)


def solve_pkp(H, V, max_kernel_dimension):
    """Recover a permutation E satisfying H E V = 0."""
    field_two = GF(2)
    n = H.ncols()
    nu = H.base_ring().degree()
    derived = binary_expansion_of_tensor(H, V, nu)
    permutation_equations = permutation_equation_matrix(n)
    system = derived.stack(permutation_equations)
    syndrome = vector(
        field_two,
        [0] * derived.nrows() + [1] * permutation_equations.nrows(),
    )

    particular, basis = affine_solution_space(system, syndrome)
    if particular is None:
        raise RuntimeError("the augmented PKP system is inconsistent")

    tested = 0
    for candidate in enumerate_affine_space(
        particular, basis, max_kernel_dimension
    ):
        tested += 1
        permutation = column_vector_to_matrix(candidate, n)
        if is_permutation_matrix(permutation) and H * permutation * V == 0:
            return permutation, len(basis), tested, int(system.rank())
    raise RuntimeError("the affine space contains no valid PKP permutation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample and solve PKP instances using the polynomial attack."
    )
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("ell", type=int)
    parser.add_argument("nu", type=int)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-kernel-dimension",
        type=int,
        default=20,
        help="refuse affine enumeration above this dimension",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.ell < args.k < args.n:
        raise ValueError("require 0 < ell < k < n")
    if args.nu <= 0 or args.instances <= 0:
        raise ValueError("nu and instances must be positive")
    if args.max_kernel_dimension < 0:
        raise ValueError("max-kernel-dimension must be nonnegative")

    binary_rows = (args.n - args.k) * args.ell * args.nu
    boundary = (args.n - 1) ** 2
    if binary_rows < boundary:
        raise ValueError(
            "parameters are outside the polynomial regime: require "
            "(n-k) ell nu >= (n-1)^2"
        )

    random.seed(args.seed)
    set_random_seed(args.seed)
    field = GF(2**args.nu, name=f"z{args.nu}")
    print(
        f"PKP(n={args.n}, k={args.k}, ell={args.ell}, nu={args.nu}), "
        f"binary rows={binary_rows}"
    )

    for instance in range(args.instances):
        H, V, _ = sample_random_pkp(args.n, args.k, args.ell, field)
        start = time.perf_counter()
        _, dimension, tested, rank = solve_pkp(
            H, V, args.max_kernel_dimension
        )
        elapsed = time.perf_counter() - start
        print(
            f"instance={instance:3d}: verified, rank={rank}/{args.n**2}, "
            f"affine_dimension={dimension}, candidates={tested}, "
            f"time={elapsed:.3f}s"
        )


if __name__ == "__main__":
    main()
