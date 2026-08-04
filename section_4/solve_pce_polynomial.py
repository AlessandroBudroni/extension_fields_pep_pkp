#!/usr/bin/env sage
"""Sample and solve self-orthogonal or self-dual PCE/PEP instances.

The solver uses only the public generator matrices.  For each possible
position of the nonzero entry in a fixed row, it fixes the corresponding row
and column, solves the descended binary system, and enumerates the remaining
affine solution space as in the polynomial-time lemma.
"""

import argparse
import random
import time

from sage.all import GF, set_random_seed, vector, zero_matrix

from polynomial_solver_utils import (
    affine_solution_space,
    column_vector_to_matrix,
    enumerate_affine_space,
    fixed_row_and_column_constraints,
    is_permutation_matrix,
)
from verify_assumption_2 import (
    project_matrix_to_binary,
    random_invertible_matrix,
    random_permutation_matrix,
    random_self_orthogonal_generator,
)


def sample_pce_instance(field, n, k, max_tries):
    """Return public (G,G') and the parity check of G'."""
    G = random_self_orthogonal_generator(field, n, k, max_tries)
    equivalence = random_permutation_matrix(field, n)
    mixing = random_invertible_matrix(field, k)
    G_prime = mixing * G * equivalence
    H_prime = G_prime.right_kernel().basis_matrix()
    if G_prime * H_prime.transpose() != 0:
        raise AssertionError("invalid parity-check matrix")
    return G, G_prime, H_prime


def solve_pce(G, H_prime, max_kernel_dimension):
    """Recover P such that H' P G^T = 0."""
    field_two = GF(2)
    field = G.base_ring()
    n = G.ncols()
    tensor_system = G.tensor_product(H_prime)
    derived = project_matrix_to_binary(tensor_system)
    total_tested = 0

    for guessed_column in range(n):
        constraints, constraint_syndrome = fixed_row_and_column_constraints(
            n, 0, guessed_column
        )
        system = derived.stack(constraints)
        syndrome = vector(
            field_two,
            [0] * derived.nrows() + list(constraint_syndrome),
        )
        particular, basis = affine_solution_space(system, syndrome)
        if particular is None:
            continue

        for candidate in enumerate_affine_space(
            particular, basis, max_kernel_dimension
        ):
            total_tested += 1
            permutation = column_vector_to_matrix(candidate, n)
            permutation_ext = permutation.change_ring(field)
            if (
                is_permutation_matrix(permutation)
                and H_prime * permutation_ext * G.transpose()
                == zero_matrix(field, H_prime.nrows(), G.nrows())
            ):
                return (
                    permutation,
                    guessed_column,
                    len(basis),
                    total_tested,
                    int(derived.rank()),
                )
    raise RuntimeError("no valid permutation was found")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample and solve self-orthogonal or self-dual PCE/PEP instances "
            "using the polynomial attack."
        )
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-orthogonal", action="store_true")
    modes.add_argument("--self-dual", action="store_true")
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("nu", type=int)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-tries", type=int, default=10000)
    parser.add_argument(
        "--max-kernel-dimension",
        type=int,
        default=20,
        help="refuse affine enumeration above this dimension",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    case = "self-orthogonal" if args.self_orthogonal else "self-dual"
    if not 0 < args.k <= args.n // 2:
        raise ValueError("require 0 < k <= n/2")
    if args.self_orthogonal and not args.k < args.n / 2:
        raise ValueError("strictly self-orthogonal mode requires k < n/2")
    if args.self_dual and 2 * args.k != args.n:
        raise ValueError("self-dual mode requires k=n/2")
    if min(args.nu, args.instances, args.max_tries) <= 0:
        raise ValueError("nu, instances, and max-tries must be positive")
    if args.max_kernel_dimension < 0:
        raise ValueError("max-kernel-dimension must be nonnegative")

    binary_rows = (args.n - args.k) * args.k * args.nu
    if binary_rows <= args.n**2:
        raise ValueError(
            "parameters are outside the polynomial regime: require "
            "(n-k) k nu > n^2"
        )

    random.seed(args.seed)
    set_random_seed(args.seed)
    field = GF(2**args.nu, name=f"z{args.nu}")
    print(
        f"PCE case={case}, n={args.n}, k={args.k}, nu={args.nu}, "
        f"binary rows={binary_rows}"
    )

    for instance in range(args.instances):
        G, G_prime, H_prime = sample_pce_instance(
            field, args.n, args.k, args.max_tries
        )
        start = time.perf_counter()
        permutation, guess, dimension, tested, rank = solve_pce(
            G, H_prime, args.max_kernel_dimension
        )
        elapsed = time.perf_counter() - start

        # The reformulated problem returns P with H' P G^T=0.  Its transpose
        # maps the original code generated by G to the code generated by G'.
        equivalence = permutation.transpose().change_ring(field)
        if (G * equivalence).row_space() != G_prime.row_space():
            raise AssertionError("recovered permutation does not map the codes")
        print(
            f"instance={instance:3d}: verified, rank={rank}/{args.n**2}, "
            f"guess={guess}, affine_dimension={dimension}, "
            f"candidates={tested}, time={elapsed:.3f}s"
        )


if __name__ == "__main__":
    main()
