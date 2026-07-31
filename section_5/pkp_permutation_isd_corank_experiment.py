#!/usr/bin/env sage
"""Compare rank profiles of PKP-derived and uniform-random PSD matrices."""

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sage.all import GF, matrix, random_matrix, set_random_seed


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def choose_r(N, K, ell, nu, requested_r):
    """Choose r | N with t=Nr-(N-K)ell*nu in [0,2N-N/r]."""
    a = (N - K) * ell * nu
    feasible = [
        r
        for r in divisors(N)
        if 0 <= N * r - a <= 2 * N - N // r
    ]
    if requested_r is not None:
        if requested_r not in feasible:
            raise ValueError(
                f"r={requested_r} is infeasible; feasible values are {feasible}"
            )
        return requested_r
    if not feasible:
        raise ValueError("no feasible divisor r; supply different PKP parameters")
    return max(feasible)


def random_full_rank_matrix(field, nrows, ncols):
    while True:
        value = random_matrix(field, nrows, ncols)
        if value.rank() == nrows:
            return value


def sample_pkp_instance(N, K, ell, field):
    """Sample a PKP instance H,V together with a solution E."""
    H = random_full_rank_matrix(field, N - K, N)
    kernel_basis = H.right_kernel().basis_matrix()  # K x N
    coefficients = random_full_rank_matrix(field, ell, K)
    kernel_vectors = (coefficients * kernel_basis).transpose()

    permutation = list(range(N))
    random.shuffle(permutation)
    E = matrix(field, N, N, sparse=True)
    for column, row in enumerate(permutation):
        E[row, column] = 1
    V = E.transpose() * kernel_vectors
    assert H * E * V == 0
    return H, V, permutation


def derive_binary_matrix(H, V, nu):
    """Expand V^T tensor H coefficient-wise over GF(2)."""
    field_two = GF(2)
    tensor = V.transpose().tensor_product(H)
    _, _, to_vector = tensor.base_ring().vector_space(map=True)
    output = matrix(field_two, tensor.nrows() * nu, tensor.ncols())
    for row in range(tensor.nrows()):
        for column in range(tensor.ncols()):
            coefficients = to_vector(tensor[row, column])
            for bit in range(nu):
                output[row * nu + bit, column] = coefficients[bit]
    return output


def canonical_groups(N, r):
    return [list(range(start, start + r)) for start in range(0, N, r)]


def random_groups(N, r):
    indices = list(range(N))
    random.shuffle(indices)
    return [indices[start : start + r] for start in range(0, N, r)]


def binary_expansion_rows(value, nu):
    """Expand each row over GF(2), concatenating entry coordinates."""
    field_two = GF(2)
    _, _, to_vector = value.base_ring().vector_space(map=True)
    expanded = matrix(field_two, value.nrows(), value.ncols() * nu)
    for row in range(value.nrows()):
        for column in range(value.ncols()):
            coefficients = to_vector(value[row, column])
            for bit in range(nu):
                expanded[row, column * nu + bit] = coefficients[bit]
    return expanded


def group_rank(expanded_V, group):
    return expanded_V.matrix_from_rows(group).rank()


def independent_groups(expanded_V, r, attempts):
    """Find a partition whose r rows are independent inside every group."""
    N = expanded_V.nrows()
    if r > expanded_V.ncols():
        return None
    for _ in range(attempts):
        candidate = random_groups(N, r)
        if all(group_rank(expanded_V, group) == r for group in candidate):
            return candidate
    return None


def build_selected_permutation_rows(N, r, t, fixed_groups):
    """
    Select t permutation equations that stay independent on every T_Q.

    Position-side equations are selected first.  If t>N, add at most r-1
    block-side equations from each fixed group.
    """
    field_two = GF(2)
    rows = []

    # Position j across all column-blocks.
    for position in range(min(t, N)):
        row = [0] * (N * N)
        for block in range(N):
            row[block * N + position] = 1
        rows.append(row)

    remaining = t - len(rows)
    if remaining > 0:
        for group in fixed_groups:
            for block in group[: r - 1]:
                if remaining == 0:
                    break
                row = [0] * (N * N)
                for position in range(N):
                    row[block * N + position] = 1
                rows.append(row)
                remaining -= 1
            if remaining == 0:
                break

    if remaining:
        raise ValueError("could not construct enough independent permutation rows")
    return matrix(field_two, rows)


def random_partition(N, r):
    positions = list(range(N))
    random.shuffle(positions)
    return [positions[start : start + r] for start in range(0, N, r)]


def compatible_partition(permutation, fixed_groups):
    return [[permutation[block] for block in group] for group in fixed_groups]


def support_columns(N, fixed_groups, position_groups):
    columns = []
    for block_group, position_group in zip(fixed_groups, position_groups):
        for block in block_group:
            for position in position_group:
                columns.append(block * N + position)
    return columns


def corank_on_support(binary_H, selected_K, support):
    restricted = binary_H.matrix_from_columns(support).stack(
        selected_K.matrix_from_columns(support)
    )
    return restricted.ncols() - restricted.rank()


def add_record(
    records, instance, partition, source, grouping, kind, corank, group_deficiency
):
    records.append(
        {
            "instance": instance,
            "partition": partition,
            "source": source,
            "grouping": grouping,
            "kind": kind,
            "corank": int(corank),
            "two_to_corank": 2 ** int(corank),
            "group_deficiency": group_deficiency,
        }
    )


def write_raw_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[
            (record["source"], record["grouping"], record["kind"])
        ].append(record["corank"])
    rows = []
    for (source, grouping, kind), values in sorted(grouped.items()):
        rows.append(
            {
                "source": source,
                "grouping": grouping,
                "kind": kind,
                "samples": len(values),
                "invertible_probability": float(
                    sum(d == 0 for d in values) / len(values)
                ),
                "mean_two_to_corank": float(
                    sum(2 ** d for d in values) / len(values)
                ),
                "mean_corank": float(sum(values) / len(values)),
                "max_corank": max(values),
            }
        )
    return rows


def write_summary_csv(path, summary_rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)


def make_plot(path, records, title):
    styles = {
        ("derived", "canonical"): ("o-", "Derived, canonical groups"),
        ("derived", "random"): ("s-", "Derived, random groups"),
        ("derived", "adaptive"): ("^-", "Derived, independent groups"),
        ("uniform", "baseline"): ("x--", "Uniform baseline"),
    }
    grouped = defaultdict(list)
    for record in records:
        grouped[
            (record["source"], record["grouping"], record["kind"])
        ].append(record["corank"])
    max_d = max(record["corank"] for record in records)
    xs = list(range(max_d + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, kind in enumerate(("random_Q", "compatible_Q")):
        axis_pmf, axis_tail = axes[row]
        for (source, grouping), (style, label) in styles.items():
            values = grouped.get((source, grouping, kind), [])
            if not values:
                continue
            counts = Counter(values)
            pmf = [counts[d] / len(values) for d in xs]
            tail = [sum(value >= d for value in values) / len(values) for d in xs]
            axis_pmf.plot(xs, pmf, style, label=label)
            axis_tail.semilogy(xs, tail, style, label=label)

        axis_pmf.set_xlabel("corank d(Q)")
        axis_pmf.set_ylabel("empirical probability")
        axis_pmf.set_title(f"{kind}: corank distribution")
        axis_tail.set_xlabel("corank threshold d")
        axis_tail.set_ylabel("Pr[corank >= d]")
        axis_tail.set_title(f"{kind}: corank tail")
        for axis in (axis_pmf, axis_tail):
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)
            axis.set_xticks(xs)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure coranks of ISD submatrices for PKP-derived PSD "
            "instances and uniform binary controls."
        )
    )
    parser.add_argument("N", type=int, help="PKP length")
    parser.add_argument("K", type=int, help="PKP dimension")
    parser.add_argument("ell", type=int, help="number of columns of V")
    parser.add_argument("nu", type=int, help="extension degree, q=2^nu")
    parser.add_argument("--r", type=int, help="group size; default: largest feasible")
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--partitions", type=int, default=100)
    parser.add_argument(
        "--groupings",
        nargs="+",
        choices=("canonical", "random", "adaptive"),
        default=("canonical", "random", "adaptive"),
        help="derived-matrix grouping strategies to evaluate",
    )
    parser.add_argument(
        "--skip-compatible",
        action="store_true",
        help="omit the single secret-compatible support per instance",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="write CSV outputs without the per-parameter plot",
    )
    parser.add_argument(
        "--group-attempts",
        type=int,
        default=1000,
        help="random partitions tried when seeking independent groups",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-prefix", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0 < args.K < args.N and 0 < args.ell <= args.K and args.nu > 0):
        raise ValueError("require 0<K<N, 0<ell<=K, and nu>0")
    if args.instances <= 0 or args.partitions <= 0:
        raise ValueError("instances and partitions must be positive")

    random.seed(args.seed)
    set_random_seed(args.seed)
    r = choose_r(args.N, args.K, args.ell, args.nu, args.r)
    a = (args.N - args.K) * args.ell * args.nu
    t = args.N * r - a
    field = GF(2 ** args.nu, name="z")
    field_two = GF(2)

    prefix = args.output_prefix or Path(
        f"pkp_psd_rank_N{args.N}_K{args.K}_L{args.ell}_nu{args.nu}"
        f"_r{r}_I{args.instances}_Q{args.partitions}"
    )
    png_path = prefix.with_suffix(".png")
    raw_path = Path(f"{prefix}_raw.csv")
    summary_path = Path(f"{prefix}_summary.csv")
    for path in (png_path, raw_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    global_rows = []
    group_rank_rows = []
    for instance in range(args.instances):
        H, V, permutation = sample_pkp_instance(
            args.N, args.K, args.ell, field
        )
        derived = derive_binary_matrix(H, V, args.nu)
        uniform = random_matrix(field_two, a, args.N * args.N)
        expanded_source = binary_expansion_rows(V, args.nu)
        solution_permutation = permutation
        grouping_families = {"canonical": canonical_groups(args.N, r)}
        if "random" in args.groupings:
            grouping_families["random"] = random_groups(args.N, r)
        if "adaptive" in args.groupings:
            grouping_families["adaptive"] = independent_groups(
                expanded_source, r, args.group_attempts
            )
        if (
            "adaptive" in grouping_families
            and grouping_families["adaptive"] is None
        ):
            print(
                "warning: "
                f"instance {instance}: no independent grouping found after "
                f"{args.group_attempts} attempts; skipping adaptive grouping"
            )
            del grouping_families["adaptive"]

        selected_rows = {
            name: build_selected_permutation_rows(
                args.N, r, t, fixed_groups
            )
            for name, fixed_groups in grouping_families.items()
        }

        for name, fixed_groups in grouping_families.items():
            if name not in args.groupings:
                continue
            ranks = [
                group_rank(expanded_source, group) for group in fixed_groups
            ]
            deficiency = sum(r - rank for rank in ranks)
            for group_index, (group, rank) in enumerate(
                zip(fixed_groups, ranks)
            ):
                group_rank_rows.append(
                    {
                        "instance": instance,
                        "grouping": name,
                        "group": group_index,
                        "indices": " ".join(map(str, group)),
                        "rank": rank,
                        "deficiency": r - rank,
                    }
                )
            selected_K = selected_rows[name]
            global_rows.append(
                {
                    "instance": instance,
                    "grouping": name,
                    "group_deficiency": deficiency,
                    "derived_rank": derived.rank(),
                    "uniform_rank": uniform.rank(),
                    "derived_augmented_rank": derived.stack(selected_K).rank(),
                    "uniform_augmented_rank": uniform.stack(selected_K).rank(),
                }
            )

            if not args.skip_compatible:
                compatible_Q = compatible_partition(
                    solution_permutation, fixed_groups
                )
                compatible_support = support_columns(
                    args.N, fixed_groups, compatible_Q
                )
                add_record(
                    records,
                    instance,
                    -1,
                    "derived",
                    name,
                    "compatible_Q",
                    corank_on_support(
                        derived, selected_K, compatible_support
                    ),
                    deficiency,
                )

            for partition in range(args.partitions):
                Q = random_partition(args.N, r)
                support = support_columns(args.N, fixed_groups, Q)
                add_record(
                    records,
                    instance,
                    partition,
                    "derived",
                    name,
                    "random_Q",
                    corank_on_support(derived, selected_K, support),
                    deficiency,
                )

        # One uniform baseline suffices: a uniform matrix is independent of
        # the grouping and support distributions.
        baseline_groups = grouping_families["canonical"]
        baseline_K = selected_rows["canonical"]
        if not args.skip_compatible:
            compatible_Q = compatible_partition(
                solution_permutation, baseline_groups
            )
            compatible_support = support_columns(
                args.N, baseline_groups, compatible_Q
            )
            add_record(
                records,
                instance,
                -1,
                "uniform",
                "baseline",
                "compatible_Q",
                corank_on_support(uniform, baseline_K, compatible_support),
                0,
            )

        for partition in range(args.partitions):
            Q = random_partition(args.N, r)
            support = support_columns(args.N, baseline_groups, Q)
            add_record(
                records,
                instance,
                partition,
                "uniform",
                "baseline",
                "random_Q",
                corank_on_support(uniform, baseline_K, support),
                0,
            )
        print(f"completed instance {instance + 1}/{args.instances}")

    write_raw_csv(raw_path, records)
    summary_rows = summarize(records)
    write_summary_csv(summary_path, summary_rows)
    global_path = Path(f"{prefix}_global_ranks.csv")
    with global_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=global_rows[0].keys())
        writer.writeheader()
        writer.writerows(global_rows)
    group_ranks_path = Path(f"{prefix}_group_ranks.csv")
    with group_ranks_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=group_rank_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(group_rank_rows)

    if not args.skip_plot:
        make_plot(
            png_path,
            records,
            (
                f"PKP-derived vs uniform PSD: N={args.N}, K={args.K}, "
                f"L={args.ell}, nu={args.nu}, r={r}, t={t}"
            ),
        )

    print(f"\nr={r}, t={t}, binary rows a={a}, square size Nr={args.N * r}")
    for row in summary_rows:
        print(
            f"{row['source']:7s} {row['grouping']:9s} "
            f"{row['kind']:12s}: "
            f"Pr[d=0]={row['invertible_probability']:.4f}, "
            f"E[2^d]={row['mean_two_to_corank']:.4f}, "
            f"mean(d)={row['mean_corank']:.4f}, max(d)={row['max_corank']}"
        )
    if not args.skip_plot:
        print(f"Plot        : {png_path}")
    print(f"Raw samples : {raw_path}")
    print(f"Summary     : {summary_path}")
    print(f"Global ranks: {global_path}")
    print(f"Group ranks : {group_ranks_path}")


if __name__ == "__main__":
    main()
