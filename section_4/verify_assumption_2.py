#!/usr/bin/env sage
"""Empirically test the rank claim in Assumption 2 for structured PEP."""

import argparse
import csv
import math
import multiprocessing
import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sage.all import (
    GF,
    Integer,
    matrix,
    random_matrix,
    set_random_seed,
    vector,
    zero_matrix,
)

from abl_ec25 import random_selforthogonal_code as random_selforthogonal_code_fe


PASTEL_BLUE = "#6F9CC5"


def random_invertible_matrix(field, size):
    while True:
        value = random_matrix(field, size, size)
        if value.is_invertible():
            return value


def random_permutation_matrix(field, n):
    permutation = list(range(n))
    random.shuffle(permutation)
    value = matrix(field, n, n, sparse=True)
    for row, column in enumerate(permutation):
        value[row, column] = 1
    return value


def extension_field(nu):
    """Return the field the sampler works over, so parents always match."""
    return GF(Integer(2) ** Integer(nu))


def random_self_orthogonal_generator(field, n, k, max_tries):
    """Sample a uniformly random self-orthogonal code with the [AlbBenLai25] sampler."""
    if not 0 < k <= n // 2:
        raise ValueError("self-orthogonal codes require 0 < k <= n/2")
    for _ in range(max_tries):
        # The sampler is Las Vegas: None signals a rejection, so restart it.
        G = random_selforthogonal_code_fe(n, k, field.order())
        if G is None:
            continue
        if G.rank() != k or G * G.transpose() != 0:
            raise AssertionError("invalid self-orthogonal generator")
        return G
    raise RuntimeError(
        f"failed to sample a self-orthogonal generator after {max_tries} trials"
    )


def sample_pep_instance(field, n, k, max_tries):
    G = random_self_orthogonal_generator(field, n, k, max_tries)
    P = random_permutation_matrix(field, n)
    S = random_invertible_matrix(field, k)
    G_prime = S * G * P
    H_prime = G_prime.right_kernel().basis_matrix()
    if G_prime * H_prime.transpose() != 0:
        raise AssertionError("invalid parity-check matrix for equivalent code")
    system = G.tensor_product(H_prime)
    solution_extension = vector(field, P.list())
    if system * solution_extension != 0:
        raise AssertionError("secret permutation does not solve PEP system")
    return system, P


def project_matrix_to_binary(value):
    """Expand coefficients over GF(2), leaving the binary variables unchanged."""
    field_two = GF(2)
    field = value.base_ring()
    nu = field.degree()
    _, _, to_vector = field.vector_space(map=True)
    output = matrix(field_two, value.nrows() * nu, value.ncols())
    for row in range(value.nrows()):
        for column in range(value.ncols()):
            coordinates = to_vector(value[row, column])
            for bit in range(nu):
                output[row * nu + bit, column] = coordinates[bit]
    return output


def binary_permutation_vector(P):
    return vector(GF(2), [int(entry) for entry in P.list()])


def structural_kernel_basis(n, solution, case):
    """Return the known structural kernel, including the secret permutation."""
    field_two = GF(2)
    rows = []

    # With Sage's row-major tensor convention, G tensor H' acts as
    # X -> G X H'^T.  Thus G 1 = 0 gives the matrices 1 u^T.
    for column in range(n):
        value = [0] * (n * n)
        for row in range(n):
            value[row * n + column] = 1
        rows.append(value)

    if case == "self_dual":
        # Self-duality also gives H' 1 = 0, hence the matrices u 1^T.
        # One row is omitted because the all-one matrix is already in the
        # span of the column indicators above.
        for row in range(n - 1):
            value = [0] * (n * n)
            for column in range(n):
                value[row * n + column] = 1
            rows.append(value)

    rows.append(list(solution))
    basis = matrix(field_two, rows)
    expected_dimension = n + 1 if case == "self_orthogonal" else 2 * n
    if basis.rank() != expected_dimension:
        raise AssertionError("known structural kernel has unexpected dimension")
    return basis


def rank_record(source, value, case, n, k, nu, d, instance):
    binary_rows = (n - k) * k * nu
    target = min(binary_rows, n * n - d)
    rank_value = int(value.rank())
    return {
        "case": case,
        "n": n,
        "k": k,
        "rate": k / n,
        "nu": nu,
        "binary_rows": binary_rows,
        "structural_kernel_dimension": d,
        "target_rank": target,
        "instance": instance,
        "source": source,
        "rank_matrix": rank_value,
        "kernel_dimension": n * n - rank_value,
        "expected_kernel_dimension": n * n - target,
        "deficiency": target - rank_value,
    }


def run_parameter_set(task):
    case, n, k, nu, instances, seed, max_tries = task
    random.seed(seed)
    set_random_seed(seed)
    field = extension_field(nu)
    d = n + 1 if case == "self_orthogonal" else 2 * n
    records = []
    for instance in range(instances):
        system, P = sample_pep_instance(field, n, k, max_tries)
        derived = project_matrix_to_binary(system)
        solution = binary_permutation_vector(P)
        if derived * solution != 0:
            raise AssertionError("projected PEP system lost the secret solution")
        known_kernel = structural_kernel_basis(n, solution, case)
        if derived * known_kernel.transpose() != 0:
            raise AssertionError("derived matrix lost a structural kernel vector")
        records.append(
            rank_record("pep_derived", derived, case, n, k, nu, d, instance)
        )
    return records


def nearest_rank_quantile(values, probability):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault((record["n"], record["source"]), []).append(record)
    rows = []
    for (n, source), group in sorted(groups.items()):
        deficiencies = [row["deficiency"] for row in group]
        first = group[0]
        log_bound = math.ceil(math.log2(n))
        rows.append(
            {
                "case": first["case"],
                "n": n,
                "k": first["k"],
                "rate": first["rate"],
                "nu": first["nu"],
                "binary_rows": first["binary_rows"],
                "structural_kernel_dimension": first[
                    "structural_kernel_dimension"
                ],
                "target_rank": first["target_rank"],
                "source": source,
                "instances": len(group),
                "probability_deficiency_zero": sum(d == 0 for d in deficiencies)
                / len(deficiencies),
                "probability_deficiency_at_most_log2_n": sum(
                    d <= log_bound for d in deficiencies
                )
                / len(deficiencies),
                "mean_deficiency": statistics.mean(deficiencies),
                "median_deficiency": statistics.median(deficiencies),
                "p95_deficiency": nearest_rank_quantile(deficiencies, 0.95),
                "p99_deficiency": nearest_rank_quantile(deficiencies, 0.99),
                "max_deficiency": max(deficiencies),
                "mean_kernel_dimension": statistics.mean(
                    row["kernel_dimension"] for row in group
                ),
                "max_kernel_dimension": max(row["kernel_dimension"] for row in group),
            }
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_dat(path, summaries):
    columns = [
        "n",
        "k",
        "nu",
        "derived_mean",
        "derived_median",
        "derived_p95",
        "derived_p99",
        "derived_max",
    ]
    with path.open("w") as handle:
        handle.write(" ".join(columns) + "\n")
        for derived in summaries_for(summaries, "pep_derived"):
            values = [
                derived["n"],
                derived["k"],
                derived["nu"],
                derived["mean_deficiency"],
                derived["median_deficiency"],
                derived["p95_deficiency"],
                derived["p99_deficiency"],
                derived["max_deficiency"],
            ]
            handle.write(" ".join(str(value) for value in values) + "\n")


def summaries_for(summaries, source):
    return sorted(
        (row for row in summaries if row["source"] == source),
        key=lambda row: row["n"],
    )


def make_main_plot(path, records, summaries, title):
    derived = summaries_for(summaries, "pep_derived")
    ns = [row["n"] for row in derived]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    axis = axes[0]
    axis.plot(
        ns,
        [row["median_deficiency"] for row in derived],
        color=PASTEL_BLUE,
        linewidth=2.2,
        label="PEP-derived median",
    )
    axis.plot(
        ns,
        [row["p95_deficiency"] for row in derived],
        color=PASTEL_BLUE,
        linewidth=2.2,
        linestyle="--",
        label="PEP-derived 95th percentile",
    )
    axis.plot(
        ns,
        [row["max_deficiency"] for row in derived],
        color=PASTEL_BLUE,
        linewidth=2.0,
        linestyle=":",
        label="PEP-derived maximum",
    )
    axis.plot(
        ns,
        [math.log2(n) for n in ns],
        color="0.45",
        linewidth=1.8,
        linestyle="-.",
        label=r"$\log_2 n$",
    )
    axis.set_xlabel(r"PEP length $n$")
    axis.set_ylabel(r"rank deficiency $D$")
    axis.grid(axis="y", color="0.88", linewidth=0.8)
    axis.legend(frameon=False, fontsize=8)

    largest_n = max(ns)
    largest = [row for row in records if row["n"] == largest_n]
    derived_values = [
        row["deficiency"] for row in largest if row["source"] == "pep_derived"
    ]
    maximum = max(derived_values)
    thresholds = list(range(maximum + 2))
    axis = axes[1]
    axis.step(
        thresholds,
        [
            sum(value >= d for value in derived_values) / len(derived_values)
            for d in thresholds
        ],
        where="post",
        color=PASTEL_BLUE,
        linewidth=2.2,
        label="PEP-derived",
    )
    axis.set_xlabel(r"deficiency threshold $d$")
    axis.set_ylabel(r"$\Pr[D\geq d]$")
    axis.set_ylim(0, 1.02)
    axis.set_xticks(thresholds)
    axis.set_title(rf"{title}, $n={largest_n}$")
    axis.grid(axis="y", color="0.88", linewidth=0.8)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_kernel_plot(path, summaries, title):
    derived = summaries_for(summaries, "pep_derived")
    ns = [row["n"] for row in derived]
    expected = [
        row["n"] ** 2 - row["target_rank"] for row in derived
    ]
    fig, axis = plt.subplots(figsize=(8.5, 4.1))
    axis.plot(
        ns,
        [row["mean_kernel_dimension"] for row in derived],
        color=PASTEL_BLUE,
        linewidth=2.2,
        label="PEP-derived mean",
    )
    axis.plot(
        ns,
        [row["max_kernel_dimension"] for row in derived],
        color=PASTEL_BLUE,
        linewidth=2.0,
        linestyle=":",
        label="PEP-derived maximum",
    )
    axis.plot(
        ns,
        expected,
        color="0.45",
        linewidth=1.8,
        linestyle="-.",
        label="predicted kernel dimension",
    )
    axis.set_xlabel(r"PEP length $n$")
    axis.set_ylabel("kernel dimension")
    axis.set_title(title)
    axis.grid(axis="y", color="0.88", linewidth=0.8)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def selected_parameters(args, case):
    if args.n_values is None:
        ns = (
            (24, 30, 36, 48, 60)
            if case == "self_orthogonal"
            else (20, 24, 32, 48, 64)
        )
    else:
        ns = args.n_values

    rate = Fraction(args.rate or "1/3")
    output = []
    for n in ns:
        if case == "self_dual":
            if n % 2:
                raise ValueError("self-dual mode requires even n")
            k = n // 2
        else:
            numerator = n * rate.numerator
            if numerator % rate.denominator:
                raise ValueError(f"rate {rate} does not give integral k for n={n}")
            k = numerator // rate.denominator
            if not 0 < k < n / 2:
                raise ValueError("self-orthogonal mode requires 0 < k < n/2")
        d = n + 1 if case == "self_orthogonal" else 2 * n
        boundary_nu = math.ceil((n * n - d) / ((n - k) * k))
        nu = args.nu if args.nu is not None else boundary_nu + args.nu_offset
        if nu <= 0:
            raise ValueError("nu must be positive")
        output.append((n, k, nu, d))
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Assumption 2 for self-orthogonal or self-dual PEP."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-orthogonal", action="store_true")
    modes.add_argument("--self-dual", action="store_true")
    parser.add_argument("--n-values", type=int, nargs="+")
    parser.add_argument(
        "--rate",
        help="k/n in self-orthogonal mode (default: 1/3)",
    )
    parser.add_argument("--nu", type=int)
    parser.add_argument("--nu-offset", type=int, default=0)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-tries", type=int, default=10000)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.instances <= 0 or args.jobs <= 0 or args.max_tries <= 0:
        raise ValueError("instances, jobs, and max-tries must be positive")
    case = "self_orthogonal" if args.self_orthogonal else "self_dual"
    if case == "self_dual" and args.rate is not None:
        raise ValueError("--rate is fixed to 1/2 in self-dual mode")
    selected = selected_parameters(args, case)
    for n, k, nu, d in selected:
        rows = (n - k) * k * nu
        print(
            f"case={case:15s}, n={n:3d}, k={k:3d}, nu={nu:2d}, "
            f"binary rows={rows}, d={d}, target={min(rows, n*n-d)}"
        )

    tasks = [
        (
            case,
            n,
            k,
            nu,
            args.instances,
            args.seed + 1000003 * index,
            args.max_tries,
        )
        for index, (n, k, nu, _) in enumerate(selected)
    ]
    records = []
    if args.jobs == 1:
        for task in tasks:
            records.extend(run_parameter_set(task))
            print(f"completed n={task[1]}")
    else:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=args.jobs, mp_context=context
        ) as executor:
            futures = {
                executor.submit(run_parameter_set, task): task[1] for task in tasks
            }
            for future in as_completed(futures):
                n = futures[future]
                records.extend(future.result())
                print(f"completed n={n}")

    records.sort(key=lambda row: (row["n"], row["instance"], row["source"]))
    summaries = summarize(records)
    out_dir = (
        args.out_dir or Path(f"assumption_2_{case}")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"assumption_2_{case}"
    raw_path = out_dir / f"{stem}_raw.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    dat_path = out_dir / f"{stem}_summary.dat"
    plot_path = out_dir / f"{stem}_rank_deficiency.png"
    kernel_path = out_dir / f"{stem}_kernel_dimension.png"
    write_csv(raw_path, records)
    write_csv(summary_path, summaries)
    write_dat(dat_path, summaries)
    title = "self-orthogonal" if case == "self_orthogonal" else "self-dual"
    make_main_plot(plot_path, records, summaries, title)
    make_kernel_plot(kernel_path, summaries, title)

    print()
    for row in summaries:
        print(
            f"n={row['n']:3d} {row['source']:14s}: "
            f"Pr[D=0]={row['probability_deficiency_zero']:.3f}, "
            f"mean={row['mean_deficiency']:.3f}, "
            f"p95={row['p95_deficiency']}, max={row['max_deficiency']}"
        )
    print(f"\nRaw data  : {raw_path}")
    print(f"Summary   : {summary_path}")
    print(f"Paper data: {dat_path}")
    print(f"Main plot : {plot_path}")
    print(f"Kernel plot: {kernel_path}")


if __name__ == "__main__":
    main()
