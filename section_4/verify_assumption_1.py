#!/usr/bin/env sage
"""Empirically test the augmented-rank claim in Assumption 1."""

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
from sage.all import GF, matrix, random_matrix, set_random_seed, vector


PASTEL_BLUE = "#6F9CC5"


def random_full_rank_matrix(field, nrows, ncols):
    while True:
        value = random_matrix(field, nrows, ncols)
        if value.rank() == nrows:
            return value


def sample_random_pkp(n, k, ell, field):
    """Sample a random PKP instance as in the definition in the paper."""
    H = random_full_rank_matrix(field, n - k, n)
    kernel_basis = H.right_kernel().basis_matrix()
    coefficients = random_full_rank_matrix(field, ell, k)
    kernel_vectors = (coefficients * kernel_basis).transpose()

    permutation = list(range(n))
    random.shuffle(permutation)
    E = matrix(field, n, n, sparse=True)
    for column, row in enumerate(permutation):
        E[row, column] = 1
    V = E.transpose() * kernel_vectors
    if H * E * V != 0:
        raise AssertionError("sampled PKP solution is invalid")
    return H, V, permutation


def binary_expansion_of_tensor(H, V, nu):
    """Construct the binary expansion of V^T tensor H."""
    field_two = GF(2)
    tensor = V.transpose().tensor_product(H)
    _, _, to_vector = tensor.base_ring().vector_space(map=True)
    output = matrix(field_two, tensor.nrows() * nu, tensor.ncols())
    for row in range(tensor.nrows()):
        for column in range(tensor.ncols()):
            coordinates = to_vector(tensor[row, column])
            for bit in range(nu):
                output[row * nu + bit, column] = coordinates[bit]
    return output


def permutation_equation_matrix(n):
    """Return the 2n permutation equations, whose rank is 2n-1."""
    field_two = GF(2)
    K = matrix(field_two, 2 * n, n * n, sparse=True)
    for block in range(n):
        for position in range(n):
            K[block, block * n + position] = 1
            K[n + position, block * n + position] = 1
    if K.rank() != 2 * n - 1:
        raise AssertionError("permutation-equation matrix has unexpected rank")
    return K


def solution_vector(n, permutation):
    values = [0] * (n * n)
    for block, position in enumerate(permutation):
        values[block * n + position] = 1
    return vector(GF(2), values)


def rank_record(source, value, K, n, k, ell, nu, instance):
    binary_rows = (n - k) * ell * nu
    target = min(n * n, binary_rows + 2 * n - 1)
    rank_value = int(value.rank())
    rank_augmented = int(value.stack(K).rank())
    h_target = min(binary_rows, n * n - 1)
    expected_gain = min(2 * n - 1, n * n - rank_value)
    actual_gain = rank_augmented - rank_value
    return {
        "n": n,
        "k": k,
        "rate": k / n,
        "ell": ell,
        "nu": nu,
        "binary_rows": binary_rows,
        "target_rank": target,
        "instance": instance,
        "source": source,
        "rank_matrix": rank_value,
        "rank_augmented": rank_augmented,
        "matrix_deficiency": h_target - rank_value,
        "K_rank_gain": actual_gain,
        "K_gain_deficiency": expected_gain - actual_gain,
        "deficiency": target - rank_augmented,
    }


def run_parameter_set(task):
    n, k, ell, nu, instances, seed = task
    random.seed(seed)
    set_random_seed(seed)
    field = GF(2**nu, name=f"z{nu}")
    K = permutation_equation_matrix(n)
    records = []

    for instance in range(instances):
        H, V, permutation = sample_random_pkp(n, k, ell, field)
        derived = binary_expansion_of_tensor(H, V, nu)
        solution = solution_vector(n, permutation)
        if derived * solution != 0:
            raise AssertionError("derived matrix does not annihilate PKP solution")
        records.append(
            rank_record("pkp_derived", derived, K, n, k, ell, nu, instance)
        )
    return records


def nearest_rank_quantile(values, probability):
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def summarize(records):
    groups = {}
    for record in records:
        key = (record["n"], record["source"])
        groups.setdefault(key, []).append(record)

    rows = []
    for (n, source), group in sorted(groups.items()):
        deficiencies = [row["deficiency"] for row in group]
        first = group[0]
        log_bound = math.ceil(math.log2(n))
        rows.append(
            {
                "n": n,
                "k": first["k"],
                "rate": first["rate"],
                "ell": first["ell"],
                "nu": first["nu"],
                "binary_rows": first["binary_rows"],
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
                "mean_matrix_deficiency": statistics.mean(
                    row["matrix_deficiency"] for row in group
                ),
                "max_matrix_deficiency": max(
                    row["matrix_deficiency"] for row in group
                ),
                "mean_K_gain_deficiency": statistics.mean(
                    row["K_gain_deficiency"] for row in group
                ),
                "max_K_gain_deficiency": max(
                    row["K_gain_deficiency"] for row in group
                ),
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
        "ell",
        "nu",
        "derived_mean",
        "derived_median",
        "derived_p95",
        "derived_p99",
        "derived_max",
    ]
    with path.open("w") as handle:
        handle.write(" ".join(columns) + "\n")
        for derived in summary_by_source(summaries, "pkp_derived"):
            values = [
                derived["n"],
                derived["k"],
                derived["ell"],
                derived["nu"],
                derived["mean_deficiency"],
                derived["median_deficiency"],
                derived["p95_deficiency"],
                derived["p99_deficiency"],
                derived["max_deficiency"],
            ]
            handle.write(" ".join(str(value) for value in values) + "\n")


def summary_by_source(summaries, source):
    return sorted(
        (row for row in summaries if row["source"] == source),
        key=lambda row: row["n"],
    )


def make_main_plot(path, records, summaries):
    derived = summary_by_source(summaries, "pkp_derived")
    ns = [row["n"] for row in derived]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    axis = axes[0]
    axis.plot(
        ns,
        [row["median_deficiency"] for row in derived],
        color=PASTEL_BLUE,
        linewidth=2.2,
        label="PKP-derived median",
    )
    axis.plot(
        ns,
        [row["p95_deficiency"] for row in derived],
        color=PASTEL_BLUE,
        linewidth=2.2,
        linestyle="--",
        label="PKP-derived 95th percentile",
    )
    axis.plot(
        ns,
        [row["max_deficiency"] for row in derived],
        color=PASTEL_BLUE,
        linewidth=2.0,
        linestyle=":",
        label="PKP-derived maximum",
    )
    axis.plot(
        ns,
        [math.log2(n) for n in ns],
        color="0.45",
        linewidth=1.8,
        linestyle="-.",
        label=r"$\log_2 n$",
    )
    axis.set_xlabel(r"PKP length $n$")
    axis.set_ylabel(r"rank deficiency $D$")
    axis.grid(axis="y", color="0.88", linewidth=0.8)
    axis.legend(frameon=False, fontsize=8)

    largest_n = max(ns)
    largest = [row for row in records if row["n"] == largest_n]
    derived_values = [
        row["deficiency"] for row in largest if row["source"] == "pkp_derived"
    ]
    maximum = max(derived_values)
    thresholds = list(range(maximum + 2))
    axis = axes[1]
    axis.step(
        thresholds,
        [sum(value >= d for value in derived_values) / len(derived_values) for d in thresholds],
        where="post",
        color=PASTEL_BLUE,
        linewidth=2.2,
        label="PKP-derived",
    )
    axis.set_xlabel(r"deficiency threshold $d$")
    axis.set_ylabel(r"$\Pr[D\geq d]$")
    axis.set_ylim(0, 1.02)
    axis.set_xticks(thresholds)
    axis.set_title(rf"$n={largest_n}$")
    axis.grid(axis="y", color="0.88", linewidth=0.8)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_diagnostic_plot(path, summaries):
    derived = summary_by_source(summaries, "pkp_derived")
    ns = [row["n"] for row in derived]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    for axis, mean_key, max_key, title in (
        (
            axes[0],
            "mean_matrix_deficiency",
            "max_matrix_deficiency",
            r"deficiency of $\bar{H}$",
        ),
        (
            axes[1],
            "mean_K_gain_deficiency",
            "max_K_gain_deficiency",
            r"missing rank contribution from $K_n$",
        ),
    ):
        axis.plot(
            ns,
            [row[mean_key] for row in derived],
            color=PASTEL_BLUE,
            linewidth=2.2,
            label="PKP-derived mean",
        )
        axis.plot(
            ns,
            [row[max_key] for row in derived],
            color=PASTEL_BLUE,
            linewidth=2.0,
            linestyle=":",
            label="PKP-derived maximum",
        )
        axis.set_xlabel(r"PKP length $n$")
        axis.set_ylabel("deficiency")
        axis.set_title(title)
        axis.grid(axis="y", color="0.88", linewidth=0.8)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parameter_sets(args):
    rate = Fraction(args.rate)
    if not 0 < rate < 1:
        raise ValueError("rate must lie strictly between zero and one")
    output = []
    for n in args.n_values:
        numerator = n * rate.numerator
        if numerator % rate.denominator:
            raise ValueError(f"rate {rate} does not give integral k for n={n}")
        k = numerator // rate.denominator
        if not 0 < args.ell < k < n:
            raise ValueError(f"require 0 < ell < k < n, failed for n={n}")
        denominator = (n - k) * args.ell
        boundary_nu = math.ceil((n - 1) ** 2 / denominator)
        nu = args.nu if args.nu is not None else boundary_nu + args.nu_offset
        if nu <= 0:
            raise ValueError(f"nu must be positive, failed for n={n}")
        output.append((n, k, args.ell, nu))
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test the augmented-rank deficiency in Assumption 1 on random "
            "PKP instances."
        )
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=(20, 24, 32, 48, 64),
    )
    parser.add_argument("--rate", default="1/2", help="PKP rate k/n")
    parser.add_argument("--ell", type=int, default=1)
    parser.add_argument(
        "--nu",
        type=int,
        help="fixed extension degree; default: polynomial-boundary value per n",
    )
    parser.add_argument(
        "--nu-offset",
        type=int,
        default=0,
        help="offset from the boundary value when --nu is omitted",
    )
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("assumption_1"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.instances <= 0 or args.jobs <= 0:
        raise ValueError("instances and jobs must be positive")
    selected = parameter_sets(args)
    for n, k, ell, nu in selected:
        binary_rows = (n - k) * ell * nu
        print(
            f"n={n:3d}, k={k:3d}, ell={ell}, nu={nu:3d}, "
            f"binary rows={binary_rows}, target="
            f"{min(n*n, binary_rows + 2*n - 1)}"
        )

    tasks = [
        (n, k, ell, nu, args.instances, args.seed + 1000003 * index)
        for index, (n, k, ell, nu) in enumerate(selected)
    ]
    records = []
    if args.jobs == 1:
        for task in tasks:
            result = run_parameter_set(task)
            records.extend(result)
            print(f"completed n={task[0]}")
    else:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=args.jobs, mp_context=context
        ) as executor:
            futures = {
                executor.submit(run_parameter_set, task): task[0] for task in tasks
            }
            for future in as_completed(futures):
                n = futures[future]
                records.extend(future.result())
                print(f"completed n={n}")

    records.sort(key=lambda row: (row["n"], row["instance"], row["source"]))
    summaries = summarize(records)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "assumption_1_raw.csv"
    summary_path = out_dir / "assumption_1_summary.csv"
    dat_path = out_dir / "assumption_1_summary.dat"
    plot_path = out_dir / "assumption_1_rank_deficiency.png"
    diagnostics_path = out_dir / "assumption_1_diagnostics.png"
    write_csv(raw_path, records)
    write_csv(summary_path, summaries)
    write_dat(dat_path, summaries)
    make_main_plot(plot_path, records, summaries)
    make_diagnostic_plot(diagnostics_path, summaries)

    print()
    for row in summaries:
        print(
            f"n={row['n']:3d} {row['source']:14s}: "
            f"Pr[D=0]={row['probability_deficiency_zero']:.3f}, "
            f"mean={row['mean_deficiency']:.3f}, "
            f"p95={row['p95_deficiency']}, max={row['max_deficiency']}"
        )
    print(f"\nRaw data   : {raw_path}")
    print(f"Summary    : {summary_path}")
    print(f"Paper data : {dat_path}")
    print(f"Main plot  : {plot_path}")
    print(f"Diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
