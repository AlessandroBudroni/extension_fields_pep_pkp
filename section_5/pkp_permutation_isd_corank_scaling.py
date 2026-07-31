#!/usr/bin/env python3
"""Run the PKP-to-PSD corank experiment over growing N at fixed nu."""

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PASTEL_BLUE = "#6F9CC5"


def divisors(value):
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


def choose_r(N, K, ell, nu):
    """Return the largest feasible divisor r, or None if none exists."""
    binary_rows = (N - K) * ell * nu
    feasible = [
        r
        for r in divisors(N)
        if 0 <= N * r - binary_rows <= 2 * N - N // r
    ]
    return max(feasible) if feasible else None


def parameters(args):
    rate = Fraction(args.rate)
    if not 0 < rate < 1:
        raise ValueError("rate must lie strictly between zero and one")

    candidates = (
        args.n_values
        if args.n_values
        else range(args.n_min, args.n_max + 1, args.n_step)
    )
    selected = []
    for N in candidates:
        scaled_dimension = N * rate.numerator
        if scaled_dimension % rate.denominator:
            continue
        K = scaled_dimension // rate.denominator
        r = choose_r(N, K, args.ell, args.nu)
        if r is not None:
            selected.append((N, K, r))
    return selected


def find_summary_row(rows, source, grouping):
    for row in rows:
        if (
            row["source"] == source
            and row["grouping"] == grouping
            and row["kind"] == "random_Q"
        ):
            return row
    return None


def run_point(task):
    args, experiment_script, out_dir, N, K, r = task
    prefix = out_dir / "runs" / (
        f"rank_N{N}_K{K}_L{args.ell}_nu{args.nu}_r{r}"
        f"_I{args.instances}_Q{args.partitions}"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    log_path = prefix.with_suffix(".log")
    command = [
        args.sage,
        str(experiment_script),
        str(N),
        str(K),
        str(args.ell),
        str(args.nu),
        "--r",
        str(r),
        "--instances",
        str(args.instances),
        "--partitions",
        str(args.partitions),
        "--groupings",
        "adaptive",
        "--skip-compatible",
        "--skip-plot",
        "--group-attempts",
        str(args.group_attempts),
        "--seed",
        str(args.seed + N),
        "--output-prefix",
        str(prefix),
    ]
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"N={N} failed with exit status {result.returncode}; see {log_path}"
        )

    summary_path = Path(f"{prefix}_summary.csv")
    with summary_path.open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    derived = find_summary_row(summary, "derived", "adaptive")
    uniform = find_summary_row(summary, "uniform", "baseline")
    if derived is None:
        raise RuntimeError(
            f"N={N} produced no adaptive samples; see {log_path}"
        )
    if uniform is None:
        raise RuntimeError(f"N={N} produced no uniform samples; see {log_path}")

    adaptive_instances = int(derived["samples"]) // args.partitions
    return {
        "N": N,
        "K": K,
        "rate": float(Fraction(K, N)),
        "ell": args.ell,
        "nu": args.nu,
        "r": r,
        "adaptive_instances": adaptive_instances,
        "adaptive_fraction": adaptive_instances / args.instances,
        "derived_invertible": float(derived["invertible_probability"]),
        "uniform_invertible": float(uniform["invertible_probability"]),
        "derived_E2d": float(derived["mean_two_to_corank"]),
        "uniform_E2d": float(uniform["mean_two_to_corank"]),
        "derived_max_corank": int(derived["max_corank"]),
        "uniform_max_corank": int(uniform["max_corank"]),
    }


def write_outputs(out_dir, nu, rows):
    csv_path = out_dir / f"pkp_rank_scaling_nu{nu}.csv"
    dat_path = out_dir / f"pkp_rank_statistics_nu{nu}.dat"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with dat_path.open("w") as handle:
        handle.write(
            "N derived_E2d uniform_E2d "
            "derived_invertible uniform_invertible\n"
        )
        for row in rows:
            handle.write(
                f"{row['N']} {row['derived_E2d']:.10g} "
                f"{row['uniform_E2d']:.10g} "
                f"{row['derived_invertible']:.10g} "
                f"{row['uniform_invertible']:.10g}\n"
            )
    return csv_path, dat_path


def finish_axis(axis, ns):
    midpoint = ns[len(ns) // 2]
    axis.set_xticks(sorted({ns[0], midpoint, ns[-1]}))
    axis.set_xlabel(r"PKP length $N$")
    axis.grid(axis="y", color="0.88", linewidth=0.8)
    axis.legend(frameon=False)


def make_plots(out_dir, nu, rows):
    ns = [row["N"] for row in rows]

    probability_path = out_dir / f"invertibility_probability_nu{nu}.png"
    fig, axis = plt.subplots(figsize=(10, 3.4))
    axis.plot(
        ns,
        [row["derived_invertible"] for row in rows],
        color=PASTEL_BLUE,
        linewidth=2.2,
        label="PKP-derived",
    )
    axis.plot(
        ns,
        [row["uniform_invertible"] for row in rows],
        color=PASTEL_BLUE,
        linewidth=2.2,
        linestyle="--",
        label="Uniformly random",
    )
    axis.set_ylim(0, 1)
    axis.set_yticks([0, 1])
    axis.set_ylabel(r"$\Pr[d(Q)=0]$")
    axis.set_title(rf"$\nu={nu}$")
    finish_axis(axis, ns)
    fig.tight_layout()
    fig.savefig(probability_path, dpi=220)
    plt.close(fig)

    expectation_path = out_dir / f"expected_two_to_corank_nu{nu}.png"
    fig, axis = plt.subplots(figsize=(10, 3.4))
    axis.plot(
        ns,
        [row["derived_E2d"] for row in rows],
        color=PASTEL_BLUE,
        linewidth=2.2,
        label="PKP-derived",
    )
    axis.plot(
        ns,
        [row["uniform_E2d"] for row in rows],
        color=PASTEL_BLUE,
        linewidth=2.2,
        linestyle="--",
        label="Uniformly random",
    )
    axis.set_ylim(1.6, 2.4)
    axis.set_yticks([1.6, 2.4])
    axis.set_ylabel(r"$\mathbb{E}[2^{d(Q)}]$")
    axis.set_title(rf"$\nu={nu}$")
    finish_axis(axis, ns)
    fig.tight_layout()
    fig.savefig(expectation_path, dpi=220)
    plt.close(fig)
    return probability_path, expectation_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "At fixed nu and code rate, run the adaptive PKP-to-PSD corank "
            "experiment for growing N and produce aggregate data and plots."
        )
    )
    parser.add_argument("--nu", type=int, required=True, help="q=2^nu")
    parser.add_argument("--rate", default="1/2", help="PKP rate K/N")
    parser.add_argument("--ell", type=int, default=1)
    parser.add_argument("--n-min", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=128)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        help="explicit N values; overrides --n-min, --n-max, and --n-step",
    )
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--partitions", type=int, default=100)
    parser.add_argument("--group-attempts", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sage", default="sage", help="SageMath executable")
    parser.add_argument("--out-dir", type=Path, default=Path("rank_scaling"))
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.nu <= 0
        or args.ell <= 0
        or args.instances <= 0
        or args.partitions <= 0
        or args.jobs <= 0
        or args.n_step <= 0
    ):
        raise ValueError(
            "nu, ell, instances, partitions, jobs, and n-step must be positive"
        )
    if not args.n_values and (args.n_min <= 0 or args.n_min > args.n_max):
        raise ValueError("require 0 < n-min <= n-max")

    selected = parameters(args)
    if not selected:
        raise ValueError("no feasible parameter sets in the requested N range")

    script_dir = Path(__file__).resolve().parent
    experiment_script = script_dir / "pkp_permutation_isd_corank_experiment.py"
    if not experiment_script.exists():
        raise FileNotFoundError(experiment_script)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Selected parameter sets:")
    for N, K, r in selected:
        print(f"  N={N:3d}, K={K:3d}, r={r}")

    tasks = [
        (args, experiment_script, out_dir, N, K, r)
        for N, K, r in selected
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_point, task): task[3] for task in tasks}
        for future in as_completed(futures):
            N = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"completed N={N}: "
                f"adaptive fraction={row['adaptive_fraction']:.3f}, "
                f"Pr[d=0]={row['derived_invertible']:.4f}/"
                f"{row['uniform_invertible']:.4f}, "
                f"E[2^d]={row['derived_E2d']:.4f}/"
                f"{row['uniform_E2d']:.4f}"
            )
    rows.sort(key=lambda row: row["N"])

    csv_path, dat_path = write_outputs(out_dir, args.nu, rows)
    probability_path, expectation_path = make_plots(out_dir, args.nu, rows)
    print(f"\nFull data     : {csv_path}")
    print(f"Paper data    : {dat_path}")
    print(f"Probability   : {probability_path}")
    print(f"Expected cost : {expectation_path}")


if __name__ == "__main__":
    main()
