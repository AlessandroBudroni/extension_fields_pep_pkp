    # Solver for PCE between self-orthogonal codes defined over field extension

import os
import sys
import argparse
from contextlib import redirect_stdout
from time import time
from multiprocessing import Pool
from abl_ec25 import random_selforthogonal_code as random_selforthogonal_code
from bms_eprint25 import solve_GIP
import numpy as np
import matplotlib.pyplot as plt
from random import randint, shuffle

# SageMath imports
from sage.all import (
    matrix,
    zero_matrix,
    FiniteField,
    GF,
    ZZ,
)

from common import (
    parse_args,
    parse_range,
    pi,
    tau,
    sample_full_rank_matrix,
    sample_permutation_matrix,
)


def _init_worker():
    # each forked worker inherits the parent's RNG state, so reseed per-process
    seed = os.getpid() ^ int(time() * 1e6)
    import random as _random
    _random.seed(seed)
    from sage.all import set_random_seed
    set_random_seed(seed)

def run_trial(args):
    n, k, p, nu, q, v = args
    G = None
    while G is None:
        G = random_selforthogonal_code(n, k, q, verbose=v)
    Gs = pi(G, ZZ(p))
    GsG = Gs * G.transpose()
    if GsG.rank() < k:
        return True, False  # rank_deficient, wrong_solution

    # check that PCE instance can be solved
    S = sample_full_rank_matrix(k, k, q)
    P = sample_permutation_matrix(n, q)
    G1 = S * G * P
    G1s = pi(G1, ZZ(p))
    Gb = tau(G, Gs)
    G1b = tau(G1, G1s)
    Psol = solve_GIP(GF(q), Gb, G1b, n)
    return False, Psol != P

def write_data(n_list, p_list, nu_list, prob):
    print("n p nu prob")
    for i in range(len(n_list)):
        for j in range(len(p_list)):
            print(f"{n_list[i]} {p_list[j]} {nu_list[j]} {prob[i][j]}")



if __name__ == '__main__':

    args = parse_args()
    nn = parse_range(args.n)
    pp = parse_range(args.p)
    nunu = parse_range(args.nu)
    Nexp = args.nexp
    restore = args.restore
    jobs = args.j

    if not pp or not nunu or not nn:
        print(
            "Error: --n, --p, and --nu are all required, must be non-empty, " \
            "and at least one of --p or --nu must have length 1.\n"
            "Example: sage -python so_frob_rank_experiments.py "
            "--n 4,20,4 --p 2,3,5,7 --nu 4"
        )
        sys.exit(1)

    if len(nunu)>=1 and len(pp)==1:
        _nu = nunu
        _p = [pp[0]]*len(nunu)
    elif len(pp)>=1 and len(nunu)==1:
        _nu = [nunu[0]]*len(pp)
        _p = pp
    else:
        print(
            "Error: --n, --p, and --nu are all required, must be non-empty, " \
            "and at least one of --p or --nu must have length 1.\n"
            "Example: sage -python so_frob_rank_experiments.py "
            "--n 4,20,4 --p 2,3,5,7 --nu 4"
        )
        sys.exit(1)


    # Check if previous results are available
    os.makedirs("results", exist_ok=True)
    restored=False
    if restore:
        try:
            ranks,wrong = np.load(f"results/rank_deficiency_results_p={pp}_n={nn}_nu={nunu}.npy", allow_pickle=True)
            restored=True
        except:
            print("Could not find", f"rank_deficiency_results_p={pp}_n={nn}_nu={nunu}.npy\nRedoing experiments..")

    # Run experiments
    if not restored:
        ranks = []
        wrong = []
        for n in nn:
            ranks_n = []
            wrong_n = []
            for j in range(len(_p)):
                p = _p[j]
                nu = _nu[j]
                q = ZZ(p**nu)
                if p > 2: k = n // 2
                else: k = n // 2 - 1
                rank_deficient = 0
                wrong_count = 0
                trial_args = [(n, k, p, nu, q, jobs==1)] * Nexp
                with Pool(processes=jobs, initializer=_init_worker) as pool:
                    for trial, (is_deficient, is_wrong) in enumerate(pool.imap_unordered(run_trial, trial_args)):
                        if is_deficient:
                            rank_deficient += 1
                        elif is_wrong:
                            wrong_count += 1
                        print(f"Processing n={n}, k={k}, p={p}, nu={nu}, deficient {rank_deficient}, trial {trial+1}/{Nexp}")
                ranks_n.append(rank_deficient/Nexp)
                wrong_n.append(wrong_count/(Nexp - rank_deficient))
            ranks.append(ranks_n)
            wrong.append(wrong_n)
        np.save(f"results/rank_deficiency_results_p={pp}_n={nn}_nu={nunu}.npy", np.array([ranks,wrong]))

    # Do the tables
    with open(f"results/rank_deficiency_results_p={pp}_n={nn}_nu={nunu}.txt", "w") as f, redirect_stdout(f):
        write_data(nn, _p, _nu, ranks)
    with open(f"results/wrong_result_prob_p={pp}_n={nn}_nu={nunu}.txt", "w") as f, redirect_stdout(f):
        write_data(nn, _p, _nu, wrong)
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
    plt.figure(figsize=(8, 6))
    n_cont = np.linspace(min(nn), max(nn), 200)
    if(len(pp) > 1):
            nu = nunu[0]
            p_cont = np.linspace(min(pp), max(pp), 200)
            for i, n in enumerate(nn):
                plt.plot(pp, ranks[i], color=colors[i], marker='o', label=f'n = {n}')
                exponent = (n * (nu / 2 - 1) + 1)
                # if i == 0: plt.plot(p_cont, 1 / p_cont**exponent, color='#000000', linestyle=':')
            plt.xlabel('p')
            plt.title(f'Rank deficiency probability vs. p (q=p^{nu}, {Nexp} trials each)')
            plt.legend(title='')
    elif(len(nunu) > 1):
        p = pp[0]
        nu_cont = np.linspace(min(nunu), max(nunu), 200)
        solid_ys = []
        for i, n in enumerate(nn):
            solid_ys.extend(ranks[i])
            plt.plot(nunu, ranks[i], color=colors[i], marker='o', label=f'n = {n}')
            exponent = (n * (nu_cont / 2 - 1) + 1)
            # plt.plot(nu_cont, 1 / p**exponent, color=colors[i], linestyle=':')
        plt.xlabel('nu')
        plt.title(f'Rank deficiency probability vs. nu (q={p}^nu, {Nexp} trials each)')
        plt.legend(title='')
    plt.ylabel('Rank-deficiency probability')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/rank_deficiency_results_{pp}_{nn}_{nunu}.png", dpi=150)
    plt.show()

