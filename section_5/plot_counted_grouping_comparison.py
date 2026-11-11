#!/usr/bin/env python3
"""Plot the Section 5.3 complexity comparison (Figure 2).

PSD uses the single-solution bound of Section 5.2 with three spare equations.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from counted_grouping import counted_grouping_estimate, prepare_grouping_table
from comparison_estimates import baseline_log2_costs, largest_k_below_one


def make_row(n,k,nu,surplus):
    kmp_log2,es_log2=baseline_log2_costs(n,k,1,nu)
    estimate=counted_grouping_estimate(n,(n-k)*nu,nu,surplus)
    return dict(N=n,K=k,nu=nu,ell=1,
                components=' '.join(f'{g}:{s}' for g,s in estimate.components),
                available_equations=estimate.available_equations,
                selected_columns=estimate.selected_columns,
                permutation_rank=estimate.permutation_rank,
                residual_dimension=estimate.residual_dimension,
                spare_equations=estimate.spare_equations,
                log2_compatible_count=estimate.log2_compatible_count,
                psd_log2_counted=estimate.log2_trials,
                kmp_log2=kmp_log2,
                esser_santini_log2=es_log2,
                )


def write_outputs(rows,x,stem,out_dir):
    with (out_dir/f'{stem}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    with (out_dir/f'{stem}.dat').open('w') as f:
        f.write(f'{x} kmp es psd\n')
        for r in rows:
            f.write(f"{r[x]} {r['kmp_log2']:.6f} {r['esser_santini_log2']:.6f} "
                    f"{r['psd_log2_counted']:.6f}\n")
    fig,ax=plt.subplots(figsize=(8,5))
    draw(ax,rows,x)
    ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(out_dir/f'{stem}.png',dpi=200)
    plt.close(fig)


def draw(ax,rows,x):
    for key,label,color,style in [
        ('kmp_log2','KMP','#6F9CC5','-'),
        ('esser_santini_log2','Esser–Santini','#EBA469','--'),
        ('psd_log2_counted','Section 5 (counted construction)','#69B199','-')]:
        ax.plot([r[x] for r in rows],[r[key] for r in rows],style,color=color,label=label)
    ax.set_xlabel(r'$\nu$' if x=='nu' else r'$\bar n$')
    ax.set_ylabel(r'$\log_2 T$ (polynomial factors suppressed)')
    ax.grid(axis='y',alpha=.25)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nu-min',type=int,default=11)
    p.add_argument('--nu-max',type=int,default=60)
    p.add_argument('--n-min',type=int,default=20)
    p.add_argument('--n-max',type=int,default=160)
    p.add_argument('--surplus',type=int,default=3)
    p.add_argument('--out-dir',type=Path,default=Path('data'))
    a=p.parse_args()
    if not 11<=a.nu_min<=a.nu_max<=146:
        p.error('require 11 <= nu-min <= nu-max <= 146 for this comparison')
    if not 20<=a.n_min<=a.n_max or a.surplus<0:
        p.error('require 20 <= n-min <= n-max, surplus >= 0')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    prepare_grouping_table(64,27*a.nu_max-a.surplus)
    left=[make_row(64,37,nu,a.surplus)
          for nu in range(a.nu_min,a.nu_max+1)]
    right=[make_row(n,largest_k_below_one(n,11,1),11,a.surplus)
           for n in range(a.n_min,a.n_max+1)]
    assert all(x['psd_log2_counted']>=y['psd_log2_counted']-1e-10 for x,y in zip(left,left[1:]))
    write_outputs(left,'nu',f'counted_grouping_N64_K37_L1_nu{a.nu_min}to{a.nu_max}',a.out_dir)
    write_outputs(right,'N','counted_grouping_q11_vs_n',a.out_dir)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    draw(axes[0],left,'nu');draw(axes[1],right,'N')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(a.out_dir/'counted_grouping_comparison.png',dpi=200)
    plt.close(fig)
    print('PSD below KMP at nu:',[r['nu'] for r in left if r['psd_log2_counted']<r['kmp_log2']])
    print('PSD below KMP at N:',[r['N'] for r in right if r['psd_log2_counted']<r['kmp_log2']])
    print('nu=33:',next((r for r in left if r['nu']==33),None))
    print('Wrote tables and previews to',a.out_dir)


if __name__=='__main__':
    main()
