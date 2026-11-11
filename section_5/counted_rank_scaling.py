#!/usr/bin/env python3
"""Rank-statistics experiment for Section 5.3 (Figure 1).

Each parameter point is independent. Native binary elimination is optional;
all sampling, finite-field arithmetic, and support choices are the same with
--backend python. The uniform reference is calculated exactly for M by c.
"""
import argparse
import csv
import ctypes
import json
import math
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from counted_grouping import counted_grouping_estimate
from counted_grouping_rank_experiment import (
    binary_rank, expanded_columns, field_modulus, independent_groups,
    random_cells, sample_instance,
)


def uniform_reference(m,c):
    """Exact E[2^nullity] and full-column-rank probability for uniform M x c."""
    expectation=1+2.0**(c-m)-2.0**(-m)
    probability=(math.prod(1-2.0**(i-m) for i in range(c)) if c<=m else 0.0)
    return expectation,probability


def reduced_columns(columns,groups,cells,components,omitted,w,m):
    """Eliminate all permutation equations using an explicit cycle basis.

    A fully retained row and a position retained in every row form the base.
    Each other retained coordinate determines one independent four-coordinate
    cycle. These vectors form a basis of ker(K_T), of dimension c.
    """
    result=[];mask=(1<<m)-1
    for group,cell,(g,s),z in zip(groups,cells,components,omitted):
        if g==1:
            continue
        root=group[s]  # An unrestricted block exists since s<g.
        base=next(j for j in cell if j!=z)
        root_base=columns[root*w+base]
        for local,block in enumerate(group):
            if block==root:
                continue
            offset=columns[block*w+base]^root_base
            for position in cell:
                if position==base or (local<s and position==z):
                    continue
                value=columns[block*w+position]^columns[root*w+position]^offset
                assert value>>m==0  # Every cycle annihilates the permutation rows.
                result.append(value&mask)
    return result


class NativeRank:
    def __init__(self,path):
        self.library=ctypes.CDLL(str(path))
        self.function=self.library.gf2_rank
        self.function.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
        self.function.restype=ctypes.c_size_t

    def __call__(self,columns,bits):
        if not columns:
            return 0
        width=((bits+63)//64)*8
        data=b''.join(value.to_bytes(width,'little') for value in columns)
        buffer=ctypes.create_string_buffer(data)
        return int(self.function(buffer,len(columns),bits))


def build_native(directory):
    directory.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('gf2_rank.c')
    target=directory/('gf2_rank.dylib' if sys.platform=='darwin' else 'gf2_rank.so')
    subprocess.run(['cc','-O3','-shared','-fPIC',str(source),'-o',str(target)],check=True)
    return target.resolve()


def run_point(task):
    n,nu,args,library=task
    k=n//2;m=(n-k)*nu
    estimate=counted_grouping_estimate(n,m,nu,args.surplus)
    components=estimate.components;sizes=[g for g,s in components]
    c=estimate.residual_dimension
    rank=(NativeRank(library) if library else lambda cols,bits:binary_rank(cols))
    # Separate streams prevent the reference implementation or output order
    # from affecting generated instances and selected supports.
    rng=random.Random(f'{args.seed}:{n}:{nu}')
    modulus=field_modulus(nu)
    records=[];instances=[]
    for instance in range(args.instances):
        H,V,pi=sample_instance(n,k,nu,rng,modulus)
        try:
            groups=independent_groups(V,sizes,rng,args.group_attempts)
        except RuntimeError:
            instances.append(dict(instance=instance,grouping_found=False,
                                  zero_source_rows=sum(v==0 for v in V)))
            continue
        instances.append(dict(instance=instance,grouping_found=True,
                              zero_source_rows=sum(v==0 for v in V)))
        columns=expanded_columns(H,V,nu,modulus)
        for trial in range(args.partitions):
            cells=random_cells(sizes,rng)
            omitted=[rng.choice(cell) for cell in cells]
            reduced=reduced_columns(columns,groups,cells,components,omitted,n,m)
            assert len(reduced)==c
            d=c-rank(reduced,m)
            records.append(dict(instance=instance,partition=trial,nullity=d,two_to_nullity=2**d))
    accepted=sum(r['grouping_found'] for r in instances)
    uniform_mean,uniform_full=uniform_reference(m,c)
    row=dict(N=n,K=k,nu=nu,ell=1,attempted_instances=args.instances,
             accepted_instances=accepted,grouping_fraction=accepted/args.instances,
             samples=len(records),components=' '.join(f'{g}:{s}' for g,s in components),
             selected_columns=estimate.selected_columns,
             permutation_rank=estimate.permutation_rank,
             M=m,c=c,spare_equations=m-c,
             derived_E2d=(sum(r['two_to_nullity'] for r in records)/len(records) if records else math.nan),
             uniform_E2d=uniform_mean,
             derived_full_rank=(sum(r['nullity']==0 for r in records)/len(records) if records else math.nan),
             uniform_full_rank=uniform_full,
             max_nullity=max((r['nullity'] for r in records),default=-1))
    folder=args.out_dir/'runs'/f'N{n}_K{k}_nu{nu}'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'ranks.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['instance','partition','nullity','two_to_nullity'],lineterminator='\n')
        writer.writeheader();writer.writerows(records)
    (folder/'instances.json').write_text(json.dumps(instances,indent=2)+'\n')
    (folder/'summary.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


def write_outputs(args,rows):
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with (args.out_dir/f'counted_rank_scaling_nu{args.nu}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    with (args.out_dir/f'pkp_rank_statistics_nu{args.nu}.dat').open('w') as f:
        f.write('N derived_E2d uniform_E2d derived_full_rank uniform_full_rank\n')
        for r in rows:
            f.write(f"{r['N']} {r['derived_E2d']:.10g} {r['uniform_E2d']:.10g} "
                    f"{r['derived_full_rank']:.10g} {r['uniform_full_rank']:.10g}\n")
    make_plots(args,rows)


def make_plots(args,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field,label in [('E2d',r'$\mathbb{E}[2^{d(Q,z)}]$'),
                        ('full_rank',r'$\Pr[d(Q,z)=0]$')]:
        fig,ax=plt.subplots(figsize=(8,3.5))
        for prefix,title,style in [('derived','PKP-derived','-'),('uniform','Uniform reference','--')]:
            ax.plot([r['N'] for r in rows],[r[f'{prefix}_{field}'] for r in rows],style,
                    color='#6F9CC5',label=title)
        ax.set(xlabel=r'$\bar n$',ylabel=label,title=rf'$\nu={args.nu}$')
        ax.legend(frameon=False);ax.grid(axis='y',alpha=.25)
        fig.tight_layout();fig.savefig(args.out_dir/f'{field}_nu{args.nu}.png',dpi=200)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nu',type=int,required=True)
    p.add_argument('--replot',action='store_true',help='redraw previews from saved aggregate CSV without resampling')
    p.add_argument('--n-values',type=int,nargs='+',required=True)
    p.add_argument('--instances',type=int,default=20)
    p.add_argument('--partitions',type=int,default=100)
    p.add_argument('--surplus',type=int,default=3)
    p.add_argument('--group-attempts',type=int,default=1000)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--backend',choices=['native','python'],default='native')
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    if args.nu<2 or min(args.instances,args.partitions,args.jobs,args.group_attempts)<1 or args.surplus<0:
        p.error('require nu>=2, positive sample/job counts, and surplus>=0')
    if any(n<2 or n%2 for n in args.n_values):
        p.error('lengths must be positive even integers, for rate 1/2')
    args.out_dir=args.out_dir.resolve()
    if args.replot:
        with (args.out_dir/f'counted_rank_scaling_nu{args.nu}.csv').open() as f:
            rows=list(csv.DictReader(f))
        for row in rows:
            row['N']=int(row['N'])
            for key in ['derived_E2d','uniform_E2d','derived_full_rank','uniform_full_rank']:
                row[key]=float(row[key])
        make_plots(args,rows)
        return
    library=build_native(args.out_dir/'_native') if args.backend=='native' else None
    tasks=[(n,args.nu,args,library) for n in sorted(set(args.n_values))]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(run_point,t) for t in tasks]):
            row=future.result();rows.append(row)
            print(json.dumps(row),flush=True)
    rows.sort(key=lambda r:r['N']);write_outputs(args,rows)
    print('Accepted instances:',sum(r['accepted_instances'] for r in rows),'/',len(rows)*args.instances,flush=True)


if __name__=='__main__':
    main()
