"""Fixed-parameter rank experiments for Section 5.3.

Instances use a random systematic parity check over GF(2^nu), with a planted
kernel vector and random secret permutation. The measurements test the enumeration-cost assumption.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

from counted_grouping import optimize_counted_groupings as optimize_star_components


def binary_rank(columns):
    pivots={}
    for value in columns:
        while value:
            pivot=value.bit_length()-1
            if pivot not in pivots:
                pivots[pivot]=value
                break
            value ^= pivots[pivot]
    return len(pivots)


def poly_remainder(value, modulus):
    degree=modulus.bit_length()
    while value and value.bit_length()>=degree:
        value ^= modulus << (value.bit_length()-degree)
    return value


def poly_gcd(a,b):
    while b:
        a,b=b,poly_remainder(a,b)
    return a


def square_mod(value, modulus):
    result=0
    while value:
        bit=value & -value
        result ^= 1 << (2*(bit.bit_length()-1))
        value ^= bit
    return poly_remainder(result,modulus)


def irreducible(modulus,degree):
    x=2
    # The gcd criterion for every i<=degree//2 excludes every proper factor.
    for i in range(1,degree+1):
        x=square_mod(x,modulus)
        if i<=degree//2 and poly_gcd(x^2,modulus)!=1:
            return False
    return x==2


def field_modulus(degree):
    if degree < 1:
        raise ValueError("field degree must be positive")
    if degree == 1:
        return 3
    rng=random.Random(10000+degree)
    while True:
        candidate=(1<<degree)|rng.getrandbits(degree)|1
        if irreducible(candidate,degree):
            return candidate


def gf_multiply(a,b,modulus,degree):
    result=0
    high=1<<degree
    while b:
        if b&1:
            result ^= a
        b >>= 1
        a <<= 1
        if a&high:
            a ^= modulus
    return result


def multiplication_tables(value,modulus,degree,width=8):
    basis=[]
    for _ in range(degree):
        basis.append(value)
        value <<= 1
        if value&(1<<degree):
            value ^= modulus
    tables=[]
    for start in range(0,degree,width):
        bits=basis[start:start+width]
        table=[0]*(1<<len(bits))
        for index in range(1,len(table)):
            bit=index&-index
            table[index]=table[index^bit]^bits[bit.bit_length()-1]
        tables.append(table)
    return tables


def table_multiply(value,tables):
    result=0
    for table in tables:
        result ^= table[value&255]
        value >>= 8
    return result


def sample_instance(w,k,nu,rng,modulus):
    c=w-k
    # H=[I|A]; W=(A b,b) lies in ker(H), in characteristic two.
    A=[[rng.getrandbits(nu) for _ in range(k)] for _ in range(c)]
    b=[rng.getrandbits(nu) for _ in range(k)]
    while not any(b):
        b=[rng.getrandbits(nu) for _ in range(k)]
    first=[]
    for row in A:
        value=0
        for h,x in zip(row,b):
            value ^= gf_multiply(h,x,modulus,nu)
        first.append(value)
    W=first+b
    pi=list(range(w));rng.shuffle(pi)
    V=[W[pi[i]] for i in range(w)]
    H=[[int(i==j) for j in range(c)]+row for i,row in enumerate(A)]
    # Check the full planted PKP relation independently.
    recovered=[0]*w
    for block,position in enumerate(pi):
        recovered[position]=V[block]
    for row in H:
        value=0
        for h,x in zip(row,recovered):
            value ^= gf_multiply(h,x,modulus,nu)
        assert value==0
    return H,V,pi


def expanded_columns(H,V,nu,modulus):
    w=len(V);m=len(H)*nu
    result=[]
    for block,value in enumerate(V):
        tables=multiplication_tables(value,modulus,nu)
        for position in range(w):
            column=0
            for row_index,row in enumerate(H):
                column |= table_multiply(row[position],tables) << (row_index*nu)
            column |= 1 << (m+block)
            column |= 1 << (m+w+position)
            result.append(column)
    return result


def independent_groups(V,sizes,rng,attempts=1000):
    indices=list(range(len(V)))
    for _ in range(attempts):
        rng.shuffle(indices)
        groups=[];start=0
        for size in sizes:
            groups.append(indices[start:start+size]);start+=size
        if all(binary_rank(V[i] for i in group)==len(group) for group in groups):
            return groups
    raise RuntimeError('could not find independent source groups')


def support(groups,cells,w):
    return [block*w+position for group,cell in zip(groups,cells)
            for block in group for position in cell]


def random_cells(sizes,rng):
    positions=list(range(sum(sizes)));rng.shuffle(positions)
    cells=[];start=0
    for size in sizes:
        cells.append(positions[start:start+size]);start+=size
    return cells


def run(args):
    rng=random.Random(args.seed)
    records=[];summaries=[]
    solutions_factorial=math.factorial(args.w)
    for nu in args.nus:
        m=(args.w-args.k)*nu
        variants={}
        if args.variant in ('star','stable'):
            _,components,_=optimize_star_components(args.w,m,max_group=nu)[-1]
            if args.variant != 'stable':
                variants['star']=components
            # A little surplus of equations also tests the stable-rank variant.
            _,slack_components,_=optimize_star_components(args.w,max(0,m-3),max_group=nu)[-1]
            variants['star_slack3']=slack_components
        modulus=field_modulus(nu)
        for variant,components in variants.items():
            sizes=tuple(g for g,s in components)
            t=sum(g*g-s for g,s in components);rankK=2*args.w-len(sizes)
            print(f'nu={nu} variant={variant} components={components} columns={t} binary_rows={m} rank(K_T)={rankK}',flush=True)
            for instance in range(args.instances):
                H,V,pi=sample_instance(args.w,args.k,nu,rng,modulus)
                try:
                    groups=independent_groups(V,sizes,rng)
                except RuntimeError:
                    if not args.skip_unavailable:
                        raise
                    print(f'nu={nu} variant={variant} instance={instance}: independent grouping unavailable; skipped',flush=True)
                    continue
                columns=expanded_columns(H,V,nu,modulus)
                for kind in ['random','compatible']:
                    count=args.partitions if kind=='random' else 1
                    for trial in range(count):
                        cells=(random_cells(sizes,rng) if kind=='random' else
                               [[pi[i] for i in group] for group in groups])
                        chosen=[]
                        for group,cell,(g,s) in zip(groups,cells,components):
                            short=set(group[:s])
                            allowed=([pi[i] for i in group[s:]] if kind=='compatible' else cell)
                            omitted=rng.choice(allowed)
                            chosen.extend(block*args.w+position for block in group for position in cell
                                          if block not in short or position!=omitted)
                        assert len(chosen)==t
                        if kind=='compatible':
                            assert all(block*args.w+pi[block] in chosen for block in range(args.w))
                        # Verify the exact support rank of the permutation rows.
                        assert binary_rank(columns[i]>>m for i in chosen)==rankK
                        d=t-binary_rank(columns[i] for i in chosen)
                        records.append(dict(nu=nu,variant=variant,instance=instance,
                                            kind=kind,trial=trial,corank=d,two_to_corank=2**d,
                                            columns=t,residual_dimension=t-rankK,
                                            binary_equations=m,groups=' '.join(map(str,sizes)),
                                            deletions=' '.join(str(s) for g,s in components)))
            for kind in ['random','compatible']:
                selected=[r for r in records if r['nu']==nu and r['variant']==variant and r['kind']==kind]
                if not selected:
                    continue
                summary=dict(N=args.w,K=args.k,nu=nu,variant=variant,kind=kind,samples=len(selected),
                             attempted_instances=args.instances,
                             accepted_instances=len({r['instance'] for r in selected}),
                             mean_two_to_corank=sum(r['two_to_corank'] for r in selected)/len(selected),
                             max_corank=max(r['corank'] for r in selected),
                             probability_full_column_rank=sum(r['corank']==0 for r in selected)/len(selected),
                             group_sizes=list(sizes),deletions=[s for g,s in components],residual_dimension=t-rankK,
                             binary_equations=m,
                             partition_log2=(math.log2(solutions_factorial)
                                             -math.log2(math.prod((g-s)*math.factorial(g-1) for g,s in components))))
                summaries.append(summary);print(json.dumps(summary),flush=True)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if records:
        with (args.out_dir/'rank_raw.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(records)
    else:
        (args.out_dir/'rank_raw.csv').write_text('')
        print('No accepted independent groupings; rank statistics are unavailable.',flush=True)
    (args.out_dir/'rank_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--w',type=int,default=64);parser.add_argument('--k',type=int,default=37)
    parser.add_argument('--nus',type=int,nargs='+',default=[33,34,35,43,50,60])
    parser.add_argument('--instances',type=int,default=5);parser.add_argument('--partitions',type=int,default=20)
    parser.add_argument('--variant',choices=['star','stable'],default='stable')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--skip-unavailable',action='store_true')
    parser.add_argument('--out-dir',type=Path,default=Path(__file__).parent/'results')
    run(parser.parse_args())
