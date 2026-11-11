"""KMP and regular-ISD baseline estimates for Section 5.3."""
import math

from kmp import kmp_complexity


def largest_k_below_one(n,nu,ell):
    """Largest K with the additional-input estimate (N!-1)/q^((N-K)ell)<1."""
    factorial_n=math.factorial(n)-1
    q=1<<nu
    for codimension in range(n+1):
        if factorial_n<q**(codimension*ell):
            return n-codimension
    raise ValueError(f'no K in [0,{n}] gives an estimated additional-solution count below one')


def baseline_log2_costs(n,k,ell,nu):
    """Return the KMP and Esser–Santini exponents used in Figure 2.

    Regular ISD includes n block-sum equations and selects integer retained
    sizes in each block. Polynomial linear-algebra factors are suppressed.
    """
    if not 0<=k<n or min(n,ell,nu)<1:
        raise ValueError('require N>0, 0<=K<N, ell>0 and nu>0')
    rows=(n-k)*ell*nu
    k_psd=n*n-rows
    k_eff=k_psd-n
    if k_eff<0:
        raise ValueError('regular-ISD effective dimension must be nonnegative')
    a,b=divmod(k_eff,n)
    r0=n-a;r1=r0-1
    if r0<=0 or (b and r1<=0):
        raise ValueError('regular-ISD retained sizes must be positive')
    log2_s=max(math.lgamma(n+1)/math.log(2)-rows,0.0)
    es=((n-b)*math.log2(n/r0)
        +(b*math.log2(n/r1) if b else 0.0)-log2_s)
    _,kmp=kmp_complexity(n,n-k,ell,1<<nu,debug=False)
    return kmp,es
