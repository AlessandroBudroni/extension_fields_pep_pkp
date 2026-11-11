"""PSD support counts and integer optimization (Section 5.2).

All work estimates are for covering one fixed solution. Polynomial linear
algebra costs and any benefit from additional solutions are not included.
"""
from dataclasses import dataclass
import math

def optimize_counted_groupings(w,max_budget,max_group=None):
    """Return exact optima in the stated family for every available row budget.

    Component (g,s) has g rows/positions, s distinct rows omit the same
    position (0<=s<g). Its permanent is (g-s)*(g-1)!, its number of support
    coordinates is g^2-s, and the restricted permutation-equation rank is
    2g-1. Thus it needs (g-1)^2-s generic binary equations.
    """
    max_group=min(w,max_group or w)
    options=[]
    for g in range(1,max_group+1):
        fact=math.factorial(g-1)
        opts=[]
        for s in range(g):
            cost=(g-1)**2-s
            if 0<=cost<=max_budget:
                opts.append((cost,(g-s)*fact,s))
        options.append((g,opts))
    dp=[{} for _ in range(w+1)]
    dp[0][0]=(1,())
    for total in range(1,w+1):
        state=dp[total]
        for g,opts in options[:total]:
            for previous_cost,(product,components) in dp[total-g].items():
                for cost,permanent,s in opts:
                    used=previous_cost+cost
                    if used<=max_budget:
                        candidate=product*permanent
                        if used not in state or candidate>state[used][0]:
                            state[used]=(candidate,components+((g,s),))
    out=[];best=(0,(),0)
    for budget in range(max_budget+1):
        if budget in dp[w] and dp[w][budget][0]>best[0]:
            product,components=dp[w][budget]
            best=(product,tuple(sorted(components,reverse=True)),budget)
        out.append(best)
    return out


# Keep only the largest unrestricted table for each w. Plot sweeps precompute
# their maximum budget once; individual calls can also prepare a table.
_TABLES = {}
_CAPPED_TABLES = {}


def prepare_grouping_table(w, max_budget):
    if w <= 0 or max_budget < 0:
        raise ValueError("require w > 0 and a nonnegative residual budget")
    existing = _TABLES.get(w)
    if existing is None or existing[0] < max_budget:
        _TABLES[w] = (max_budget, optimize_counted_groupings(w, max_budget))
    return _TABLES[w][1]


@dataclass(frozen=True)
class GroupingEstimate:
    components: tuple
    log2_trials: float
    log2_compatible_count: float
    selected_columns: int
    permutation_rank: int
    residual_dimension: int
    available_equations: int
    spare_equations: int


def counted_grouping_estimate(w, binary_equations, source_dimension, surplus=3):
    """Optimize the proven support probability with a rank-aware row budget.

    The source-dimension cap supports the independent-V-row grouping heuristic;
    it does not itself prove that the restricted PKP matrix has generic rank.
    """
    if w < 1 or source_dimension < 1 or surplus < 0:
        raise ValueError("require positive w/source dimension and surplus >= 0")
    budget = binary_equations - surplus
    if budget < 0:
        raise ValueError("not enough binary equations for the requested surplus")
    existing = _TABLES.get(w)
    if existing is not None and existing[0] >= budget:
        product, components, used = existing[1][budget]
    else:
        components = ((source_dimension + 1, 0),)  # Request a capped table below.
    if max(g for g, s in components) > source_dimension:
        key = (w, source_dimension)
        capped = _CAPPED_TABLES.get(key)
        if capped is None or capped[0] < budget:
            capped = (budget, optimize_counted_groupings(
                w, budget, max_group=source_dimension))
            _CAPPED_TABLES[key] = capped
        product, components, used = capped[1][budget]
    columns = sum(g*g-s for g, s in components)
    permutation_rank = 2*w-len(components)
    assert columns-permutation_rank == used
    assert binary_equations-used >= surplus
    log2_product = math.log2(product)
    return GroupingEstimate(
        components=components,
        log2_trials=math.log2(math.factorial(w))-log2_product,
        log2_compatible_count=log2_product,
        selected_columns=columns,
        permutation_rank=permutation_rank,
        residual_dimension=used,
        available_equations=binary_equations,
        spare_equations=binary_equations-used,
    )
