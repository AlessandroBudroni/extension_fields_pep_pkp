import sys
import argparse

# SageMath imports
from sage.all import (
    FiniteField,
    zero_matrix,
    matrix,
    randint,
)

def parse_range(s):
    if s=="": return None
    return [int(x) for x in s.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank-deficiency experiments for self-orthogonal codes over field extensions."
    )
    parser.add_argument("--n", type=str, default="4,8,12",
                         help="Values of n (code dimension) separated by commas (default: 4,8,12)")
    parser.add_argument("--p", type=str, default="",
                         help="Values of p (field charcteristic) separated by commas")
    parser.add_argument("--nu", type=str, default="",
                         help="Values of nu (field extension degree) separated by commas")
    parser.add_argument("--nexp", type=int, default=100,
                         help="Number of trials per experiment (default: 100)")
    parser.add_argument("--j", type=int, default=1,
                         help="Concurrent threads (default: 1)")
    parser.add_argument("--restore", action="store_true", dest="restore",
                         help="Restores results from file if present.")
    return parser.parse_args()


def sample_full_rank_matrix(rows, cols, q):
    F = FiniteField(q)
    l = rows * cols

    stop = False
    while not stop:
        m = [F.random_element() for _ in range(0, l, 1)]
        M = matrix(F, rows, cols, [mij for mij in m])
        stop = (M.rank() == rows)
    
    assert(M.rank() == rows)
    return M


def sample_permutation_matrix(n, q):
    P = zero_matrix(FiniteField(q), n, n)
    a = [i for i in range(0, n)]
    for i in range(n - 1, 0, -1):
        j = randint(0, i)
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp
    for i in range(0, n):
        P[i, a[i]] = 1

    return P

def random_twopce_instance_so(k, n, q):

    assert(ZZ(q).is_prime_power())
    if ZZ(q).is_prime():
        sample_selforthogonal_code = random_selforthogonal_code
    else:
        sample_selforthogonal_code = random_selforthogonal_code_fe

    # Secret matrices
    S = sample_full_rank_matrix(k, k, q)
    P = sample_permutation_matrix(n, q)
    
    # Public matrices
    G0 = sample_selforthogonal_code(n, k, q)
    while G0 is None:
        G0 = sample_selforthogonal_code(n, k, q)
    G1 = S * G0 * P

    # We avoid scenarios where dual(G0) intersects G2 non-trivialialy
    G2 = sample_selforthogonal_code(n, k, q)
    while G2 is None or tau(G0, G2) is None:
        if not ZZ(q).is_prime() and not G2 is None:
            # Remarkably, over binary field extensions dual(G0) intersects G2 non-trivialialy
            break
        G2 = sample_selforthogonal_code(n, k, q)
    G3 = S * G2 * P
    
    return (G0, G1), (G2, G3)


def tau(G, G_):
        # Moore-Penrose operator
        center = G * G_.transpose()
        if center.is_invertible():
            return G_.transpose() * center.inverse() * G
        else:
            return None


def pi(G, p):
    k, n = G.dimensions()
    return matrix(k, n, [g_ij**p for g_ij in G.list()])


def permutation_matrix_to_dict(M):
    return {i: M.row(i).nonzero_positions()[0] for i in range(M.nrows())}


def is_permutation_like_matrix(M):
    if M.nrows() != M.ncols():
        return False
    
    # Check rows
    if any(r.nonzero_positions().__len__() != 1 for r in M.rows()):
        return False
    
    # Check columns
    if any(c.nonzero_positions().__len__() != 1 for c in M.columns()):
        return False
    
    return True

def permutation_equations(n, q):
	idn = identity_matrix(FiniteField(q), n)
	one = matrix(FiniteField(q), 1, n, [1] * n)
	row_system = idn.tensor_product(one, subdivide=False)
	column_system = one.tensor_product(idn, subdivide=False)
	return row_system.augment(-one.transpose()), column_system.augment(-one.transpose())


def get_reduced_system(n, k, q , system_, column, row=0):
	to_delete = [(i * n + j) for i in range(0, n) for j in range(0, n) if (i == row or j == column)]
	system_ij = system_[:,:-1].delete_columns(to_delete).augment(system_[:,-1] + system_[:,n*row + column])
	return system_ij


def inspect_system(n, k, q, system, row, column):
    mtrx = get_reduced_system(n, k, q , system, column, row=row)
    for solution in mtrx.right_kernel().matrix():
        P = matrix(n - 1, n - 1, solution[:-1])
        if is_permutation_like_matrix(P):
            return P
    return None


def expand_permutation(M, row, col):
    """
    Expand an (n-1)x(n-1) matrix M to an n x n matrix
    with a 1 at position (j, i) and zeros in the rest
    of row j and column i.
    
    M: (n-1)x(n-1) matrix over some field
    i, j: target column and row index (0-based) in n x n matrix
    """
    n_minus_1 = M.nrows()
    n = n_minus_1 + 1
    F = M.base_ring()
    
    # Initialize n x n zero matrix
    N = matrix(F, n, n, 0)
    
    # Place 1 at (j, i)
    N[row, col] = 1
    
    # Fill in M in the remaining positions
    row_idx = 0
    for r in range(n):
        if r == row:
            continue  # skip the special row
        col_idx = 0
        for c in range(n):
            if c == col:
                continue  # skip the special column
            N[r, c] = M[row_idx, col_idx]
            col_idx += 1
        row_idx += 1
    
    return N


def print_matrix(input):
    #f'{exponent:03}'
    F = input.base_ring()
    z = F.gens()[0]
    length = len(str(F.cardinality()))
    m, n = input.dimensions()
    mat = matrix(ZZ, m, n, [0 if not entry else discrete_log(entry, z) for entry in input.list()])
    for i in range(0, m, 1):
        row = '[ '
        for j in range(0, n, 1):
            if not mat[i,j]:
                row += f'{str(mat[i,j]).ljust(length + 3)} '
            elif mat[i,j] == 1:
                row += f'{str(mat[i,j]).ljust(length + 3)} '
            else:
                row += f'{z}^{str(mat[i,j]).ljust(length)} '
        row += ']'
        print(f'{row}')
