#!/usr/bin/env sage
"""Shared linear-algebra helpers for the polynomial-regime solvers."""

from sage.all import GF, matrix, vector


class EnumerationLimitError(RuntimeError):
    """Raised when an affine solution space is too large to enumerate."""


def affine_solution_space(value, syndrome):
    """Return one solution and a kernel basis, or ``(None, None)`` if empty."""
    # Sage's sparse GF(2) backend does not reliably solve rectangular systems;
    # the dense representation uses the optimized M4RI backend instead.
    value = value.dense_matrix()
    try:
        particular = value.solve_right(syndrome)
    except ValueError:
        return None, None
    return particular, list(value.right_kernel().basis())


def enumerate_affine_space(particular, basis, max_dimension):
    """Enumerate an affine binary space in Gray-code order."""
    dimension = len(basis)
    if dimension > max_dimension:
        raise EnumerationLimitError(
            f"affine dimension {dimension} exceeds limit {max_dimension}"
        )

    current = vector(GF(2), particular)
    yield vector(GF(2), current)
    previous_gray = 0
    for index in range(1, 1 << dimension):
        gray = index ^ (index >> 1)
        changed = gray ^ previous_gray
        basis_index = (changed & -changed).bit_length() - 1
        current += basis[basis_index]
        yield vector(GF(2), current)
        previous_gray = gray


def column_vector_to_matrix(value, n):
    """Invert column-wise vectorization of an n-by-n binary matrix."""
    field_two = GF(2)
    return matrix(
        field_two,
        n,
        n,
        lambda row, column: value[column * n + row],
    )


def is_permutation_matrix(value):
    """Test permutation structure over GF(2), using Hamming rather than parity."""
    if value.nrows() != value.ncols():
        return False
    n = value.nrows()
    return all(
        sum(entry != 0 for entry in value.row(row)) == 1 for row in range(n)
    ) and all(
        sum(entry != 0 for entry in value.column(column)) == 1
        for column in range(n)
    )


def fixed_row_and_column_constraints(n, row, column):
    """Fix a permutation's selected row/column to a guessed nonzero position."""
    if not 0 <= row < n or not 0 <= column < n:
        raise ValueError("row and column must lie between 0 and n-1")

    assignments = {}
    for current_column in range(n):
        assignments[current_column * n + row] = int(current_column == column)
    for current_row in range(n):
        assignments[column * n + current_row] = int(current_row == row)

    field_two = GF(2)
    ordered = sorted(assignments.items())
    constraints = matrix(field_two, len(ordered), n * n, sparse=True)
    for equation, (coordinate, _) in enumerate(ordered):
        constraints[equation, coordinate] = 1
    syndrome = vector(field_two, [rhs for _, rhs in ordered])
    return constraints, syndrome
