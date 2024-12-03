from typing import List, Tuple

import numpy as numpy
from numba import njit


@njit
def foldings(p: List[int], flag: bool=True, res:int=0, mod:int=0) -> int:
    """
    Calculate number of ways to fold a map with given dimensions.
    Parameters:
        p: array of dimensions [n, m] for nxm map or [n,n,n...] for n-dimensional
        flag: when True, only count "normal" foldings
        res/mod: specify which subset of foldings to compute (for parallel processing)
    Returns:
        foldingsTotal: Total number of valid foldings
    """
    leavesTotal, number_of_dimensions, leaf_connection_matrix = setupFoldings(p)

    a = numpy.zeros(leavesTotal + 1, dtype=numpy.int32)  # leaf above m
    b = numpy.zeros(leavesTotal + 1, dtype=numpy.int32) # leaf below m
    count = numpy.zeros(leavesTotal + 1, dtype=numpy.int32)  # Counts sections with gaps for new leaf
    gapter = numpy.zeros(leavesTotal + 1, dtype=numpy.int32)  # Indices/pointers for each stack/level per leaf
    gap = numpy.zeros((leavesTotal + 1) * (number_of_dimensions + 1), dtype=numpy.int32)  # All possible gaps for each leaf
    # Initialize variables for backtracking
    foldingsTotal = 0  # Total number of foldings
    g = 0            # Gap index
    l = 1            # Current leaf

    # Main folding loop using a stack-based approach
    while l > 0:
        if (not flag) or l <= 1 or b[0] == 1:
            if l > leavesTotal:
                foldingsTotal += leavesTotal
            else:
                dd = 0  # Number of sections where leaf l is unconstrained
                gg = gapter[l - 1]  # track possible gaps
                g = gg

                # Find potential gaps for leaf l in each dimension
                for i in range(1, number_of_dimensions + 1):
                    if leaf_connection_matrix[i][l][l] == l:
                        dd += 1
                    else:
                        m = leaf_connection_matrix[i][l][l]
                        while m != l:
                            if mod == 0 or l != mod or m % mod == res:
                                gap[gg] = m
                                if count[m] == 0:
                                    gg += 1
                                count[m] += 1
                            m = leaf_connection_matrix[i][l][b[m]]
                # If leaf l is unconstrained in all sections, it can be inserted anywhere
                # Handle unconstrained case and filter common gaps
                if dd == number_of_dimensions:
                    for m in range(l):
                        gap[gg] = m
                        gg += 1

                for j in range(g, gg):
                    gap[g] = gap[j]
                    if count[gap[j]] == number_of_dimensions - dd:
                        g += 1
                    count[gap[j]] = 0

        # Backtrack if no more gaps
        while l > 0 and g == gapter[l - 1]:
            l -= 1
            b[a[l]] = b[l]
            a[b[l]] = a[l]

        # Insert leaf and advance
        if l > 0:
            g -= 1
            a[l] = gap[g]
            b[l] = b[a[l]]
            b[a[l]] = l
            a[b[l]] = l
            gapter[l] = g
            l += 1

    return foldingsTotal

@njit
def setupFoldings(p: List[int]) -> Tuple[int, int, numpy.ndarray[numpy.int32, numpy.dtype[numpy.int32]]]:
    # Calculate total number of leaves
    leavesTotal = 1
    for dimension in p:
        leavesTotal *= dimension
    
    # Initialize arrays needed for tracking the folding state
    
    number_of_dimensions = len(p)  # number of dimensions
    
    # Calculate dimensional products
    big_p = numpy.ones(number_of_dimensions + 1, dtype=numpy.int32)
    for i in range(1, number_of_dimensions + 1):
        big_p[i] = big_p[i - 1] * p[i - 1]
        
    # Calculate coordinates in each dimension
    # c[i][m] holds the i-th coordinate of leaf m
    c = numpy.zeros((number_of_dimensions + 1, leavesTotal + 1), dtype=numpy.int32)
    for i in range(1, number_of_dimensions + 1):
        for m in range(1, leavesTotal + 1):
            c[i][m] = ((m - 1) // big_p[i - 1]) % p[i - 1] + 1

    # Calculate connections in each dimension
    # d[i][l][m] computes the leaf connected to m in section i when inserting l
    leaf_connection_matrix = numpy.zeros((number_of_dimensions + 1, leavesTotal + 1, leavesTotal + 1), dtype=numpy.int32)
    for i in range(1, number_of_dimensions + 1):
        for l in range(1, leavesTotal + 1):
            for m in range(1, l + 1):
                delta = c[i][l] - c[i][m]
                if delta % 2 == 0:
                    # If delta is even
                    leaf_connection_matrix[i][l][m] = m if c[i][m] == 1 else m - big_p[i - 1]
                else:
                    # If delta is odd
                    leaf_connection_matrix[i][l][m] = m if (c[i][m] == p[i - 1] or m + big_p[i - 1] > l) else m + big_p[i - 1]
    return leavesTotal,number_of_dimensions,leaf_connection_matrix
