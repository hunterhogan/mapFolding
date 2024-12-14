from typing import List
from numba import njit
import numpy

from mapFolding import getLeavesTotal, parseListDimensions, OEISsequenceID

A = 0       # Leaf above leaf m
B = 1       # Leaf below leaf m
count = 2  # Counts for potential gaps
gapter = 3  # Indices for gap stack per leaf

@njit(cache=True)
def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    if listDimensions is None:
        raise ValueError(f"listDimensions is a required parameter.")

    listNonNegative = parseListDimensions(listDimensions, 'listDimensions')
    listPositive = [dimension for dimension in listNonNegative if dimension > 0]

    if len(listPositive) < 2:
        from typing import get_args
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. Other functions in this package implement the sequences {get_args(OEISsequenceID)}. You may want to look at https://oeis.org/.")
    
    listDimensions = listPositive
    n = getLeavesTotal(listDimensions)

    if computationDivisions > n:
        raise ValueError(f"computationDivisions, {computationDivisions}, must be less than or equal to the total number of leaves, {n}.")
    if computationDivisions > 1 and computationIndex >= computationDivisions:
        raise ValueError(f"computationIndex, {computationIndex}, must be less than computationDivisions, {computationDivisions}.")
    if computationDivisions < 0 or computationIndex < 0 or not isinstance(computationDivisions, int) or not isinstance(computationIndex, int):
        raise ValueError(f"computationDivisions, {computationDivisions}, and computationIndex, {computationIndex}, must be non-negative integers.")

    d = len(listDimensions)  # Number of dimensions
    P = numpy.ones(d + 1, dtype=numpy.int64)
    for i in range(1, d + 1):
        P[i] = P[i - 1] * listDimensions[i - 1]

    # C[i][m] holds the i-th coordinate of leaf m
    C = numpy.zeros((d + 1, n + 1), dtype=numpy.int64)
    for i in range(1, d + 1):
        for m in range(1, n + 1):
            C[i][m] = ((m - 1) // P[i - 1]) % listDimensions[i - 1] + 1 
            # C[i][m] = ((m - 1) // P[i - 1]) - ((m - 1) // P[i]) * p[i - 1] + 1 # while different than above line, either one works

    # D[i][l][m] computes the leaf connected to m in section i when inserting l
    D = numpy.zeros((d + 1, n + 1, n + 1), dtype=numpy.int64)
    for i in range(1, d + 1):
        for l in range(1, n + 1):
            for m in range(1, l + 1):
                delta = C[i][l] - C[i][m]
                if delta % 2 == 0: # If delta is even
                    if C[i][m] == 1:
                        D[i][l][m] = m
                    else:
                        D[i][l][m] = m - P[i - 1]
                else: # If delta is odd
                    if C[i][m] == listDimensions[i - 1] or m + P[i - 1] > l:
                        D[i][l][m] = m
                    else:
                        D[i][l][m] = m + P[i - 1]

    track = numpy.zeros((4, n + 1), dtype=numpy.int64)
    gap = numpy.zeros(n * n + 1, dtype=numpy.int64) # Stack of potential gaps

    total_count = 0  # Total number of foldings
    g = 0            # Gap index
    l = 1            # Current leaf

    # start Performance Measurement Area
    # Performance measurements should be segregated by the parameter values of `foldings`,
    # whether or not the "Performance Measurement Area" is part of the `foldings` function.
    while l > 0:
        if l <= 1 or track[B][0] == 1: 
            if l > n:
                total_count += n
            else:
                dd = 0     # Number of sections where leaf l is unconstrained
                gg = track[gapter][l - 1]  # Track possible gaps 
                g = gg      # Reset gap index

                for i in range(1, d + 1): # Count possible gaps for leaf l in each section
                    if D[i][l][l] == l:
                        dd += 1
                    else:
                        m = D[i][l][l]
                        while m != l:
                            if computationDivisions == 0 or l != computationDivisions or m % computationDivisions == computationIndex:
                                gap[gg] = m
                                if track[count][m] == 0:
                                    gg += 1
                                track[count][m] += 1
                            m = D[i][l][track[B][m]]

                if dd == d: # If leaf l is unconstrained in all sections, it can be inserted anywhere
                    for m in range(l):
                        gap[gg] = m
                        gg += 1

                for j in range(g, gg): # Filter gaps that are common to all sections
                    gap[g] = gap[j]
                    if track[count][gap[j]] == d - dd:
                        g += 1
                    track[count][gap[j]] = 0  # Reset track[count] for next iteration

        while l > 0 and g == track[gapter][l - 1]: # Recursive backtracking steps
            l -= 1
            track[B][track[A][l]] = track[B][l]
            track[A][track[B][l]] = track[A][l]

        if l > 0:
            g -= 1
            track[A][l] = gap[g]
            track[B][l] = track[B][track[A][l]]
            track[B][track[A][l]] = l
            track[A][track[B][l]] = l
            track[gapter][l] = g  # Save current gap index
            l += 1         # Move to next leaf
    # end Performance Measurement Area

    return total_count
