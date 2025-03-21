"""
Comparison of two nearly identical counting implementations with vastly different performance.

This file provides a direct comparison between two variants of the map folding algorithm
that differ only in their approach to incrementing the folding counter. Despite their apparent
similarity, one implementation demonstrates orders of magnitude better performance than the other.

Key characteristics:
- Both implementations use Numba for performance optimization
- Both use identical data structures and array initializations
- `foldings_plus_1`: Increments the counter by 1 for each valid folding
- `foldings`: Increments the counter by n (total leaves) when certain conditions are met

The performance difference illustrates how subtle algorithmic changes can dramatically
impact computational efficiency, even when the overall algorithm structure remains unchanged.
This example serves as a compelling demonstration of the importance of algorithm analysis
and optimization for combinatorial problems.

Note: These functions are isolated for educational purposes to highlight the specific
optimization technique. The main package uses more comprehensive optimizations derived
from this and other lessons.
"""

from numba import njit
import numpy

@njit(cache=True)
def foldings_plus_1(p: list[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    n: int = 1  # Total number of leaves
    for dimension in p:
        n *= dimension

    d = len(p)  # Number of dimensions
    # Compute arrays P, C, D as per the algorithm
    P = numpy.ones(d + 1, dtype=numpy.int64)
    for i in range(1, d + 1):
        P[i] = P[i - 1] * p[i - 1]

    # C[i][m] holds the i-th coordinate of leaf m
    C = numpy.zeros((d + 1, n + 1), dtype=numpy.int64)
    for i in range(1, d + 1):
        for m in range(1, n + 1):
            C[i][m] = ((m - 1) // P[i - 1]) - ((m - 1) // P[i]) * p[i - 1] + 1

    # D[i][l][m] computes the leaf connected to m in section i when inserting l
    D = numpy.zeros((d + 1, n + 1, n + 1), dtype=numpy.int64)
    for i in range(1, d + 1):
        for l in range(1, n + 1):
            for m in range(1, l + 1):
                delta = C[i][l] - C[i][m]
                if delta % 2 == 0:
                    # If delta is even
                    if C[i][m] == 1:
                        D[i][l][m] = m
                    else:
                        D[i][l][m] = m - P[i - 1]
                else:
                    # If delta is odd
                    if C[i][m] == p[i - 1] or m + P[i - 1] > l:
                        D[i][l][m] = m
                    else:
                        D[i][l][m] = m + P[i - 1]
    # Initialize arrays/lists
    A = numpy.zeros(n + 1, dtype=numpy.int64)	   # Leaf above leaf m
    B = numpy.zeros(n + 1, dtype=numpy.int64)	   # Leaf below leaf m
    count = numpy.zeros(n + 1, dtype=numpy.int64)   # Counts for potential gaps
    gapter = numpy.zeros(n + 1, dtype=numpy.int64)  # Indices for gap stack per leaf
    gap = numpy.zeros(n * n + 1, dtype=numpy.int64) # Stack of potential gaps


    # Initialize variables for backtracking
    total_count = 0  # Total number of foldings
    g = 0            # Gap index
    l = 1            # Current leaf

    # Start backtracking loop
    while l > 0:
        # If we have processed all leaves, increment total count
        if l > n:
            total_count += 1
        else:
            dd = 0    # Number of sections where leaf l is unconstrained
            gg = g    # Temporary gap index
            g = gapter[l - 1]  # Reset gap index for current leaf

            # Count possible gaps for leaf l in each section
            for i in range(1, d + 1):
                if D[i][l][l] == l:
                    dd += 1
                else:
                    m = D[i][l][l]
                    while m != l:
                        if computationDivisions == 0 or l != computationDivisions or m % computationDivisions == computationIndex:
                            gap[gg] = m
                            if count[m] == 0:
                                gg += 1
                            count[m] += 1
                        m = D[i][l][B[m]]

            # If leaf l is unconstrained in all sections, it can be inserted anywhere
            if dd == d:
                for m in range(l):
                    gap[gg] = m
                    gg += 1

            # Filter gaps that are common to all sections
            for j in range(g, gg):
                gap[g] = gap[j]
                if count[gap[j]] == d - dd:
                    g += 1
                count[gap[j]] = 0  # Reset count for next iteration

        # Recursive backtracking steps
        while l > 0 and g == gapter[l - 1]:
            l -= 1
            B[A[l]] = B[l]
            A[B[l]] = A[l]

        if l > 0:
            g -= 1
            A[l] = gap[g]
            B[l] = B[A[l]]
            B[A[l]] = l
            A[B[l]] = l
            gapter[l] = g  # Save current gap index
            l += 1		 # Move to next leaf

    return total_count

@njit(cache=True)
def foldings(p: list[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    n: int = 1  # Total number of leaves
    for dimension in p:
        n *= dimension

    d = len(p)  # Number of dimensions
    # Compute arrays P, C, D as per the algorithm
    P = numpy.ones(d + 1, dtype=numpy.int64)
    for i in range(1, d + 1):
        P[i] = P[i - 1] * p[i - 1]

    # C[i][m] holds the i-th coordinate of leaf m
    C = numpy.zeros((d + 1, n + 1), dtype=numpy.int64)
    for i in range(1, d + 1):
        for m in range(1, n + 1):
            C[i][m] = ((m - 1) // P[i - 1]) - ((m - 1) // P[i]) * p[i - 1] + 1
            # C[i][m] = ((m - 1) // P[i - 1]) % p[i - 1] + 1 # NOTE different, but either one works

    # D[i][l][m] computes the leaf connected to m in section i when inserting l
    D = numpy.zeros((d + 1, n + 1, n + 1), dtype=numpy.int64)
    for i in range(1, d + 1):
        for l in range(1, n + 1):
            for m in range(1, l + 1):
                delta = C[i][l] - C[i][m]
                if delta % 2 == 0:
                    # If delta is even
                    if C[i][m] == 1:
                        D[i][l][m] = m
                    else:
                        D[i][l][m] = m - P[i - 1]
                else:
                    # If delta is odd
                    if C[i][m] == p[i - 1] or m + P[i - 1] > l:
                        D[i][l][m] = m
                    else:
                        D[i][l][m] = m + P[i - 1]
    # Initialize arrays/lists
    A = numpy.zeros(n + 1, dtype=numpy.int64)       # Leaf above leaf m
    B = numpy.zeros(n + 1, dtype=numpy.int64)       # Leaf below leaf m
    count = numpy.zeros(n + 1, dtype=numpy.int64)   # Counts for potential gaps
    gapter = numpy.zeros(n + 1, dtype=numpy.int64)  # Indices for gap stack per leaf
    gap = numpy.zeros(n * n + 1, dtype=numpy.int64) # Stack of potential gaps


    # Initialize variables for backtracking
    total_count = 0  # Total number of foldings
    g = 0            # Gap index
    l = 1            # Current leaf

    # Start backtracking loop
    while l > 0:
        if l <= 1 or B[0] == 1: # NOTE different
            # NOTE the above `if` statement encloses the the if/else block below
            # NOTE these changes increase the throughput by more than an order of magnitude
            if l > n:
                total_count += n
            else:
                dd = 0    # Number of sections where leaf l is unconstrained
                gg = gapter[l - 1]  # Track possible gaps # NOTE different, but not important
                g = gg # NOTE different, but not important

                # Count possible gaps for leaf l in each section
                for i in range(1, d + 1):
                    if D[i][l][l] == l:
                        dd += 1
                    else:
                        m = D[i][l][l]
                        while m != l:
                            if computationDivisions == 0 or l != computationDivisions or m % computationDivisions == computationIndex:
                                gap[gg] = m
                                if count[m] == 0:
                                    gg += 1
                                count[m] += 1
                            m = D[i][l][B[m]]

                # If leaf l is unconstrained in all sections, it can be inserted anywhere
                if dd == d:
                    for m in range(l):
                        gap[gg] = m
                        gg += 1

                # Filter gaps that are common to all sections
                for j in range(g, gg):
                    gap[g] = gap[j]
                    if count[gap[j]] == d - dd:
                        g += 1
                    count[gap[j]] = 0  # Reset count for next iteration

        # Recursive backtracking steps
        while l > 0 and g == gapter[l - 1]:
            l -= 1
            B[A[l]] = B[l]
            A[B[l]] = A[l]

        if l > 0:
            g -= 1
            A[l] = gap[g]
            B[l] = B[A[l]]
            B[A[l]] = l
            A[B[l]] = l
            gapter[l] = g  # Save current gap index
            l += 1         # Move to next leaf

    return total_count
