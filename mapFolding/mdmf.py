import numpy as numpy
from numba import njit, int64, types

@njit
def foldings(p: types.List(int64)) -> int64: # type: ignore
    """
    Calculate number of ways to fold a map with given dimensions.
    Parameters:
        p: array of dimensions [n, m] for nxm map or [n,n,n...] for n-dimensional
    Returns:
        foldingsTotal: Total number of valid foldings
    """
    # Calculate total number of leaves
    leavesTotal = 1
    for dimension in p:
        leavesTotal *= dimension
    
    numberOfDimensions = len(p)  # number of dimensions
    
    # Calculate dimensional products
    big_p = numpy.ones(numberOfDimensions + 1, dtype=numpy.int64)
    for i in range(1, numberOfDimensions + 1):
        big_p[i] = big_p[i - 1] * p[i - 1]
        
    # Calculate coordinates in each dimension
    c = numpy.zeros((numberOfDimensions + 1, leavesTotal + 1), dtype=numpy.int64)
    for i in range(1, numberOfDimensions + 1):
        for m in range(1, leavesTotal + 1):
            c[i][m] = ((m - 1) // big_p[i - 1]) % p[i - 1] + 1

    # Calculate connections in each dimension
    leafConnectionGraph = numpy.zeros((numberOfDimensions + 1, leavesTotal + 1, leavesTotal + 1), dtype=numpy.int64)
    for i in range(1, numberOfDimensions + 1):
        for l in range(1, leavesTotal + 1):
            for m in range(1, l + 1):
                delta = c[i][l] - c[i][m]
                if delta % 2 == 0:
                    # If delta is even
                    leafConnectionGraph[i][l][m] = m if c[i][m] == 1 else m - big_p[i - 1]
                else:
                    # If delta is odd
                    leafConnectionGraph[i][l][m] = m if (c[i][m] == p[i - 1] or m + big_p[i - 1] > l) else m + big_p[i - 1]

    a = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)  # leaf above m
    b = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)  # leaf below m
    count = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)  # Counts sections with gaps for new leaf
    gapter = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)  # Indices/pointers for each stack/level per leaf
    gap = numpy.zeros((leavesTotal + 1) * (numberOfDimensions + 1), dtype=numpy.int64)  # All possible gaps for each leaf
    # Initialize variables for backtracking
    foldingsTotal: int = 0  # Total number of foldings
    g: int = 0            # Gap index
    l: int = 1            # Current leaf

    # Main folding loop using a stack-based approach
    while l > 0:
        if l <= 1 or b[0] == 1:
            if l > leavesTotal:
                foldingsTotal += leavesTotal
            else:
                dd: int = 0  # Number of sections where leaf l is unconstrained
                gg: int = gapter[l - 1]  # Track possible gaps
                g = gg

                # Find potential gaps for leaf l in each dimension
                for i in range(1, numberOfDimensions + 1):
                    if leafConnectionGraph[i][l][l] == l:
                        dd += 1
                    else:
                        m: int = leafConnectionGraph[i][l][l]
                        while m != l:
                            gap[gg] = m
                            if count[m] == 0:
                                gg += 1
                            count[m] += 1
                            m = leafConnectionGraph[i][l][b[m]]
                # If leaf l is unconstrained in all sections, it can be inserted anywhere
                if dd == numberOfDimensions:
                    for m in range(l):
                        gap[gg] = m
                        gg += 1

                for j in range(g, gg):
                    gap[g] = gap[j]
                    if count[gap[j]] == numberOfDimensions - dd:
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

