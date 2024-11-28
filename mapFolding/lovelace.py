"""
The module enumerates all possible ways to fold a multi-dimensional map with given dimensions.

Implements algorithm from 
W. F. Lunnon, Multi-dimensional map-folding, "The Computer Journal", Volume 14, Issue 1, 1971, Pages 75-80, 
https://doi.org/10.1093/comjnl/14.1.75 (see also "./citations/Lunnon.bibtex") 
but directly based on Sean Irvine's Java port of Fred Lunnon's C version.

See https://github.com/archmageirvine/joeis/blob/80e3e844b11f149704acbab520bc3a3a25ac34ff/src/irvine/oeis/a001/A001415.java
"""

def foldings(p: list[int], computationDivisions: int = 0, computationIndex: int = 0, normalFoldings: bool = True) -> int:
    """
    Enumerate map foldings.
    
    Parameters:
        p: dimensions of the counting array, which may or may not be identical to 
            the dimensions of the map, see `mapFolding.getDimensions()`
        computationDivisions: attempt* to split computation into this many parts. (*See `mapFolding.countMinimumParsePoints()`)
        computationIndex: an integer in `range(0, computationDivisions)` to select the part to compute
        normalFoldings: when True, enumerate only normal foldings

    Returns:
        countTotal: total number of foldings

    The algorithm uses the following key data structures:
                computationDivisions=0 means compute everything in one go
                computationDivisions>1 means only compute part 'computationIndex' of 'computationDivisions' parts
                Example: computationIndex=2, computationDivisions=5 computes the third fifth
        - b[m]: leaf below leaf m in current folding
        - a[m]: leaf above leaf m in current folding
        - count[m]: counts sections with gaps for new leaf l below leaf m
        - gap[gapter[l-1] + j]: holds j-th possible/actual gap for leaf l
    """
    countTotal = 0

    # Calculate total number of leaves
    n = 1
    for pp in p:
        n *= pp

    # Initialize arrays
    a = [0] * (n + 1)  # leaf above
    b = [0] * (n + 1)  # leaf below
    count = [0] * (n + 1)
    gapter = [0] * (n + 1)
    gap = [0] * (n * n + 1)

        # Initialize dimension arrays
    dim = len(p)
    bigP = [1] * (dim + 1)
    c = [[0] * (n + 1) for _ in range(dim + 1)]
    d = [[[0] * (n + 1) for _ in range(n + 1)] for _ in range(dim + 1)]

        # Initialize bigP array with cumulative products
    for i in range(1, dim + 1):
        bigP[i] = bigP[i - 1] * p[i - 1]

        # Initialize c array - coordinate mapping
    for i in range(1, dim + 1):
        for m in range(1, n + 1):
            c[i][m] = (m - 1) // bigP[i - 1] - ((m - 1) // bigP[i]) * p[i - 1] + 1

        # Initialize d array - connection mapping
    for i in range(1, dim + 1):
        for l in range(1, n + 1):
            for m in range(1, l + 1):
                delta = c[i][l] - c[i][m]
                if (delta & 1) == 0:
                    d[i][l][m] = m if c[i][m] == 1 else m - bigP[i - 1]
                else:
                    d[i][l][m] = m if c[i][m] == p[i - 1] or m + bigP[i - 1] > l else m + bigP[i - 1]

    g = 0  # Gap counter
    l = 1  # Current leaf

    # Main backtrack loop - implements Lunnon's state machine:
    # 1. Try to extend current folding by adding new leaf
    # 2. If no extension possible, backtrack
    # 3. Process completed foldings
    while l > 0:
        if not normalFoldings or l <= 1 or b[0] == 1:
            if l > n:
                countTotal += n # Found valid foldings
            else:
                dd = 0
                gg = gapter[l - 1]
                g = gg

                    # For each dimension, track gaps
                for i in range(1, dim + 1):
                    if d[i][l][l] == l:
                        dd += 1
                    else:
                        m = d[i][l][l]
                        while m != l:
                                # The parse point
                            if computationDivisions == 0 or l != computationDivisions or m % computationDivisions == computationIndex:
                                gap[gg] = m
                                count[m] += 1  #Increment count here
                                gg += 1 #Increment gg only if a new gap is added.
                            m = d[i][l][b[m]]

                    # Handle unconstrained case
                if dd == dim:
                    for m in range(l):
                        gap[gg] = m
                        gg += 1

                    # Gap filtering and update
                k = g
                for j in range(g, gg):
                    if count[gap[j]] == dim - dd:
                        gap[k] = gap[j]
                        k += 1
                    count[gap[j]] = 0
                g = k # Update g

        # Backtrack when no more gaps
        while l > 0 and g == gapter[l - 1]:
            l -= 1
            if l > 0:
                b[a[l]] = b[l]
                a[b[l]] = a[l]

            # Make next move if possible
        if l > 0:
            g -= 1
            a[l] = gap[g]
            b[l] = b[a[l]]
            b[a[l]] = l
            a[b[l]] = l
            gapter[l] = g
            l += 1
    return countTotal
