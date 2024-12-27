"""
Key concepts
    - A "leaf" is a unit square in the map
    - A "gap" is a potential position where a new leaf can be folded
    - Connections track how leaves can connect above/below each other
    - The algorithm builds foldings incrementally by placing one leaf at a time
    - Backtracking explores all valid combinations
    - Leaves and dimensions are enumerated starting from 1, not 0; hence, leaf1ndex not leafIndex

Algorithm flow
    For each leaf
        - Find valid gaps in each dimension
        - Place leaf in valid position
            - Try to find another lead to put in the adjacent position
            - Repeat until the map is completely folded
        - Backtrack when no valid positions remain

Identifiers
    This module has two sets of identifiers. One set is active, and the other set is in uniformly formatted comments
    at the end of every line that includes an identifier that has an alternative identifier. 

    First, that might be distracting. In Visual Studio Code, the following extension will hide all comments but not docstrings:
    https://marketplace.visualstudio.com/items?itemName=eliostruyf.vscode-hide-comments

    Second, you can swap the sets of identifiers or delete one set of identifiers permanently.

    Step 1: regex find:
"""
# ^(?!#)( *?)(\S.+?)( # )(.+)
"""
    Step 2: choose a regex replace option:
        A) To SWAP the sets of identifiers
        $1$4$3$2
        B) To PERMANENTLY replace the active set of identifiers
        $1$4
        C) To PERMANENTLY delete the inactive set of identifiers, which are in the comments
        $1$2

    Equivalent identifiers:
    A = leafAbove
    B = leafBelow
    C = coordinateSystem
    count = countDimensionsGapped
    D = connectionGraph
    d = dimensionsTotal
    dd = unconstrainedLeaf
    delta = distance
    g = activeGap1ndex
    gap = potentialGaps
    gapter = gapRangeStart
    gg = gap1ndexLowerBound
    i = dimension1ndex
    j = indexMiniGap
    l = activeLeaf1ndex
    m = leaf1ndex or leaf1ndexConnectee
    n = leavesTotal
    P = cumulativeProduct
    p = listDimensions
    s = track
"""
from typing import List
import numba
import numpy

# both of the following functions are used by the test modules
from mapFolding import parseListDimensions, getLeavesTotal

def foldings(listDimensions: List[int]) -> int: # def foldings(p: List[int]) -> int:
    """I can't figure out how to make numba happy with the calls to other functions,
    so these validators live here."""
    if not listDimensions: # if not p:
        raise ValueError("`listDimensions` is a required parameter.") # raise ValueError("`p` is a required parameter.")
    listDimensionsPositive = [dimension for dimension in parseListDimensions(listDimensions, 'listDimensions') if dimension > 0] # listDimensionsPositive = [dimension for dimension in parseListDimensions(p, 'p') if dimension > 0]
    if len(listDimensionsPositive) < 2:
        raise NotImplementedError(f"This function requires `listDimensions`, {listDimensions}, to have at least two dimensions greater than 0. You may want to look at https://oeis.org/ or other functions in this package.") # raise NotImplementedError(f"This function requires `p`, {p}, to have at least two dimensions greater than 0. You may want to look at https://oeis.org/ or other functions in this package.")

    leavesTotal: int = getLeavesTotal(listDimensionsPositive) # n: int = getLeavesTotal(listDimensionsPositive)
    return countFoldings(listDimensionsPositive, leavesTotal) # return countFoldings(listDimensionsPositive, n)

@numba.njit(cache=True, fastmath=False)
def countFoldings(listDimensions: List[int], leavesTotal: int) -> int: # def countFoldings(p: List[int], n: int) -> int:
    dimensionsTotal: int = len(listDimensions) # d: int = len(p)

    """How to build a numpy.ndarray connectionGraph with sentinel values: 
    ("Cartesian Product Decomposition" or "Dimensional Product Mapping")
    Step 1: find the cumulative product of the map dimensions"""
    cumulativeProduct = numpy.ones(dimensionsTotal + 1, dtype=numpy.int64) # P = numpy.ones(d + 1, dtype=numpy.int64)
    for dimension1ndex in range(1, dimensionsTotal + 1): # for i in range(1, d + 1):
        cumulativeProduct[dimension1ndex] = cumulativeProduct[dimension1ndex - 1] * listDimensions[dimension1ndex - 1] # P[i] = P[i - 1] * p[i - 1]

    """Step 2: for each dimension, create a coordinate system
    C[i][m] holds the i-th coordinate of leaf m"""
    coordinateSystem = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=numpy.int64) # C = numpy.zeros((d + 1, n + 1), dtype=numpy.int64)
    for dimension1ndex in range(1, dimensionsTotal + 1): # for i in range(1, d + 1):
        for leaf1ndex in range(1, leavesTotal + 1): # for m in range(1, n + 1):
            coordinateSystem[dimension1ndex][leaf1ndex] = ((leaf1ndex - 1) // cumulativeProduct[dimension1ndex - 1]) % listDimensions[dimension1ndex - 1] + 1 # C[i][m] = ((m - 1) // P[i - 1]) % p[i - 1] + 1

    """Step 3: create a huge empty connectionGraph"""
    connectionGraph = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=numpy.int64) # D = numpy.zeros((d + 1, n + 1, n + 1), dtype=numpy.int64)

    """Step for... for... for...: fill the connectionGraph"""
    for dimension1ndex in range(1, dimensionsTotal + 1): # for i in range(1, d + 1):
        """D[i][l][m] computes the leaf connected to m in dimension i when inserting l"""
        for activeLeaf1ndex in range(1, leavesTotal + 1): # for l in range(1, n + 1):
            for leaf1ndexConnectee in range(1, activeLeaf1ndex + 1): # for m in range(1, l + 1):
                distance = coordinateSystem[dimension1ndex][activeLeaf1ndex] - coordinateSystem[dimension1ndex][leaf1ndexConnectee] # delta = C[i][l] - C[i][m]
                """If delta is even"""
                if distance % 2 == 0: # if delta % 2 == 0:
                    if coordinateSystem[dimension1ndex][leaf1ndexConnectee] == 1: # if C[i][m] == 1:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee # D[i][l][m] = m
                    else:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee - cumulativeProduct[dimension1ndex - 1] # D[i][l][m] = m - P[i - 1]
                else: 
                    """If delta is odd"""
                    if coordinateSystem[dimension1ndex][leaf1ndexConnectee] == listDimensions[dimension1ndex - 1] or leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1] > activeLeaf1ndex: # if C[i][m] == p[i - 1] or m + P[i - 1] > l:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee # D[i][l][m] = m
                    else:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1] # D[i][l][m] = m + P[i - 1]

    """Indices of array `track`, which is a collection of one-dimensional arrays each of length `leavesTotal + 1`.
    The values in the array cells are dynamic, small, unsigned integers."""
    leafAbove = 0 # A = 0
    leafBelow = 1 # B = 1
    countDimensionsGapped = 2 # count = 2
    gapRangeStart = 3 # gapter = 3

    """For numba, a single array is faster than four separate arrays"""
    track = numpy.zeros((4, leavesTotal + 1), dtype=numpy.int64) # s = numpy.zeros((4, n + 1), dtype=numpy.int64)
    potentialGaps = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=numpy.int64) # gap = numpy.zeros(n * n + 1, dtype=numpy.int64)

    foldingsTotal: int = 0
    activeLeaf1ndex: int = 1 # l: int = 1
    activeGap1ndex: int = 0 # g: int = 0

    while activeLeaf1ndex > 0: # while l > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow][0] == 1: # if l <= 1 or s[B][0] == 1:
            if activeLeaf1ndex > leavesTotal: # if l > n:
                foldingsTotal += leavesTotal # foldingsTotal += n
            else:
                unconstrainedLeaf: int = 0 # dd: int = 0
                """Track possible gaps for leaf l in each section"""
                gap1ndexLowerBound: int = track[gapRangeStart][activeLeaf1ndex - 1] # gg: int = s[gapter][l - 1]
                """Reset gap index"""
                activeGap1ndex = gap1ndexLowerBound # g = gg

                """Count possible gaps for leaf l in each section"""
                for dimension1ndex in range(1, dimensionsTotal + 1): # for i in range(1, d + 1):
                    if connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex: # if D[i][l][l] == l:
                        unconstrainedLeaf += 1 # dd += 1
                    else:
                        leaf1ndexConnectee: int = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] # m: int = D[i][l][l]
                        while leaf1ndexConnectee != activeLeaf1ndex: # while m != l:
                            potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee # gap[gg] = m
                            if track[countDimensionsGapped][leaf1ndexConnectee] == 0: # if s[count][m] == 0:
                                gap1ndexLowerBound += 1 # gg += 1
                            track[countDimensionsGapped][leaf1ndexConnectee] += 1 # s[count][m] += 1
                            leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][track[leafBelow][leaf1ndexConnectee]] # m = D[i][l][s[B][m]]

                """If leaf l is unconstrained in all sections, it can be inserted anywhere"""
                if unconstrainedLeaf == dimensionsTotal: # if dd == d:
                    for leaf1ndex in range(activeLeaf1ndex): # for m in range(l):
                        potentialGaps[gap1ndexLowerBound] = leaf1ndex # gap[gg] = m
                        gap1ndexLowerBound += 1 # gg += 1

                """Filter gaps that are common to all sections"""
                for indexMiniGap in range(activeGap1ndex, gap1ndexLowerBound): # for j in range(g, gg):
                    potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap] # gap[g] = gap[j]
                    if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == dimensionsTotal - unconstrainedLeaf: # if s[count][gap[j]] == d - dd:
                        activeGap1ndex += 1 # g += 1
                    """Reset track[count] for next iteration"""
                    track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0 # s[count][gap[j]] = 0 

        """Recursive backtracking steps"""
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart][activeLeaf1ndex - 1]: # while l > 0 and g == s[gapter][l - 1]:
            activeLeaf1ndex -= 1 # l -= 1
            track[leafBelow][track[leafAbove][activeLeaf1ndex]] = track[leafBelow][activeLeaf1ndex] # s[B][s[A][l]] = s[B][l]
            track[leafAbove][track[leafBelow][activeLeaf1ndex]] = track[leafAbove][activeLeaf1ndex] # s[A][s[B][l]] = s[A][l]

        """Place leaf in valid position"""
        if activeLeaf1ndex > 0: # if l > 0:
            activeGap1ndex -= 1 # g -= 1
            track[leafAbove][activeLeaf1ndex] = potentialGaps[activeGap1ndex] # s[A][l] = gap[g]
            track[leafBelow][activeLeaf1ndex] = track[leafBelow][track[leafAbove][activeLeaf1ndex]] # s[B][l] = s[B][s[A][l]]
            track[leafBelow][track[leafAbove][activeLeaf1ndex]] = activeLeaf1ndex # s[B][s[A][l]] = l
            track[leafAbove][track[leafBelow][activeLeaf1ndex]] = activeLeaf1ndex # s[A][s[B][l]] = l
            """Save current gap index"""
            track[gapRangeStart][activeLeaf1ndex] = activeGap1ndex # s[gapter][l] = g
            """Move to next leaf"""
            activeLeaf1ndex += 1 # l += 1
    return foldingsTotal
