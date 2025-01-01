"""
Identifiers
    This module has two sets of identifiers. One set is active, and the other set is in uniformly formatted comments
    at the end of every line that includes an identifier that has an alternative identifier. 

    First, that might be distracting. In Visual Studio Code, the following extension will hide all comments but not docstrings:
    https://marketplace.visualstudio.com/items?itemName=eliostruyf.vscode-hide-comments

    Second, you can swap the sets of identifiers or delete one set of identifiers permanently.

    Step 1: regex find:
"""
# ^(?! *#)( *?)(\S.+?)( # )(.+) # This line is a comment and not a docstring because the Python interpreter handles `\S` better in a comment
"""
    Step 2: choose a regex replace option:
        A) To SWAP the sets of identifiers
        $1$4$3$2
        B) To PERMANENTLY replace the active set of identifiers
        $1$4
        C) To PERMANENTLY delete the inactive set of identifiers, which are in the comments
        $1$2

"""
from typing import List
import numba
import numpy

from mapFolding import validateListDimensions, getLeavesTotal
from mapFolding.benchmarks import recordBenchmarks

dtypeDefault = numpy.uint8
dtypeMaximum = numpy.uint16

A = numba.literally(0) # leafAbove: int = 0
B = numba.literally(1) # leafBelow: int = 1
count = numba.literally(2) # countDimensionsGapped: int = 2
gapter = numba.literally(3) # gapRangeStart: int = 3

@recordBenchmarks()
@numba.njit(cache=True, fastmath=False)
def foldings(p: List[int]): # def foldings(listDimensions: List[int]) -> int:
    """
    Calculate the number of distinct possible ways to fold a map with given dimensions.
    This function computes the number of different ways a map can be folded along its grid lines,
    considering maps with at least two positive dimensions.

    Parameters:
        p : A list of integers representing the dimensions of the map. Must contain at least two positive dimensions. # listDimensions : A list of integers representing the dimensions of the map. Must contain at least two positive dimensions.

    Returns
        foldingsTotal: The total number of possible distinct foldings for the given map dimensions.
    """
    def integerSmall(value):
        return numpy.uint8(value)
        # return numba.uint8(value)

    def integerLarge(value):
        return numpy.uint64(value)
        # return numba.uint64(value)

    listDimensionsPositive = validateListDimensions(p) # listDimensionsPositive = validateListDimensions(listDimensions)

    # idk wtf `numba.literal_unroll()` is _supposed_ to do, but it turned `n` into a float which then turned `foldingsTotal` into a float
    # n = numba.literal_unroll(getLeavesTotal(listDimensionsPositive)) # leavesTotal: int = getLeavesTotal(listDimensionsPositive)
    # d = numba.literal_unroll(len(p)) # dimensionsTotal: int = len(listDimensions)
    n = integerSmall(getLeavesTotal(listDimensionsPositive)) # leavesTotal: int = getLeavesTotal(listDimensionsPositive)
    d = integerSmall(len(p)) # dimensionsTotal: int = len(listDimensions)

    """How to build a leaf connection graph, also called a "Cartesian Product Decomposition" 
    or a "Dimensional Product Mapping", with sentinels: 
    Step 1: find the cumulative product of the map's dimensions"""
    P = numpy.ones(d + 1, dtype=dtypeDefault) # cumulativeProduct = numpy.ones(dimensionsTotal + 1, dtype=dtypeDefault)
    for i in range(1, d + 1): # for dimension1ndex in range(1, dimensionsTotal + 1):
        P[i] = P[i - 1] * p[i - 1] # cumulativeProduct[dimension1ndex] = cumulativeProduct[dimension1ndex - 1] * listDimensions[dimension1ndex - 1]

    """Step 2: for each dimension, create a coordinate system """
    """C[i][m] holds the i-th coordinate of leaf m""" # """coordinateSystem[dimension1ndex][leaf1ndex] holds the dimension1ndex-th coordinate of leaf leaf1ndex"""
    C = numpy.zeros((d + 1, n + 1), dtype=dtypeDefault) # coordinateSystem = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=dtypeDefault)
    for i in range(1, d + 1): # for dimension1ndex in range(1, dimensionsTotal + 1):
        for m in range(1, n + 1): # for leaf1ndex in range(1, leavesTotal + 1):
            C[i][m] = ((m - 1) // P[i - 1]) % p[i - 1] + 1 # coordinateSystem[dimension1ndex][leaf1ndex] = ((leaf1ndex - 1) // cumulativeProduct[dimension1ndex - 1]) % listDimensions[dimension1ndex - 1] + 1

    """Step 3: create a huge empty connection graph"""
    D = numpy.zeros((d + 1, n + 1, n + 1), dtype=dtypeDefault) # connectionGraph = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=dtypeDefault)

    """D[i][l][m] computes the leaf connected to m in dimension i when inserting l""" # """connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndex] computes the leaf1ndex connected to leaf1ndex in dimension1ndex when inserting activeLeaf1ndex"""
    """Step for... for... for...: fill the connection graph"""
    for i in range(1, d + 1): # for dimension1ndex in range(1, dimensionsTotal + 1):
        for l in range(1, n + 1): # for activeLeaf1ndex in range(1, leavesTotal + 1):
            for m in range(1, l + 1): # for leaf1ndexConnectee in range(1, activeLeaf1ndex + 1):
                if (C[i][l] & 1) == (C[i][m] & 1): # if distance % 2 == 0:
                    if C[i][m] == 1: # if coordinateSystem[dimension1ndex][leaf1ndexConnectee] == 1:
                        D[i][l][m] = m # connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee
                    else:
                        D[i][l][m] = m - P[i - 1] # connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee - cumulativeProduct[dimension1ndex - 1]
                else: 
                    if C[i][m] == p[i - 1] or m + P[i - 1] > l: # if coordinateSystem[dimension1ndex][leaf1ndexConnectee] == listDimensions[dimension1ndex - 1] or leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1] > activeLeaf1ndex:
                        D[i][l][m] = m # connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee
                    else:
                        D[i][l][m] = m + P[i - 1] # connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1]

    """For numba, a single array is faster than four separate arrays"""
    s = numpy.zeros((4, n + 1), dtype=dtypeDefault) # track = numpy.zeros((4, leavesTotal + 1), dtype=dtypeDefault)

    """Indices of array `s` ("s" is for "state"), which is a collection of one-dimensional arrays each of length `n + 1`.""" # """Indices of array `track` (to "track" the state), which is a collection of one-dimensional arrays each of length `leavesTotal + 1`."""
    """The values in the array cells are dynamic, small, unsigned integers."""
    gap = numpy.zeros(integerLarge(n) * integerLarge(n) + 1, dtype=dtypeMaximum) # potentialGaps = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximum)
    """
    c:/apps/mapFolding/mapFolding/lunnon.py:107: RuntimeWarning: overflow encountered in scalar multiply
    gap = numpy.zeros(n * n + 1, dtype=dtypeMaximum) # potentialGaps = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximum)
    """

    foldingsTotal = integerLarge(0)
    l = integerSmall(1) # activeLeaf1ndex: int = 1
    g = integerSmall(0) # activeGap1ndex: int = 0

    while l > 0: # while activeLeaf1ndex > 0:
        if l <= 1 or s[B][0] == 1: # if activeLeaf1ndex <= 1 or track[leafBelow][0] == 1:
            if l > n: # if activeLeaf1ndex > leavesTotal:
                
                # foldingsTotal += integerLarge(n) # foldingsTotal += leavesTotal
                foldingsTotal += n # foldingsTotal += leavesTotal
            else:
                dd = integerSmall(0) # dimensionsUnconstrained: int = 0
                """Track possible gaps for leaf l in each section""" # """Track possible gaps for activeLeaf1ndex in each section"""
                # gg = integerSmall(s[gapter][l - 1]) # gap1ndexLowerBound: int = track[gapRangeStart][activeLeaf1ndex - 1]
                gg = s[gapter][l - 1] # gap1ndexLowerBound: int = track[gapRangeStart][activeLeaf1ndex - 1]
                """Reset gap index"""
                g = gg # activeGap1ndex = gap1ndexLowerBound

                """Count possible gaps for leaf l in each section""" # """Count possible gaps for activeLeaf1ndex in each section"""
                i = integerSmall(1) # dimension1ndex: int = 1
                while i <= d: # for dimension1ndex in range(1, dimensionsTotal + 1):
                    if D[i][l][l] == l: # if connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex:
                        dd += 1 # dimensionsUnconstrained += 1
                    else:
                        # m = integerSmall(D[i][l][l]) # leaf1ndexConnectee: int = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
                        m = D[i][l][l] # leaf1ndexConnectee: int = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
                        while m != l: # while leaf1ndexConnectee != activeLeaf1ndex:
                            gap[gg] = m # potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                            if s[count][m] == 0: # if track[countDimensionsGapped][leaf1ndexConnectee] == 0:
                                gg += 1 # gap1ndexLowerBound += 1
                            s[count][m] += 1 # track[countDimensionsGapped][leaf1ndexConnectee] += 1
                            m = D[i][l][s[B][m]] # leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][track[leafBelow][leaf1ndexConnectee]]
                    i += 1

                """If leaf l is unconstrained in all sections, it can be inserted anywhere""" # """If activeLeaf1ndex is unconstrained in all sections, it can be inserted anywhere"""
                if dd == d: # if dimensionsUnconstrained == dimensionsTotal:
                    m = integerSmall(0) # leaf1ndex: int = 0
                    while m < l: # for leaf1ndex in range(activeLeaf1ndex):
                        gap[gg] = m # potentialGaps[gap1ndexLowerBound] = leaf1ndex
                        gg += 1 # gap1ndexLowerBound += 1
                        m += 1

                """Filter gaps that are common to all sections"""
                j = g
                while j < gg: # for indexMiniGap in range(activeGap1ndex, gap1ndexLowerBound):
                    gap[g] = gap[j] # potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                    if s[count][gap[j]] == d - dd: # if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                        g += 1 # activeGap1ndex += 1
                    """Reset s[count] for next iteration""" # """Reset track[countDimensionsGapped] for next iteration"""
                    s[count][gap[j]] = 0  # track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0
                    j += 1

        """Recursive backtracking steps"""
        while l > 0 and g == s[gapter][l - 1]: # while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart][activeLeaf1ndex - 1]:
            l -= 1 # activeLeaf1ndex -= 1
            s[B][s[A][l]] = s[B][l] # track[leafBelow][track[leafAbove][activeLeaf1ndex]] = track[leafBelow][activeLeaf1ndex]
            s[A][s[B][l]] = s[A][l] # track[leafAbove][track[leafBelow][activeLeaf1ndex]] = track[leafAbove][activeLeaf1ndex]

        """Place leaf in valid position"""
        if l > 0: # if activeLeaf1ndex > 0:
            g -= 1 # activeGap1ndex -= 1
            s[A][l] = gap[g] # track[leafAbove][activeLeaf1ndex] = potentialGaps[activeGap1ndex]
            s[B][l] = s[B][s[A][l]] # track[leafBelow][activeLeaf1ndex] = track[leafBelow][track[leafAbove][activeLeaf1ndex]]
            s[B][s[A][l]] = l # track[leafBelow][track[leafAbove][activeLeaf1ndex]] = activeLeaf1ndex
            s[A][s[B][l]] = l # track[leafAbove][track[leafBelow][activeLeaf1ndex]] = activeLeaf1ndex
            """Save current gap index"""
            s[gapter][l] = g # track[gapRangeStart][activeLeaf1ndex] = activeGap1ndex
            """Move to next leaf"""
            l += 1 # activeLeaf1ndex += 1
    return foldingsTotal
