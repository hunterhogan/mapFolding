from typing import List
import numba
import numpy

from mapFolding.beDRY import validateListDimensions, getLeavesTotal
from mapFolding.benchmarks import recordBenchmarks

dtypeDefault = numpy.uint8
dtypeMaximum = numpy.uint16

leafAbove = numba.literally(0)
leafBelow = numba.literally(1)
countDimensionsGapped = numba.literally(2)
gapRangeStart = numba.literally(3)

@recordBenchmarks()
@numba.njit(cache=True, fastmath=False)
def foldings(listDimensions: List[int]):
    """
    Calculate the number of distinct possible ways to fold a map with given dimensions.
    This function computes the number of different ways a map can be folded along its grid lines,
    considering maps with at least two positive dimensions.

    Parameters:
        listDimensions : A list of integers representing the dimensions of the map. Must contain at least two positive dimensions.

    Returns
        foldingsTotal: The total number of possible distinct foldings for the given map dimensions.
    """
    def integerSmall(value):
        return numpy.uint8(value)
        # return numba.uint8(value)

    def integerLarge(value):
        return numpy.uint64(value)
        # return numba.uint64(value)

    listDimensionsPositive = validateListDimensions(listDimensions)

    # idk wtf `numba.literal_unroll()` is _supposed_ to do, but it turned `n` into a float which then turned `foldingsTotal` into a float
    # n = numba.literal_unroll(getLeavesTotal(listDimensionsPositive)) # leavesTotal: int = getLeavesTotal(listDimensionsPositive)
    # d = numba.literal_unroll(len(p)) # dimensionsTotal: int = len(listDimensions)
    leavesTotal = integerSmall(getLeavesTotal(listDimensionsPositive))
    dimensionsTotal = integerSmall(len(listDimensionsPositive))

    """How to build a leaf connection graph, also called a "Cartesian Product Decomposition" 
    or a "Dimensional Product Mapping", with sentinels: 
    Step 1: find the cumulative product of the map's dimensions"""
    cumulativeProduct = numpy.ones(dimensionsTotal + 1, dtype=dtypeDefault)
    for dimension1ndex in range(1, dimensionsTotal + 1):
        cumulativeProduct[dimension1ndex] = cumulativeProduct[dimension1ndex - 1] * listDimensions[dimension1ndex - 1]

    """Step 2: for each dimension, create a coordinate system """
    """coordinateSystem[dimension1ndex][leaf1ndex] holds the dimension1ndex-th coordinate of leaf leaf1ndex"""
    coordinateSystem = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=dtypeDefault)
    for dimension1ndex in range(1, dimensionsTotal + 1):
        for leaf1ndex in range(1, leavesTotal + 1):
            coordinateSystem[dimension1ndex][leaf1ndex] = ((leaf1ndex - 1) // cumulativeProduct[dimension1ndex - 1]) % listDimensions[dimension1ndex - 1] + 1

    """Step 3: create a huge empty connection graph"""
    connectionGraph = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=dtypeDefault)

    """Step for... for... for...: fill the connection graph"""
    for dimension1ndex in range(1, dimensionsTotal + 1):
        for activeLeaf1ndex in range(1, leavesTotal + 1):
            for leaf1ndexConnectee in range(1, activeLeaf1ndex + 1):
                if (coordinateSystem[dimension1ndex][activeLeaf1ndex] & 1) == (coordinateSystem[dimension1ndex][leaf1ndexConnectee] & 1):
                    if coordinateSystem[dimension1ndex][leaf1ndexConnectee] == 1:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee
                    else:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee - cumulativeProduct[dimension1ndex - 1]
                else: 
                    if coordinateSystem[dimension1ndex][leaf1ndexConnectee] == listDimensions[dimension1ndex - 1] or leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1] > activeLeaf1ndex:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee
                    else:
                        connectionGraph[dimension1ndex][activeLeaf1ndex][leaf1ndexConnectee] = leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1]

    """For numba, a single array is faster than four separate arrays"""
    track = numpy.zeros((4, leavesTotal + 1), dtype=dtypeDefault)

    """Indices of array `track` (to "track" the state), which is a collection of one-dimensional arrays each of length `leavesTotal + 1`."""
    """The values in the array cells are dynamic, small, unsigned integers."""
    potentialGaps = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximum)
    """
    c:/apps/mapFolding/mapFolding/lunnon.py:107: RuntimeWarning: overflow encountered in scalar multiply
    potentialGaps = numpy.zeros(leavesTotal * leavesTotal + 1, dtype=dtypeMaximum)
    """

    foldingsTotal = integerLarge(0)
    activeLeaf1ndex = integerSmall(1)
    activeGap1ndex = integerSmall(0)

    while activeLeaf1ndex > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow][0] == 1:
            if activeLeaf1ndex > leavesTotal:
                
                # foldingsTotal += integerLarge(n) # foldingsTotal += leavesTotal
                foldingsTotal += leavesTotal
            else:
                dimensionsUnconstrained = integerSmall(0)
                """Track possible gaps for activeLeaf1ndex in each section"""
                # gg = integerSmall(s[gapter][l - 1]) # gap1ndexLowerBound: int = track[gapRangeStart][activeLeaf1ndex - 1]
                gap1ndexLowerBound = track[gapRangeStart][activeLeaf1ndex - 1]
                """Reset gap index"""
                activeGap1ndex = gap1ndexLowerBound

                """Count possible gaps for activeLeaf1ndex in each section"""
                dimension1ndex = integerSmall(1)
                while dimension1ndex <= dimensionsTotal:
                    if connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex:
                        dimensionsUnconstrained += 1
                    else:
                        # m = integerSmall(D[i][l][l]) # leaf1ndexConnectee: int = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
                        leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
                        while leaf1ndexConnectee != activeLeaf1ndex:
                            potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                            if track[countDimensionsGapped][leaf1ndexConnectee] == 0:
                                gap1ndexLowerBound += 1
                            track[countDimensionsGapped][leaf1ndexConnectee] += 1
                            leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][track[leafBelow][leaf1ndexConnectee]]
                    dimension1ndex += 1

                """If activeLeaf1ndex is unconstrained in all sections, it can be inserted anywhere"""
                if dimensionsUnconstrained == dimensionsTotal:
                    leaf1ndex = integerSmall(0)
                    while leaf1ndex < activeLeaf1ndex:
                        potentialGaps[gap1ndexLowerBound] = leaf1ndex
                        gap1ndexLowerBound += 1
                        leaf1ndex += 1

                """Filter gaps that are common to all sections"""
                indexMiniGap = activeGap1ndex
                while indexMiniGap < gap1ndexLowerBound:
                    potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                    if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                        activeGap1ndex += 1
                    """Reset track[countDimensionsGapped] for next iteration"""
                    track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0
                    indexMiniGap += 1

        """Recursive backtracking steps"""
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart][activeLeaf1ndex - 1]:
            activeLeaf1ndex -= 1
            track[leafBelow][track[leafAbove][activeLeaf1ndex]] = track[leafBelow][activeLeaf1ndex]
            track[leafAbove][track[leafBelow][activeLeaf1ndex]] = track[leafAbove][activeLeaf1ndex]

        """Place leaf in valid position"""
        if activeLeaf1ndex > 0:
            activeGap1ndex -= 1
            track[leafAbove][activeLeaf1ndex] = potentialGaps[activeGap1ndex]
            track[leafBelow][activeLeaf1ndex] = track[leafBelow][track[leafAbove][activeLeaf1ndex]]
            track[leafBelow][track[leafAbove][activeLeaf1ndex]] = activeLeaf1ndex
            track[leafAbove][track[leafBelow][activeLeaf1ndex]] = activeLeaf1ndex
            """Save current gap index"""
            track[gapRangeStart][activeLeaf1ndex] = activeGap1ndex
            """Move to next leaf"""
            activeLeaf1ndex += 1
    return foldingsTotal
