from mapFolding import outfitFoldings
from mapFolding.benchmarks import recordBenchmarks
from numba import njit
from typing import List
import numpy
from numpy.typing import NDArray

"""
It is possible to enter the main `while` loop from an arbitrary point. This version is "rotated" to effectively enter at the modulo operator.
"""

# Indices of array `track`, which is a collection of one-dimensional arrays each of length `the[leavesTotal] + 1`.
# The values in the array cells are dynamic, small, unsigned integers.
A = leafAbove = 0
"""Leaf above leaf m"""
B = leafBelow = 1
"""Leaf below leaf m"""
count = countDimensionsGapped = 2
"""Number of gaps available for leaf l"""
gapter = gapRangeStart = 3
"""Index of gap stack for leaf l"""

# Indices of array `my`, which holds dynamic, small, unsigned, integer values.
tricky = [
(activeLeaf1ndex := 0),
(activeGap1ndex := 1),
(unconstrainedLeaf := 2),
(gap1ndexLowerBound := 3),
(leaf1ndexConnectee := 4),
(taskIndex := 5),
(dimension1ndex := 6),
(foldingsSubtotal := 7),
]

COUNTindicesDynamic = len(tricky)

# Indices of array `the`, which holds unchanging, small, unsigned, integer values.
tricky = [
(dimensionsPlus1 := 0),
(dimensionsTotal := 1),
(leavesTotal := 2),
]

COUNTindicesStatic = len(tricky)

def countFolds(listDimensions: List[int]):
    static = numpy.zeros(COUNTindicesStatic, dtype=numpy.int64)

    listDimensions, static[leavesTotal], D, track,potentialGaps = outfitFoldings(listDimensions)

    static[dimensionsTotal] = len(listDimensions)
    static[dimensionsPlus1] = static[dimensionsTotal] + 1

    # Pass listDimensions and taskDivisions to _sherpa for benchmarking
    foldingsTotal = _sherpa(track, potentialGaps, static, D, listDimensions)
    return foldingsTotal

# @recordBenchmarks()
def _sherpa(track: NDArray, gap: NDArray, static: NDArray, D: NDArray, p: List[int]):
    """Performance critical section that counts foldings.

    Parameters:
        track: Array tracking folding state
        gap: Array for potential gaps
        static: Array containing static configuration values
        D: Array of leaf connections
        p: List of dimensions for benchmarking
    """
    foldingsTotal = countFoldings(track, gap, static, D)
    return foldingsTotal

@njit(cache=True, parallel=False, fastmath=False)
def countFoldings(TEMPLATEtrack: NDArray,
                    TEMPLATEpotentialGaps: NDArray,
                    the: NDArray,
                    connectionGraph: NDArray
                    ):

    TEMPLATEmy = numpy.zeros(COUNTindicesDynamic, dtype=numpy.int64)
    TEMPLATEmy[activeLeaf1ndex] = 1

    taskDivisions = 0
    # taskDivisions = the[leavesTotal]
    TEMPLATEmy[taskIndex] = taskDivisions - 1 # the first modulo is leavesTotal - 1

    def prepareWork(track: NDArray,
                    potentialGaps: NDArray,
                    my: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        foldingsTotal = 0
        while True:
            if my[activeLeaf1ndex] <= 1 or track[leafBelow][0] == 1:
                if my[activeLeaf1ndex] > the[leavesTotal]:
                    foldingsTotal += the[leavesTotal]
                else:
                    my[unconstrainedLeaf] = 0
                    my[gap1ndexLowerBound] = track[gapRangeStart][my[activeLeaf1ndex] - 1]
                    my[activeGap1ndex] = my[gap1ndexLowerBound]

                    for PREPAREdimension1ndex in range(1, the[dimensionsPlus1]):
                        if connectionGraph[PREPAREdimension1ndex][my[activeLeaf1ndex]][my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
                            my[unconstrainedLeaf] += 1
                        else:
                            my[leaf1ndexConnectee] = connectionGraph[PREPAREdimension1ndex][my[activeLeaf1ndex]][my[activeLeaf1ndex]]
                            while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:

                                if my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                                    my[dimension1ndex] = PREPAREdimension1ndex
                                    return track, potentialGaps, my

                                if my[activeLeaf1ndex] != the[leavesTotal]:
                                    potentialGaps[my[gap1ndexLowerBound]] = my[leaf1ndexConnectee]
                                    if track[countDimensionsGapped][my[leaf1ndexConnectee]] == 0:
                                        my[gap1ndexLowerBound] += 1
                                    track[countDimensionsGapped][my[leaf1ndexConnectee]] += 1
                                else:
                                    print("else")
                                    my[dimension1ndex] = PREPAREdimension1ndex
                                    return track, potentialGaps, my
                                    # PREPAREmy[leaf1ndexConnectee] % the[leavesTotal] == PREPAREmy[taskIndex]
                                my[leaf1ndexConnectee] = connectionGraph[dimension1ndex][my[activeLeaf1ndex]][track[leafBelow][my[leaf1ndexConnectee]]]

                    if my[unconstrainedLeaf] == the[dimensionsTotal]:
                        for leaf1ndex in range(my[activeLeaf1ndex]):
                            potentialGaps[my[gap1ndexLowerBound]] = leaf1ndex
                            my[gap1ndexLowerBound] += 1

                    for indexMiniGap in range(my[activeGap1ndex], my[gap1ndexLowerBound]):
                        potentialGaps[my[activeGap1ndex]] = potentialGaps[indexMiniGap]
                        if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == the[dimensionsTotal] - my[unconstrainedLeaf]:
                            my[activeGap1ndex] += 1
                        track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0

            while my[activeLeaf1ndex] > 0 and my[activeGap1ndex] == track[gapRangeStart][my[activeLeaf1ndex] - 1]:
                my[activeLeaf1ndex] -= 1
                track[leafBelow][track[leafAbove][my[activeLeaf1ndex]]] = track[leafBelow][my[activeLeaf1ndex]]
                track[leafAbove][track[leafBelow][my[activeLeaf1ndex]]] = track[leafAbove][my[activeLeaf1ndex]]

            if my[activeLeaf1ndex] > 0:
                my[activeGap1ndex] -= 1
                track[leafAbove][my[activeLeaf1ndex]] = potentialGaps[my[activeGap1ndex]]
                track[leafBelow][my[activeLeaf1ndex]] = track[leafBelow][track[leafAbove][my[activeLeaf1ndex]]]
                track[leafBelow][track[leafAbove][my[activeLeaf1ndex]]] = my[activeLeaf1ndex]
                track[leafAbove][track[leafBelow][my[activeLeaf1ndex]]] = my[activeLeaf1ndex]
                track[gapRangeStart][my[activeLeaf1ndex]] = my[activeGap1ndex]
                my[activeLeaf1ndex] += 1

    RETURNtrack, RETURNpotentialGaps, RETURNmy = prepareWork(TEMPLATEtrack.copy(), TEMPLATEpotentialGaps.copy(), TEMPLATEmy.copy())

    foldingsTotal = doWork(RETURNtrack.copy(), RETURNpotentialGaps.copy(), RETURNmy.copy(), the, connectionGraph, taskDivisions)

    return foldingsTotal

@njit(cache=True, parallel=False, fastmath=False)
def doWork(track: NDArray,
                potentialGaps: NDArray,
                my: NDArray,
                the: NDArray,
                connectionGraph: NDArray,
                taskDivisions: int = 0
                ):

    papasGotABrandNewBag = True
    if_activeLeaf1ndex_LTE_1_or_leafBelow_index_0_equals_1 = True
    for_dimension1ndex_in_range_1_to_dimensionsPlus1 = True
    while_leaf1ndexConnectee_notEquals_activeLeaf1ndex = True

    thisIsNotTheFirstPass = False

    while papasGotABrandNewBag:
        if my[activeLeaf1ndex] <= 1 or track[leafBelow][0] == 1 or if_activeLeaf1ndex_LTE_1_or_leafBelow_index_0_equals_1 == True:
            if_activeLeaf1ndex_LTE_1_or_leafBelow_index_0_equals_1 = False
            if my[activeLeaf1ndex] > the[leavesTotal] and thisIsNotTheFirstPass:
                my[foldingsSubtotal] += the[leavesTotal]
            else:
                if thisIsNotTheFirstPass:
                    my[unconstrainedLeaf] = 0
                    my[gap1ndexLowerBound] = track[gapRangeStart][my[activeLeaf1ndex] - 1]
                    my[activeGap1ndex] = my[gap1ndexLowerBound]

                for_dimension1ndex_in_range_1_to_dimensionsPlus1 = True
                while for_dimension1ndex_in_range_1_to_dimensionsPlus1 == True:
                    for_dimension1ndex_in_range_1_to_dimensionsPlus1 = False
                    if connectionGraph[my[dimension1ndex]][my[activeLeaf1ndex]][my[activeLeaf1ndex]] == my[activeLeaf1ndex] and thisIsNotTheFirstPass:
                        my[unconstrainedLeaf] += 1
                    else:
                        if thisIsNotTheFirstPass:
                            my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex]][my[activeLeaf1ndex]][my[activeLeaf1ndex]]
                        if my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                            while_leaf1ndexConnectee_notEquals_activeLeaf1ndex = True

                        while while_leaf1ndexConnectee_notEquals_activeLeaf1ndex == True:
                            while_leaf1ndexConnectee_notEquals_activeLeaf1ndex = False
                            thisIsNotTheFirstPass = True
                            if taskDivisions==0 or my[activeLeaf1ndex] != taskDivisions:
                                myTask = True
                            else:
                                modulo = my[leaf1ndexConnectee] % the[leavesTotal]
                                if modulo == my[taskIndex]: myTask = True
                                else:
                                    myTask = False
                            if myTask:
                                potentialGaps[my[gap1ndexLowerBound]] = my[leaf1ndexConnectee]
                                if track[countDimensionsGapped][my[leaf1ndexConnectee]] == 0:
                                    my[gap1ndexLowerBound] += 1
                                track[countDimensionsGapped][my[leaf1ndexConnectee]] += 1
                            my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex]][my[activeLeaf1ndex]][track[leafBelow][my[leaf1ndexConnectee]]]
                            if my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                                while_leaf1ndexConnectee_notEquals_activeLeaf1ndex = True
                    my[dimension1ndex] += 1
                    if my[dimension1ndex] < the[dimensionsPlus1]:
                        for_dimension1ndex_in_range_1_to_dimensionsPlus1 = True
                    else:
                        my[dimension1ndex] = 1

                if my[unconstrainedLeaf] == the[dimensionsTotal]:
                    for leaf1ndex in range(my[activeLeaf1ndex]):
                        potentialGaps[my[gap1ndexLowerBound]] = leaf1ndex
                        my[gap1ndexLowerBound] += 1

                for indexMiniGap in range(my[activeGap1ndex], my[gap1ndexLowerBound]):
                    potentialGaps[my[activeGap1ndex]] = potentialGaps[indexMiniGap]
                    if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == the[dimensionsTotal] - my[unconstrainedLeaf]:
                        my[activeGap1ndex] += 1
                    track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0

        while my[activeLeaf1ndex] > 0 and my[activeGap1ndex] == track[gapRangeStart][my[activeLeaf1ndex] - 1]:
            my[activeLeaf1ndex] -= 1
            track[leafBelow][track[leafAbove][my[activeLeaf1ndex]]] = track[leafBelow][my[activeLeaf1ndex]]
            track[leafAbove][track[leafBelow][my[activeLeaf1ndex]]] = track[leafAbove][my[activeLeaf1ndex]]

        if my[activeLeaf1ndex] > 0:
            my[activeGap1ndex] -= 1
            track[leafAbove][my[activeLeaf1ndex]] = potentialGaps[my[activeGap1ndex]]
            track[leafBelow][my[activeLeaf1ndex]] = track[leafBelow][track[leafAbove][my[activeLeaf1ndex]]]
            track[leafBelow][track[leafAbove][my[activeLeaf1ndex]]] = my[activeLeaf1ndex]
            track[leafAbove][track[leafBelow][my[activeLeaf1ndex]]] = my[activeLeaf1ndex]
            track[gapRangeStart][my[activeLeaf1ndex]] = my[activeGap1ndex]
            my[activeLeaf1ndex] += 1

        if my[activeLeaf1ndex] <= 0:
            return my[foldingsSubtotal]