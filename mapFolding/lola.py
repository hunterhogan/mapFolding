from numba import njit
import numpy
"""
ALL variables instantiated by `countFoldings` are numpy.NDArray instances.
ALL of those NDArray are indexed by variables defined in `lovelaceIndices.py`.

`doWhile` has three `for` loops with a structure of `for identifier in range(p,q)`.
At the moment those three identifiers are primitive integers, rather than embedded in an NDArray instance.

The six NDArray:
    Unchanging values
        - the
        - connectionGraph
    Dynamic values that are "personal" to each worker
        - my
        - track
        - potentialGaps
    Dynamic values that the workers could share safely
        - arrayFoldingsSubtotals

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
"""
# Indices of array `the`, which holds unchanging, small, unsigned, integer values.
from mapFolding.lovelaceIndices import taskDivisions, leavesTotal, dimensionsTotal, dimensionsPlus1, taskIndex as theTaskIndex 
# Indices of array `track`, which is a collection of one-dimensional arrays each of length `the[leavesTotal] + 1`. 
# The values in the array cells are dynamic, small, unsigned integers.
from mapFolding.lovelaceIndices import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart
# Indices of array `my`, which holds dynamic, small, unsigned, integer values.
from mapFolding.lolaIndices import activeLeaf1ndex, activeGap1ndex, unconstrainedLeaf, gap1ndexLowerBound, leaf1ndexConnectee, taskIndex, COUNTindicesDynamic

# numba warnings say there is nothing to parallelize in the module.
# @njit(cache=True, parallel=True, fastmath=False)
@njit(cache=True, fastmath=False)
def countFoldings(TEMPLATEtrack: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], 
                    TEMPLATEpotentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    the: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], 
                    connectionGraph: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]) -> int:

    TEMPLATEmy = numpy.zeros(COUNTindicesDynamic, dtype=numpy.int64)
    TEMPLATEmy[activeLeaf1ndex] = 1

    # NOTE pay attention to taskIndex: who creates it, when, where, and how
    TEMPLATEmy[taskIndex] = the[theTaskIndex]
    # the point of the runLolaRun experiments is dynamic process creation, 
    # which is tracked by the taskIndex variable

    arrayFoldingsSubtotals = numpy.zeros(the[taskDivisions] + 1, dtype=numpy.int64)
    def doWhile(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], 
                    potentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    my: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]) -> None:
        while my[activeLeaf1ndex] > 0:
            if my[activeLeaf1ndex] <= 1 or track[leafBelow][0] == 1:
                if my[activeLeaf1ndex] > the[leavesTotal]:
                    arrayFoldingsSubtotals[my[taskIndex]] += the[leavesTotal]
                else:
                    my[unconstrainedLeaf] = 0
                    my[gap1ndexLowerBound] = track[gapRangeStart][my[activeLeaf1ndex] - 1]
                    my[activeGap1ndex] = my[gap1ndexLowerBound]

                    for dimension1ndex in range(1, the[dimensionsPlus1]):
                    # for dimension1ndex in range(the[dimensionsTotal], 0, -1):
                        if connectionGraph[dimension1ndex][my[activeLeaf1ndex]][my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
                            my[unconstrainedLeaf] += 1
                        else:
                            my[leaf1ndexConnectee] = connectionGraph[dimension1ndex][my[activeLeaf1ndex]][my[activeLeaf1ndex]]
                            while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:

                                if the[taskDivisions] == 0 or my[activeLeaf1ndex] != the[taskDivisions] or my[leaf1ndexConnectee] % the[taskDivisions] == my[taskIndex]:
                                    potentialGaps[my[gap1ndexLowerBound]] = my[leaf1ndexConnectee]
                                    if track[countDimensionsGapped][my[leaf1ndexConnectee]] == 0:
                                        my[gap1ndexLowerBound] += 1
                                    track[countDimensionsGapped][my[leaf1ndexConnectee]] += 1

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

    doWhile(TEMPLATEtrack, TEMPLATEpotentialGaps, TEMPLATEmy)

    return numpy.sum(arrayFoldingsSubtotals).item()
