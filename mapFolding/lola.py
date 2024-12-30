from numba import njit
import numpy
"""
ALL variables instantiated by `countFoldings` are numpy.NDArray instances.
ALL of those NDArray are indexed by variables defined in `lovelaceIndices.py`.

`doWork` has two `for` loops with a structure of `for identifier in range(p,q)`.
At the moment those two identifiers are primitive integers, rather than embedded in an NDArray instance.

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
"""
# Indices of array `the`, which holds unchanging, small, unsigned, integer values.
from mapFolding.lolaIndices import leavesTotal, dimensionsTotal, dimensionsPlus1
# Indices of array `track`, which is a collection of one-dimensional arrays each of length `the[leavesTotal] + 1`.
# The values in the array cells are dynamic, small, unsigned integers.
from mapFolding.lovelaceIndices import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart
# Indices of array `my`, which holds dynamic, small, unsigned, integer values.
from mapFolding.lolaIndices import activeLeaf1ndex, activeGap1ndex, unconstrainedLeaf, gap1ndexLowerBound, leaf1ndexConnectee, taskIndex, dimension1ndex, foldingsSubtotal, COUNTindicesDynamic
from mapFolding.benchmarks import recordBenchmarks
from typing import List
import numpy

def foldings(listDimensions: List[int]):
    from mapFolding.lolaIndices import leavesTotal, dimensionsTotal, dimensionsPlus1, COUNTindicesStatic

    static = numpy.zeros(COUNTindicesStatic, dtype=numpy.int64)

    from mapFolding.beDRY import outfitFoldings
    listDimensions, static[leavesTotal], D, track,potentialGaps = outfitFoldings(listDimensions)

    static[dimensionsTotal] = len(listDimensions)
    static[dimensionsPlus1] = static[dimensionsTotal] + 1

    # Pass listDimensions and taskDivisions to _sherpa for benchmarking
    foldingsTotal = _sherpa(track, potentialGaps, static, D, listDimensions)
    return foldingsTotal

@recordBenchmarks()
def _sherpa(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], gap: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], static: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], D: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], p: List[int]):
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
def countFoldings(TEMPLATEtrack: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    TEMPLATEpotentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    the: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    connectionGraph: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]
                    ):

    TEMPLATEmy = numpy.zeros(COUNTindicesDynamic, dtype=numpy.int64)
    TEMPLATEmy[activeLeaf1ndex] = 1

    taskDivisions = 0
    # taskDivisions = the[leavesTotal]
    TEMPLATEmy[taskIndex] = taskDivisions - 1 # the first modulo is leavesTotal - 1

    def prepareWork(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], 
                    potentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    my: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]) -> tuple[numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]]:
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
def doWork(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                potentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                my: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                the: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                connectionGraph: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
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

