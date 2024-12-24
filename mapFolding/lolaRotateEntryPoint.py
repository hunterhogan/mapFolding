from numba import njit
import numpy
"""
ALL variables instantiated by `countFoldings` are numpy.NDArray instances.
ALL of those NDArray are indexed by variables defined in `lovelaceIndices.py`.

`doWork` has three `for` loops with a structure of `for identifier in range(p,q)`.
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
from mapFolding.lolaIndices import activeLeaf1ndex, activeGap1ndex, unconstrainedLeaf, gap1ndexLowerBound, leaf1ndexConnectee, taskIndex, dimension1ndex, COUNTindicesDynamic

# numba warnings say there is nothing to parallelize in the module.
# @njit(cache=True, fastmath=False)
@njit(cache=True, parallel=True, fastmath=False)
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

    arrayFoldingsSubtotals = numpy.zeros(the[leavesTotal] + 1, dtype=numpy.int64)

    papasGotABrandNewBag = True
    def doWork(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    potentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    my: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]) -> None:

        if_l_LTE_1_or_Bindex0_equals_1 = True
        for_i_inRange1_dplus1 = True
        while_m_notEqualTo_l = True

        thisIsNotTheFirstPass = False

        while papasGotABrandNewBag:
            if my[activeLeaf1ndex] <= 1 or track[leafBelow][0] == 1 or if_l_LTE_1_or_Bindex0_equals_1 == True:
                if_l_LTE_1_or_Bindex0_equals_1 = False
                if my[activeLeaf1ndex] > the[leavesTotal] and thisIsNotTheFirstPass:
                    arrayFoldingsSubtotals[my[taskIndex]] += the[leavesTotal]
                else:
                    if thisIsNotTheFirstPass:
                        my[unconstrainedLeaf] = 0
                        my[gap1ndexLowerBound] = track[gapRangeStart][my[activeLeaf1ndex] - 1]
                        my[activeGap1ndex] = my[gap1ndexLowerBound]

                    for_i_inRange1_dplus1 = True
                    while for_i_inRange1_dplus1 == True: # for dimension1ndex in range(1, the[dimensionsPlus1]):
                        for_i_inRange1_dplus1 = False
                        if connectionGraph[my[dimension1ndex]][my[activeLeaf1ndex]][my[activeLeaf1ndex]] == my[activeLeaf1ndex] and thisIsNotTheFirstPass:
                            my[unconstrainedLeaf] += 1
                        else:
                            if thisIsNotTheFirstPass:
                                my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex]][my[activeLeaf1ndex]][my[activeLeaf1ndex]]
                            if my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                                while_m_notEqualTo_l = True

                            while while_m_notEqualTo_l == True: # while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                                while_m_notEqualTo_l = False
                                thisIsNotTheFirstPass = True
                                if the[taskDivisions] == 0 or my[activeLeaf1ndex] != the[taskDivisions] or my[leaf1ndexConnectee] % the[taskDivisions] == my[taskIndex]:
                                    potentialGaps[my[gap1ndexLowerBound]] = my[leaf1ndexConnectee]
                                    if track[countDimensionsGapped][my[leaf1ndexConnectee]] == 0:
                                        my[gap1ndexLowerBound] += 1
                                    track[countDimensionsGapped][my[leaf1ndexConnectee]] += 1
                                my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex]][my[activeLeaf1ndex]][track[leafBelow][my[leaf1ndexConnectee]]]
                                if my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                                    while_m_notEqualTo_l = True
# end if connectionGraph[dimension1ndex][my[activeLeaf1ndex]][my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
# end while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                        my[dimension1ndex] += 1
                        if my[dimension1ndex] < the[dimensionsPlus1]:
                            for_i_inRange1_dplus1 = True
                        else:
                            my[dimension1ndex] = 1
# end for dimension1ndex in range(1, the[dimensionsPlus1]):

                    if my[unconstrainedLeaf] == the[dimensionsTotal]:
                        for leaf1ndex in range(my[activeLeaf1ndex]):
                            potentialGaps[my[gap1ndexLowerBound]] = leaf1ndex
                            my[gap1ndexLowerBound] += 1

                    for indexMiniGap in range(my[activeGap1ndex], my[gap1ndexLowerBound]):
                        potentialGaps[my[activeGap1ndex]] = potentialGaps[indexMiniGap]
                        if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == the[dimensionsTotal] - my[unconstrainedLeaf]:
                            my[activeGap1ndex] += 1
                        track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0
# end if my[activeLeaf1ndex] <= 1 or track[leafBelow][0] == 1:

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
                break
# logical/typical end while my[activeLeaf1ndex] > 0:

    def prepareWork(PREPAREtrack: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], 
                    PREPAREpotentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    PREPAREmy: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]) -> tuple[numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]]:
        while True:
            if PREPAREmy[activeLeaf1ndex] <= 1 or PREPAREtrack[leafBelow][0] == 1:
                if PREPAREmy[activeLeaf1ndex] > the[leavesTotal]:
                    arrayFoldingsSubtotals[PREPAREmy[taskIndex]] += the[leavesTotal]
                else:
                    PREPAREmy[unconstrainedLeaf] = 0
                    PREPAREmy[gap1ndexLowerBound] = PREPAREtrack[gapRangeStart][PREPAREmy[activeLeaf1ndex] - 1]
                    PREPAREmy[activeGap1ndex] = PREPAREmy[gap1ndexLowerBound]

                    for PREPAREdimension1ndex in range(1, the[dimensionsPlus1]):
                        if connectionGraph[PREPAREdimension1ndex][PREPAREmy[activeLeaf1ndex]][PREPAREmy[activeLeaf1ndex]] == PREPAREmy[activeLeaf1ndex]:
                            PREPAREmy[unconstrainedLeaf] += 1
                        else:
                            PREPAREmy[leaf1ndexConnectee] = connectionGraph[PREPAREdimension1ndex][PREPAREmy[activeLeaf1ndex]][PREPAREmy[activeLeaf1ndex]]
                            if PREPAREmy[leaf1ndexConnectee] != PREPAREmy[activeLeaf1ndex]:
                                PREPAREmy[dimension1ndex] = PREPAREdimension1ndex
                                return PREPAREtrack, PREPAREpotentialGaps, PREPAREmy

                    if PREPAREmy[unconstrainedLeaf] == the[dimensionsTotal]:
                        for leaf1ndex in range(PREPAREmy[activeLeaf1ndex]):
                            PREPAREpotentialGaps[PREPAREmy[gap1ndexLowerBound]] = leaf1ndex
                            PREPAREmy[gap1ndexLowerBound] += 1

                    for indexMiniGap in range(PREPAREmy[activeGap1ndex], PREPAREmy[gap1ndexLowerBound]):
                        PREPAREpotentialGaps[PREPAREmy[activeGap1ndex]] = PREPAREpotentialGaps[indexMiniGap]
                        if PREPAREtrack[countDimensionsGapped][PREPAREpotentialGaps[indexMiniGap]] == the[dimensionsTotal] - PREPAREmy[unconstrainedLeaf]:
                            PREPAREmy[activeGap1ndex] += 1
                        PREPAREtrack[countDimensionsGapped][PREPAREpotentialGaps[indexMiniGap]] = 0

            while PREPAREmy[activeLeaf1ndex] > 0 and PREPAREmy[activeGap1ndex] == PREPAREtrack[gapRangeStart][PREPAREmy[activeLeaf1ndex] - 1]:
                PREPAREmy[activeLeaf1ndex] -= 1
                PREPAREtrack[leafBelow][PREPAREtrack[leafAbove][PREPAREmy[activeLeaf1ndex]]] = PREPAREtrack[leafBelow][PREPAREmy[activeLeaf1ndex]]
                PREPAREtrack[leafAbove][PREPAREtrack[leafBelow][PREPAREmy[activeLeaf1ndex]]] = PREPAREtrack[leafAbove][PREPAREmy[activeLeaf1ndex]]

            if PREPAREmy[activeLeaf1ndex] > 0:
                PREPAREmy[activeGap1ndex] -= 1
                PREPAREtrack[leafAbove][PREPAREmy[activeLeaf1ndex]] = PREPAREpotentialGaps[PREPAREmy[activeGap1ndex]]
                PREPAREtrack[leafBelow][PREPAREmy[activeLeaf1ndex]] = PREPAREtrack[leafBelow][PREPAREtrack[leafAbove][PREPAREmy[activeLeaf1ndex]]]
                PREPAREtrack[leafBelow][PREPAREtrack[leafAbove][PREPAREmy[activeLeaf1ndex]]] = PREPAREmy[activeLeaf1ndex]
                PREPAREtrack[leafAbove][PREPAREtrack[leafBelow][PREPAREmy[activeLeaf1ndex]]] = PREPAREmy[activeLeaf1ndex]
                PREPAREtrack[gapRangeStart][PREPAREmy[activeLeaf1ndex]] = PREPAREmy[activeGap1ndex] 
                PREPAREmy[activeLeaf1ndex] += 1

    RETURNtrack, RETURNpotentialGaps, RETURNmy = prepareWork(TEMPLATEtrack.copy(), TEMPLATEpotentialGaps.copy(), TEMPLATEmy.copy())

    doWork(RETURNtrack.copy(), RETURNpotentialGaps.copy(), RETURNmy.copy())

    return numpy.sum(arrayFoldingsSubtotals).item()
