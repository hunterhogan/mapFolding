from numba import njit
import numpy
# Indices of array `the`, which holds unchanging, small, unsigned, integer values.
from mapFolding.lovelaceIndices import leavesTotal, dimensionsTotal, dimensionsPlus1
# Indices of array `track`, which is a collection of one-dimensional arrays each of length `the[leavesTotal] + 1`.
# The values in the array cells are dynamic, small, unsigned integers.
from mapFolding.lovelaceIndices import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart
# Indices of array `my`, which holds dynamic, small, unsigned, integer values, except `foldingsSubtotalI`, which is a dynamic, large, unsigned integer.
from mapFolding.lolaIndices import activeLeaf1ndex, activeGap1ndex, unconstrainedLeaf, gap1ndexLowerBound, leaf1ndexConnectee, taskIndex, dimension1ndex, foldingsSubtotal

@njit(cache=True, parallel=False, fastmath=False)
def countSubtotal(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    potentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    my: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    the: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    connectionGraph: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]) -> int:

    taskDivisions = the[leavesTotal]
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
            break

    return int(my[foldingsSubtotal])
