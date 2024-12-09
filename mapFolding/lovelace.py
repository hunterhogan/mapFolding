from numba import njit, types
import numpy
import numba

ConnectionLeafAbove = 0
ConnectionLeafBelow = 1
CountDimensionsGap  = 2
GapRanges           = 3

leavesTotal        = 0
dimensionsTotal    = 1
leafNumberActive   = 2
foldingsSubtotal   = 3
gapNumberActive    = 4
countUnconstrained = 5
GG                 = 6
leafNumber         = 7

@njit(types.int64(types.Array(types.int64, 1, 'C'), types.Array(types.int64, 3, 'C', readonly=True), types.Array(types.int64, 2, 'C'), types.Array(types.int64, 1, 'C')), cache=True, fastmath=True, error_model='numpy', nogil=True)
def _countLeaf(arrayState: numpy.ndarray, leafConnectionGraph: numpy.ndarray, track: numpy.ndarray, arrayPotentialGaps: numpy.ndarray) -> int:
    while arrayState[leafNumberActive] > 0:
        if arrayState[leafNumberActive] <= 1 or track[ConnectionLeafBelow][0] == 1:
            if arrayState[leafNumberActive] > arrayState[leavesTotal]:
                arrayState[foldingsSubtotal] += arrayState[leavesTotal]
            else:
                arrayState[countUnconstrained] = 0
                arrayState[GG] = track[GapRanges][arrayState[leafNumberActive] - 1]  # Track possible gaps
                arrayState[gapNumberActive] = arrayState[GG]

                # Find potential gaps for leaf l in each dimension
                for dimensionNumber in range(1, arrayState[dimensionsTotal] + 1):
                    if leafConnectionGraph[dimensionNumber][arrayState[leafNumberActive]][arrayState[leafNumberActive]] == arrayState[leafNumberActive]:
                        arrayState[countUnconstrained] += 1
                    else:
                        arrayState[leafNumber] = leafConnectionGraph[dimensionNumber][arrayState[leafNumberActive]][arrayState[leafNumberActive]]
                        while arrayState[leafNumber] != arrayState[leafNumberActive]:
                            arrayPotentialGaps[arrayState[GG]] = arrayState[leafNumber]
                            if track[CountDimensionsGap][arrayState[leafNumber]] == 0:
                                arrayState[GG] += 1
                            track[CountDimensionsGap][arrayState[leafNumber]] += 1
                            arrayState[leafNumber] = leafConnectionGraph[dimensionNumber][arrayState[leafNumberActive]][track[ConnectionLeafBelow][arrayState[leafNumber]]]

                # If leaf l is unconstrained in all dimensions, it can be inserted anywhere
                if arrayState[countUnconstrained] == arrayState[dimensionsTotal]:
                    for arrayState[leafNumber] in range(arrayState[leafNumberActive]):
                        arrayPotentialGaps[arrayState[GG]] = arrayState[leafNumber]
                        arrayState[GG] += 1

                for indexGaps in range(arrayState[gapNumberActive], arrayState[GG]):
                    arrayPotentialGaps[arrayState[gapNumberActive]] = arrayPotentialGaps[indexGaps]
                    if track[CountDimensionsGap][arrayPotentialGaps[indexGaps]] == arrayState[dimensionsTotal] - arrayState[countUnconstrained]:
                        arrayState[gapNumberActive] += 1
                    track[CountDimensionsGap][arrayPotentialGaps[indexGaps]] = 0

            # Backtrack if no more gaps
        while arrayState[leafNumberActive] > 0 and arrayState[gapNumberActive] == track[GapRanges][arrayState[leafNumberActive] - 1]:
            arrayState[leafNumberActive] -= 1
            track[ConnectionLeafBelow][track[ConnectionLeafAbove][arrayState[leafNumberActive]]] = track[ConnectionLeafBelow][arrayState[leafNumberActive]]
            track[ConnectionLeafAbove][track[ConnectionLeafBelow][arrayState[leafNumberActive]]] = track[ConnectionLeafAbove][arrayState[leafNumberActive]]

            # Insert leaf and advance
        if arrayState[leafNumberActive] > 0:
            arrayState[gapNumberActive] -= 1
            track[ConnectionLeafAbove][arrayState[leafNumberActive]] = arrayPotentialGaps[arrayState[gapNumberActive]]
            track[ConnectionLeafBelow][arrayState[leafNumberActive]] = track[ConnectionLeafBelow][track[ConnectionLeafAbove][arrayState[leafNumberActive]]]
            track[ConnectionLeafBelow][track[ConnectionLeafAbove][arrayState[leafNumberActive]]] = arrayState[leafNumberActive]
            track[ConnectionLeafAbove][track[ConnectionLeafBelow][arrayState[leafNumberActive]]] = arrayState[leafNumberActive]
            track[GapRanges][arrayState[leafNumberActive]] = arrayState[gapNumberActive]
            arrayState[leafNumberActive] += 1

    return int(arrayState[foldingsSubtotal])

@njit(types.int64(types.Array(types.int64, 1, 'C', readonly=True), types.Array(types.int64, 1, 'C', readonly=True)), cache=True, fastmath=True, error_model='numpy', nogil=True)
def _makeDataStructures(dimensionsMap: numpy.ndarray, listLeaves: numpy.ndarray) -> int:
    the = numpy.zeros(8, dtype=numpy.int64)

    the[leavesTotal] = 1
    for dimension in dimensionsMap:
        the[leavesTotal] *= dimension

    the[dimensionsTotal] = len(dimensionsMap)

    # How to build a leafConnectionGraph:
    # Step 1: find the product of all dimensions
    productOfDimensions = numpy.ones(the[dimensionsTotal] + 1, dtype=numpy.int64)
    for dimensionNumber in range(1, the[dimensionsTotal] + 1):
        productOfDimensions[dimensionNumber] = productOfDimensions[dimensionNumber - 1] * dimensionsMap[dimensionNumber - 1]

    # Step 2: for each dimension, create a coordinate system
    coordinateSystem = numpy.zeros((the[dimensionsTotal] + 1, the[leavesTotal] + 1), dtype=numpy.int64)
    for dimensionNumber in range(1, the[dimensionsTotal] + 1):
        for leafFriend in range(1, the[leavesTotal] + 1):
            coordinateSystem[dimensionNumber][leafFriend] = ((leafFriend - 1) // productOfDimensions[dimensionNumber - 1]) % dimensionsMap[dimensionNumber - 1] + 1

    # Step 3: create a huge empty leafConnectionGraph
    leafConnectionGraph = numpy.zeros((the[dimensionsTotal] + 1, the[leavesTotal] + 1, the[leavesTotal] + 1), dtype=numpy.int64)

    # Step for for for: fill the leafConnectionGraph
    for dimensionNumber in range(1, the[dimensionsTotal] + 1):
        for leafProxy in range(1, the[leavesTotal] + 1):
            for leafFriend in range(1, leafProxy + 1):
                distance = coordinateSystem[dimensionNumber][leafProxy] - coordinateSystem[dimensionNumber][leafFriend]
                if distance % 2 == 0:
                    # If distance is even
                    leafConnectionGraph[dimensionNumber][leafProxy][leafFriend] = (
                        leafFriend if coordinateSystem[dimensionNumber][leafFriend] == 1
                        else leafFriend - productOfDimensions[dimensionNumber - 1]
                    )
                else:
                    # If distance is odd
                    leafConnectionGraph[dimensionNumber][leafProxy][leafFriend] = (
                        leafFriend if (
                            coordinateSystem[dimensionNumber][leafFriend] == dimensionsMap[dimensionNumber - 1]
                            or leafFriend + productOfDimensions[dimensionNumber - 1] > leafProxy
                        ) else leafFriend + productOfDimensions[dimensionNumber - 1]
                    )

    track = numpy.zeros((4, the[leavesTotal] + 1), dtype=numpy.int64)
    arrayPotentialGaps = numpy.zeros((the[leavesTotal] + 1) * (the[leavesTotal] + 1), dtype=numpy.int64)

    foldingsTotal = 0 # The point of the entire module

    the[leafNumberActive] = 1

    return _countLeaf(the, leafConnectionGraph, track, arrayPotentialGaps)

