from numba import njit, types, prange
import numpy

"""
Hypotheses:
- The counting loop should only have necessary logic and data structures
- The rest of the package should prioritize the efficiency of the counting loop
- Static values should be in static data structures
- Dynamic values should be in well organized ndarray
"""

# Dynamic
## arrayPotentialGaps
## `track` indices
ConnectionLeafAbove = 0
ConnectionLeafBelow = 1
CountDimensionsGap  = 2
GapRanges           = 3

# Static
## leafConnectionGraph

## `Z0Z_` indices
# needs to be divided into two parts: static and dynamic
listIndicesZ0Z_ = [
## Z0Z_ static data
    'leavesTotal',       
    'dimensionsTotal',
    'tasksTotal',
    'taskActive',
    ## Z0Z_ dynamic data
    'leafNumberActive',
    'foldingsSubtotal',
    'gapNumberActive',
    'countUnconstrained',
    'GG',
    'leafNumber',
]
# TODO make this more compatible with type checking
for index, indexer in enumerate(listIndicesZ0Z_):
    globals()[indexer] = index
indexersTotal = len(listIndicesZ0Z_)
# TODO think about "number" vs "index". 
# Ohh, crazy thought: "1ndex" instead of "index". 
# If these were identifiers, 1ndex would be prohibited, but these
# are keyNames, so it's okay. I don't want to use "index" because
# it creates confusion: in Python an index starts at 0, but in
# the context of the algorithm, most indexers start at 1. "1ndex"
# could signal that there is a difference between the two. Hell,
# it's a better signal than "number": "number" even confuses me.

@njit(types.int64(types.Array(types.int64, 1, 'C'), types.Array(types.int64, 3, 'C', readonly=True), types.Array(types.int64, 2, 'C'), types.Array(types.int64, 1, 'C')), cache=True, fastmath=True, error_model='numpy', nogil=True)
def _countLeaf(Z0Z_: numpy.ndarray, leafConnectionGraph: numpy.ndarray, track: numpy.ndarray, arrayPotentialGaps: numpy.ndarray) -> int:
    """
    HEY!
    Despite the many good reasons you avoid raising exceptions,
    because the counting function could take multiple days to complete,
    create unusually pedantic checks just prior to execution.
    """
    while Z0Z_[leafNumberActive] > 0:
        if Z0Z_[leafNumberActive] <= 1 or track[ConnectionLeafBelow][0] == 1:
            if Z0Z_[leafNumberActive] > Z0Z_[leavesTotal]:
                Z0Z_[foldingsSubtotal] += Z0Z_[leavesTotal]
            else:
                Z0Z_[countUnconstrained] = 0
                Z0Z_[GG] = track[GapRanges][Z0Z_[leafNumberActive] - 1]  # Track possible gaps
                Z0Z_[gapNumberActive] = Z0Z_[GG]

                # Find potential gaps for leaf l in each dimension
                for dimensionNumber in range(1, Z0Z_[dimensionsTotal] + 1):
                    if leafConnectionGraph[dimensionNumber][Z0Z_[leafNumberActive]][Z0Z_[leafNumberActive]] == Z0Z_[leafNumberActive]:
                        Z0Z_[countUnconstrained] += 1
                    else:
                        Z0Z_[leafNumber] = leafConnectionGraph[dimensionNumber][Z0Z_[leafNumberActive]][Z0Z_[leafNumberActive]]
                        while Z0Z_[leafNumber] != Z0Z_[leafNumberActive]:
                            if Z0Z_[tasksTotal] == 0 or Z0Z_[leafNumberActive] != Z0Z_[tasksTotal] or Z0Z_[leafNumber] % Z0Z_[tasksTotal] == Z0Z_[taskActive]:
                                arrayPotentialGaps[Z0Z_[GG]] = Z0Z_[leafNumber]
                                if track[CountDimensionsGap][Z0Z_[leafNumber]] == 0:
                                    Z0Z_[GG] += 1
                                track[CountDimensionsGap][Z0Z_[leafNumber]] += 1
                            Z0Z_[leafNumber] = leafConnectionGraph[dimensionNumber][Z0Z_[leafNumberActive]][track[ConnectionLeafBelow][Z0Z_[leafNumber]]]

                # If leaf l is unconstrained in all dimensions, it can be inserted anywhere
                if Z0Z_[countUnconstrained] == Z0Z_[dimensionsTotal]:
                    for Z0Z_[leafNumber] in range(Z0Z_[leafNumberActive]):
                        arrayPotentialGaps[Z0Z_[GG]] = Z0Z_[leafNumber]
                        Z0Z_[GG] += 1

                for indexGaps in range(Z0Z_[gapNumberActive], Z0Z_[GG]):
                    arrayPotentialGaps[Z0Z_[gapNumberActive]] = arrayPotentialGaps[indexGaps]
                    if track[CountDimensionsGap][arrayPotentialGaps[indexGaps]] == Z0Z_[dimensionsTotal] - Z0Z_[countUnconstrained]:
                        Z0Z_[gapNumberActive] += 1
                    track[CountDimensionsGap][arrayPotentialGaps[indexGaps]] = 0

            # Backtrack if no more gaps
        while Z0Z_[leafNumberActive] > 0 and Z0Z_[gapNumberActive] == track[GapRanges][Z0Z_[leafNumberActive] - 1]:
            Z0Z_[leafNumberActive] -= 1
            track[ConnectionLeafBelow][track[ConnectionLeafAbove][Z0Z_[leafNumberActive]]] = track[ConnectionLeafBelow][Z0Z_[leafNumberActive]]
            track[ConnectionLeafAbove][track[ConnectionLeafBelow][Z0Z_[leafNumberActive]]] = track[ConnectionLeafAbove][Z0Z_[leafNumberActive]]

            # Insert leaf and advance
        if Z0Z_[leafNumberActive] > 0:
            Z0Z_[gapNumberActive] -= 1
            track[ConnectionLeafAbove][Z0Z_[leafNumberActive]] = arrayPotentialGaps[Z0Z_[gapNumberActive]]
            track[ConnectionLeafBelow][Z0Z_[leafNumberActive]] = track[ConnectionLeafBelow][track[ConnectionLeafAbove][Z0Z_[leafNumberActive]]]
            track[ConnectionLeafBelow][track[ConnectionLeafAbove][Z0Z_[leafNumberActive]]] = Z0Z_[leafNumberActive]
            track[ConnectionLeafAbove][track[ConnectionLeafBelow][Z0Z_[leafNumberActive]]] = Z0Z_[leafNumberActive]
            track[GapRanges][Z0Z_[leafNumberActive]] = Z0Z_[gapNumberActive]
            Z0Z_[leafNumberActive] += 1

    return int(Z0Z_[foldingsSubtotal])

@njit(types.int64(types.Array(types.int64, 1, 'C', readonly=True), types.Array(types.int64, 1, 'C', readonly=True)), cache=True, fastmath=True, error_model='numpy', nogil=True, parallel=True)
def _makeDataStructures(dimensionsMap: numpy.ndarray, listLeaves: numpy.ndarray) -> int:
    # TODO `_makeDataStructures` should make the data structures, dimensionsMap should be passed here in its original format
    Z0Z_ = numpy.zeros(indexersTotal, dtype=numpy.int64) # the majority of the dynamic values initialize to 0
    Z0Z_[leafNumberActive] = 1

    Z0Z_[leavesTotal] = 1
    for dimension in dimensionsMap:
        Z0Z_[leavesTotal] *= dimension

    Z0Z_[dimensionsTotal] = len(dimensionsMap)

    # How to build a leafConnectionGraph:
    # Step 1: find the product of all dimensions
    productOfDimensions = numpy.ones(Z0Z_[dimensionsTotal] + 1, dtype=numpy.int64)
    for dimensionNumber in range(1, Z0Z_[dimensionsTotal] + 1):
        productOfDimensions[dimensionNumber] = productOfDimensions[dimensionNumber - 1] * dimensionsMap[dimensionNumber - 1]

    # Step 2: for each dimension, create a coordinate system
    coordinateSystem = numpy.zeros((Z0Z_[dimensionsTotal] + 1, Z0Z_[leavesTotal] + 1), dtype=numpy.int64)
    for dimensionNumber in range(1, Z0Z_[dimensionsTotal] + 1):
        for leafFriend in range(1, Z0Z_[leavesTotal] + 1):
            coordinateSystem[dimensionNumber][leafFriend] = ((leafFriend - 1) // productOfDimensions[dimensionNumber - 1]) % dimensionsMap[dimensionNumber - 1] + 1

    # Step 3: create a huge empty leafConnectionGraph
    leafConnectionGraph = numpy.zeros((Z0Z_[dimensionsTotal] + 1, Z0Z_[leavesTotal] + 1, Z0Z_[leavesTotal] + 1), dtype=numpy.int64)

    # Step for for for: fill the leafConnectionGraph
    for dimensionNumber in range(1, Z0Z_[dimensionsTotal] + 1):
        for leafProxy in range(1, Z0Z_[leavesTotal] + 1):
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

    track = numpy.zeros((4, Z0Z_[leavesTotal] + 1), dtype=numpy.int64)
    arrayPotentialGaps = numpy.zeros((Z0Z_[leavesTotal] + 1) * (Z0Z_[leavesTotal] + 1), dtype=numpy.int64)

    foldingsTotal = 0 # The point of the entire module

    # functional prototype logic
    computationIndex: int | list[int] | None = None
    computationDivisions = numpy.prod(dimensionsMap)*-
    listComputationIndices = [0]
    Z0Z_[tasksTotal] = computationDivisions
    if computationDivisions > 1:
        if computationIndex is None:
            listComputationIndices = range(computationDivisions)
        elif isinstance(computationIndex, int):
            listComputationIndices = [computationIndex]
        elif isinstance(computationIndex, list):
            listComputationIndices = computationIndex
        else:
            raise ValueError("computationIndex must be an integer or a list of integers")
        # and check that all values are within the range
        for index in prange(len(listComputationIndices)):
        # for index in listComputationIndices:
            Z0Z_[taskActive] = index
            foldingsTotal += _countLeaf(Z0Z_.copy(), leafConnectionGraph.copy(), track.copy(), arrayPotentialGaps.copy())
    else:
        foldingsTotal = _countLeaf(Z0Z_, leafConnectionGraph, track, arrayPotentialGaps)

    """
    This is the right idea, but the check should be against the entire list before calling any instances of the counting function.
    if 0 <= computationIndex < computationDivisions:
        Z0Z_[taskActive] = computationIndex
    else:
        raise ValueError("computationIndex must be between 0 and computationDivisions - 1")
    """

    return foldingsTotal

