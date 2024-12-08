from numba import njit
from typing import List, Optional
import numpy

@njit(cache=True, fastmath=True, error_model='numpy', nogil=True)
def foldings(dimensionsMap: List[int], listLeaves: Optional[List[int]] = None) -> int:
    """
    Calculate number of ways to fold a map of the dimensions, `dimensionsMap`.

    Parameters:
        dimensionsMap: list of dimensions [n, m ...]
        listLeaves (all): list of leaves to count foldings for; default is all

    Returns:
        foldingsTotal: Total number of valid foldings

    Key concepts
        - A "leaf" is a unit square in the map
        - A "gap" is a potential position where a new leaf can be folded
        - Connections track how leaves can connect above/below each other
        - The algorithm builds foldings incrementally by placing one leaf at a time
        - Backtracking explores all valid combinations
        - Leaves and dimensions are enumerated starting from 1, not 0; hence, leafNumber not leafIndex

    Key data structures
        - mapFoldingConnections[D][L][M]: How leaf L connects to leaf M in dimension D
        - listCountDimensionsWithGap[L]: Number of dimensions with valid gaps at leaf L
        - listGapIndicesRange[L]: Index ranges of gaps available for leaf L
        - listPotentialGapPositions[]: List of all potential gap positions

    Algorithm flow
        1. Initialize coordinate system and connection graph
        2. For each leaf:
            - Find valid gaps in each dimension
            - Place leaf in valid position
            - Backtrack when no valid positions remain
        3. Count total valid foldings found
    """
    leavesTotal = 1
    for dimension in dimensionsMap:
        leavesTotal *= dimension

    numberOfDimensions = len(dimensionsMap)

    # How to build a leafConnectionGraph:
    # Step 1: find the product of all dimensions
    productOfDimensions = numpy.ones(numberOfDimensions + 1, dtype=numpy.int64)
    for dimensionNumber in range(1, numberOfDimensions + 1):
        productOfDimensions[dimensionNumber] = productOfDimensions[dimensionNumber - 1] * dimensionsMap[dimensionNumber - 1]

    # Step 2: for each dimension, create a coordinate system
    coordinateSystem = numpy.zeros((numberOfDimensions + 1, leavesTotal + 1), dtype=numpy.int64)
    for dimensionNumber in range(1, numberOfDimensions + 1):
        for leafNumber in range(1, leavesTotal + 1):
            coordinateSystem[dimensionNumber][leafNumber] = ((leafNumber - 1) // productOfDimensions[dimensionNumber - 1]) % dimensionsMap[dimensionNumber - 1] + 1

    # Step 3: create a huge empty leafConnectionGraph
    leafConnectionGraph = numpy.zeros((numberOfDimensions + 1, leavesTotal + 1, leavesTotal + 1), dtype=numpy.int64)

    # Step for for for: fill the leafConnectionGraph
    for dimensionNumber in range(1, numberOfDimensions + 1):
        for leafNumberActive in range(1, leavesTotal + 1):
            for leafNumber in range(1, leafNumberActive + 1):
                distance = coordinateSystem[dimensionNumber][leafNumberActive] - coordinateSystem[dimensionNumber][leafNumber]
                if distance % 2 == 0:
                    # If distance is even
                    leafConnectionGraph[dimensionNumber][leafNumberActive][leafNumber] = (
                        leafNumber if coordinateSystem[dimensionNumber][leafNumber] == 1
                        else leafNumber - productOfDimensions[dimensionNumber - 1]
                    )
                else:
                    # If distance is odd
                    leafConnectionGraph[dimensionNumber][leafNumberActive][leafNumber] = (
                        leafNumber if (
                            coordinateSystem[dimensionNumber][leafNumber] == dimensionsMap[dimensionNumber - 1]
                            or leafNumber + productOfDimensions[dimensionNumber - 1] > leafNumberActive
                        ) else leafNumber + productOfDimensions[dimensionNumber - 1]
                    )

    if listLeaves is None:
        listLeaves = list(range(1, leavesTotal + 1))

    foldingsTotal = 0 # The point of the entire module

    for leafNumberActive in listLeaves:
        gapNumberActive = 0
        arrayConnectionLeafAbove = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)
        arrayConnectionLeafBelow = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)
        arrayCountDimensionsGap = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)
        arrayGapRanges = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)
        arrayPotentialGaps = numpy.zeros((leavesTotal + 1) * (leavesTotal + 1), dtype=numpy.int64)

        while leafNumberActive > 0:
            if leafNumberActive <= 1 or arrayConnectionLeafBelow[0] == 1:
                if leafNumberActive > leavesTotal:
                    foldingsTotal += leavesTotal
                else:
                    countDimensionsUnconstrained = 0
                    indexGapPointersMaximum = arrayGapRanges[leafNumberActive - 1]  # Track possible gaps
                    gapNumberActive = indexGapPointersMaximum

                    # Find potential gaps for leaf l in each dimension
                    for dimensionNumber in range(1, numberOfDimensions + 1):
                        if leafConnectionGraph[dimensionNumber][leafNumberActive][leafNumberActive] == leafNumberActive:
                            countDimensionsUnconstrained += 1
                        else:
                            leafNumber = leafConnectionGraph[dimensionNumber][leafNumberActive][leafNumberActive]
                            while leafNumber != leafNumberActive:
                                arrayPotentialGaps[indexGapPointersMaximum] = leafNumber
                                if arrayCountDimensionsGap[leafNumber] == 0:
                                    indexGapPointersMaximum += 1
                                arrayCountDimensionsGap[leafNumber] += 1
                                leafNumber = leafConnectionGraph[dimensionNumber][leafNumberActive][arrayConnectionLeafBelow[leafNumber]]

                    # If leaf l is unconstrained in all dimensions, it can be inserted anywhere
                    if countDimensionsUnconstrained == numberOfDimensions:
                        for leafNumber in range(leafNumberActive):
                            arrayPotentialGaps[indexGapPointersMaximum] = leafNumber
                            indexGapPointersMaximum += 1

                    for indexGaps in range(gapNumberActive, indexGapPointersMaximum):
                        arrayPotentialGaps[gapNumberActive] = arrayPotentialGaps[indexGaps]
                        if arrayCountDimensionsGap[arrayPotentialGaps[indexGaps]] == numberOfDimensions - countDimensionsUnconstrained:
                            gapNumberActive += 1
                        arrayCountDimensionsGap[arrayPotentialGaps[indexGaps]] = 0

            # Backtrack if no more gaps
            while leafNumberActive > 0 and gapNumberActive == arrayGapRanges[leafNumberActive - 1]:
                leafNumberActive -= 1
                arrayConnectionLeafBelow[arrayConnectionLeafAbove[leafNumberActive]] = arrayConnectionLeafBelow[leafNumberActive]
                arrayConnectionLeafAbove[arrayConnectionLeafBelow[leafNumberActive]] = arrayConnectionLeafAbove[leafNumberActive]

            # Insert leaf and advance
            if leafNumberActive > 0:
                gapNumberActive -= 1
                arrayConnectionLeafAbove[leafNumberActive] = arrayPotentialGaps[gapNumberActive]
                arrayConnectionLeafBelow[leafNumberActive] = arrayConnectionLeafBelow[arrayConnectionLeafAbove[leafNumberActive]]
                arrayConnectionLeafBelow[arrayConnectionLeafAbove[leafNumberActive]] = leafNumberActive
                arrayConnectionLeafAbove[arrayConnectionLeafBelow[leafNumberActive]] = leafNumberActive
                arrayGapRanges[leafNumberActive] = gapNumberActive
                leafNumberActive += 1

    return foldingsTotal
