import numpy
from numba import njit
from typing import List
@njit(cache=True, fastmath=True, error_model='numpy', nogil=True)
def foldings(dimensionsMap: List[int], computationDivisions:int = 0, computationIndex: int = 0) -> int: 
    """
    Calculate number of ways to fold a map with given dimensions.
    Parameters:
        dimensionsMap: list of dimensions [n, m] for nXm map or [n,n,n...] for n-dimensional
        computationDivisions (0): number of divisions for parallel computation
        computationIndex (0): index of current computation division (0 to computationDivisions-1)
    Returns:
        foldingsTotal: Total number of valid foldings
    """
    # Calculate total number of leaves
    leavesTotal = 1
    for dimension in dimensionsMap:
        leavesTotal *= dimension

    numberOfDimensions = len(dimensionsMap)

    # How to build a leafConnectionGraph:
    # Step 1: find the product of all dimensions
    productOfDimensions = numpy.ones(numberOfDimensions + 1, dtype=numpy.int64)
    for indexDimension in range(1, numberOfDimensions + 1):
        productOfDimensions[indexDimension] = productOfDimensions[indexDimension - 1] * dimensionsMap[indexDimension - 1]

    # Step 2: for each dimension, create a coordinate system
    coordinateSystem = numpy.zeros((numberOfDimensions + 1, leavesTotal + 1), dtype=numpy.int64)
    for indexDimension in range(1, numberOfDimensions + 1):
        for indexOfLeaves in range(1, leavesTotal + 1):
            coordinateSystem[indexDimension][indexOfLeaves] = ((indexOfLeaves - 1) // productOfDimensions[indexDimension - 1]) % dimensionsMap[indexDimension - 1] + 1

    # Step 3: create a huge empty leafConnectionGraph
    leafConnectionGraph = numpy.zeros((numberOfDimensions + 1, leavesTotal + 1, leavesTotal + 1), dtype=numpy.int64)

    # Step for for for: fill the leafConnectionGraph
    for indexDimension in range(1, numberOfDimensions + 1):
        for focalLeafIndex in range(1, leavesTotal + 1):
            for indexOfLeaves in range(1, focalLeafIndex + 1):
                distance = coordinateSystem[indexDimension][focalLeafIndex] - coordinateSystem[indexDimension][indexOfLeaves]
                if distance % 2 == 0:
                    # If distance is even
                    leafConnectionGraph[indexDimension][focalLeafIndex][indexOfLeaves] = indexOfLeaves if coordinateSystem[indexDimension][indexOfLeaves] == 1 else indexOfLeaves - productOfDimensions[indexDimension - 1]
                else:
                    # If distance is odd
                    leafConnectionGraph[indexDimension][focalLeafIndex][indexOfLeaves] = indexOfLeaves if (coordinateSystem[indexDimension][indexOfLeaves] == dimensionsMap[indexDimension - 1] or indexOfLeaves + productOfDimensions[indexDimension - 1] > focalLeafIndex) else indexOfLeaves + productOfDimensions[indexDimension - 1]

    # Initialize status arrays
    leafAboveStatusTracker = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)
    leafBelowStatusTracker = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)
    count = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)
    gapPointers = numpy.zeros(leavesTotal + 1, dtype=numpy.int64)
    allGaps = numpy.zeros((leavesTotal + 1) * (leavesTotal + 1), dtype=numpy.int64)

    # Initialize variables for backtracking
    foldingsTotal = 0

    # Yet more initializing: variables for backtracking
    g = 0              # Gap index
    focalLeafIndex = 1 # Current leaf

    # Main folding loop using a stack-based approach
    while focalLeafIndex > 0:
        if focalLeafIndex <= 1 or leafBelowStatusTracker[0] == 1:
            if focalLeafIndex > leavesTotal:
                foldingsTotal += leavesTotal
            else:
                dd = 0  # Number of sections where leaf l is unconstrained
                gg = gapPointers[focalLeafIndex - 1]  # Track possible gaps
                g = gg

                # Find potential gaps for leaf l in each dimension
                for indexDimension in range(1, numberOfDimensions + 1): # prange causes error
                    if leafConnectionGraph[indexDimension][focalLeafIndex][focalLeafIndex] == focalLeafIndex:
                        dd += 1
                    else:
                        indexOfLeaves = leafConnectionGraph[indexDimension][focalLeafIndex][focalLeafIndex]
                        while indexOfLeaves != focalLeafIndex:
                            if computationDivisions == 0 or focalLeafIndex != computationDivisions or indexOfLeaves % computationDivisions == computationIndex:
                                allGaps[gg] = indexOfLeaves
                                if count[indexOfLeaves] == 0:
                                    gg += 1
                                count[indexOfLeaves] += 1
                            indexOfLeaves = leafConnectionGraph[indexDimension][focalLeafIndex][leafBelowStatusTracker[indexOfLeaves]]
                # If leaf l is unconstrained in all sections, it can be inserted anywhere
                if dd == numberOfDimensions:
                    for indexOfLeaves in range(focalLeafIndex): # prange is inconsequential
                        # if (computationalDivisions == 0 or focalLeafIndex != computationalDivisions or indexOfLeaves % computationalDivisions == computationalIndex)
                        allGaps[gg] = indexOfLeaves
                        gg += 1

                for j in range(g, gg): # prange causes error
                    allGaps[g] = allGaps[j]
                    if count[allGaps[j]] == numberOfDimensions - dd:
                        g += 1
                    count[allGaps[j]] = 0

        # Backtrack if no more gaps
        while focalLeafIndex > 0 and g == gapPointers[focalLeafIndex - 1]:
            focalLeafIndex -= 1
            leafBelowStatusTracker[leafAboveStatusTracker[focalLeafIndex]] = leafBelowStatusTracker[focalLeafIndex]
            leafAboveStatusTracker[leafBelowStatusTracker[focalLeafIndex]] = leafAboveStatusTracker[focalLeafIndex]

        # Insert leaf and advance
        if focalLeafIndex > 0:
            g -= 1
            leafAboveStatusTracker[focalLeafIndex] = allGaps[g]
            leafBelowStatusTracker[focalLeafIndex] = leafBelowStatusTracker[leafAboveStatusTracker[focalLeafIndex]]
            leafBelowStatusTracker[leafAboveStatusTracker[focalLeafIndex]] = focalLeafIndex
            leafAboveStatusTracker[leafBelowStatusTracker[focalLeafIndex]] = focalLeafIndex
            gapPointers[focalLeafIndex] = g
            focalLeafIndex += 1

    return foldingsTotal

