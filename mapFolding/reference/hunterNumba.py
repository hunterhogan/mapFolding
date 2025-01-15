from typing import List
import numba
import numpy

@numba.jit(cache=True, nopython=True, fastmath=True)
def countFolds(listDimensions: List[int]) -> int:
    """
    Count the number of distinct ways to fold a map with at least two positive dimensions.

    Parameters:
        listDimensions: A list of integers representing the dimensions of the map. Error checking and DRY code are impermissible in the numba and jax universes. Validate the list yourself before passing here. There might be some tools for that in this package unless I have become a pyL33t coder.

    Returns:
        foldsTotal: The total number of distinct folds for the given map dimensions.
    """
    def integerSmall(value) -> numpy.uint8:
        return numpy.uint8(value)

    def integerLarge(value) -> numpy.uint64:
        return numpy.uint64(value)

    dtypeDefault = numpy.uint8
    dtypeMaximum = numpy.uint16

    leavesTotal = integerSmall(1)
    for 个 in listDimensions:
        leavesTotal = leavesTotal * integerSmall(个)
    dimensionsTotal = integerSmall(len(listDimensions))

    """How to build a leaf connection graph, also called a "Cartesian Product Decomposition"
    or a "Dimensional Product Mapping", with sentinels:
    Step 1: find the cumulative product of the map's dimensions"""
    cumulativeProduct = numpy.ones(dimensionsTotal + 1, dtype=dtypeDefault)
    for dimension1ndex in range(1, dimensionsTotal + 1):
        cumulativeProduct[dimension1ndex] = cumulativeProduct[dimension1ndex - 1] * listDimensions[dimension1ndex - 1]

    """Step 2: for each dimension, create a coordinate system """
    """coordinateSystem[dimension1ndex, leaf1ndex] holds the dimension1ndex-th coordinate of leaf leaf1ndex"""
    coordinateSystem = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1), dtype=dtypeDefault)
    for dimension1ndex in range(1, dimensionsTotal + 1):
        for leaf1ndex in range(1, leavesTotal + 1):
            coordinateSystem[dimension1ndex, leaf1ndex] = ((leaf1ndex - 1) // cumulativeProduct[dimension1ndex - 1]) % listDimensions[dimension1ndex - 1] + 1

    """Step 3: create a huge empty connection graph"""
    connectionGraph = numpy.zeros((dimensionsTotal + 1, leavesTotal + 1, leavesTotal + 1), dtype=dtypeDefault)

    """Step for... for... for...: fill the connection graph"""
    for dimension1ndex in range(1, dimensionsTotal + 1):
        for activeLeaf1ndex in range(1, leavesTotal + 1):
            for leaf1ndexConnectee in range(1, activeLeaf1ndex + 1):
                connectionGraph[dimension1ndex, activeLeaf1ndex, leaf1ndexConnectee] = (0 if leaf1ndexConnectee == 0
                                else ((leaf1ndexConnectee if coordinateSystem[dimension1ndex, leaf1ndexConnectee] == 1
                                            else leaf1ndexConnectee - cumulativeProduct[dimension1ndex - 1])
                                    if (coordinateSystem[dimension1ndex, activeLeaf1ndex] & 1) == (coordinateSystem[dimension1ndex, leaf1ndexConnectee] & 1)
                                    else (leaf1ndexConnectee if coordinateSystem[dimension1ndex, leaf1ndexConnectee] == listDimensions[dimension1ndex-1]
                                            or leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1] > activeLeaf1ndex
                                            else leaf1ndexConnectee + cumulativeProduct[dimension1ndex - 1])))

    """Indices of array `track` (to "track" the execution state), which is a collection of one-dimensional arrays each of length `leavesTotal + 1`."""
    leafAbove = numba.literally(0)
    leafBelow = numba.literally(1)
    countDimensionsGapped = numba.literally(2)
    gapRangeStart = numba.literally(3)
    track = numpy.zeros((4, leavesTotal + 1), dtype=dtypeDefault)

    gapsWhere = numpy.zeros(integerLarge(integerLarge(leavesTotal) * integerLarge(leavesTotal) + 1), dtype=dtypeMaximum)

    foldsTotal = integerLarge(0)
    activeLeaf1ndex = integerSmall(1)
    activeGap1ndex = integerSmall(0)

    while activeLeaf1ndex > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
            if activeLeaf1ndex > leavesTotal:
                foldsTotal += leavesTotal
            else:
                dimensionsUnconstrained = integerSmall(0)
                """Track possible gaps for activeLeaf1ndex in each section"""
                gap1ndexCeiling = track[gapRangeStart, activeLeaf1ndex - 1]

                """Count possible gaps for activeLeaf1ndex in each section"""
                dimension1ndex = integerSmall(1)
                while dimension1ndex <= dimensionsTotal:
                    if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                        dimensionsUnconstrained += 1
                    else:
                        leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                        while leaf1ndexConnectee != activeLeaf1ndex:
                            gapsWhere[gap1ndexCeiling] = leaf1ndexConnectee
                            if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
                                gap1ndexCeiling += 1
                            track[countDimensionsGapped, leaf1ndexConnectee] += 1
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                    dimension1ndex += 1

                """If activeLeaf1ndex is unconstrained in all sections, it can be inserted anywhere"""
                if dimensionsUnconstrained == dimensionsTotal:
                    leaf1ndex = integerSmall(0)
                    while leaf1ndex < activeLeaf1ndex:
                        gapsWhere[gap1ndexCeiling] = leaf1ndex
                        gap1ndexCeiling += 1
                        leaf1ndex += 1

                """Filter gaps that are common to all sections"""
                indexMiniGap = activeGap1ndex
                while indexMiniGap < gap1ndexCeiling:
                    gapsWhere[activeGap1ndex] = gapsWhere[indexMiniGap]
                    if track[countDimensionsGapped, gapsWhere[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                        activeGap1ndex += 1
                    """Reset track[countDimensionsGapped] for next iteration"""
                    track[countDimensionsGapped, gapsWhere[indexMiniGap]] = 0
                    indexMiniGap += 1

        """Recursive backtracking steps"""
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
            activeLeaf1ndex -= 1
            track[leafBelow, track[leafAbove, activeLeaf1ndex]] = track[leafBelow, activeLeaf1ndex]
            track[leafAbove, track[leafBelow, activeLeaf1ndex]] = track[leafAbove, activeLeaf1ndex]

        """Place leaf in valid position"""
        if activeLeaf1ndex > 0:
            activeGap1ndex -= 1
            track[leafAbove, activeLeaf1ndex] = gapsWhere[activeGap1ndex]
            track[leafBelow, activeLeaf1ndex] = track[leafBelow, track[leafAbove, activeLeaf1ndex]]
            track[leafBelow, track[leafAbove, activeLeaf1ndex]] = activeLeaf1ndex
            track[leafAbove, track[leafBelow, activeLeaf1ndex]] = activeLeaf1ndex
            """Save current gap index"""
            track[gapRangeStart, activeLeaf1ndex] = activeGap1ndex
            """Move to next leaf"""
            activeLeaf1ndex += 1

    return int(foldsTotal)
