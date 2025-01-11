"""A functional but untenable implementation of the Run Lola Run concept. Untenable because of excessive code duplication."""
from mapFolding import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart
import numba
import numpy

@numba.jit(nopython=True, cache=True, fastmath=True)
def updateLinkedTrackParallel(track: numpy.ndarray, activeLeaf1ndex: numpy.uint8) -> None:
    # Pre-fetch values to avoid multiple array accesses
    valueAbove = track[leafAbove, activeLeaf1ndex]
    valueBelow = track[leafBelow, activeLeaf1ndex]
    trackLeafBelow = track[leafBelow, activeLeaf1ndex]
    trackLeafAbove = track[leafAbove, activeLeaf1ndex]

    # Perform updates
    track[leafBelow, valueAbove] = trackLeafBelow
    track[leafAbove, valueBelow] = trackLeafAbove

@numba.jit(nopython=True, cache=True, fastmath=True)
def updateLeafConnections(track: numpy.ndarray, activeLeaf1ndex: numpy.uint8, potentialGaps: numpy.ndarray, activeGap1ndex: numpy.uint8) -> None:
    # Pre-fetch gap value
    gapValue = potentialGaps[activeGap1ndex]
    # Update above connections
    track[leafAbove, activeLeaf1ndex] = gapValue
    # Pre-fetch below connection
    gapBelow = track[leafBelow, gapValue]
    # Update remaining connections
    track[leafBelow, activeLeaf1ndex] = gapBelow
    track[leafBelow, gapValue] = activeLeaf1ndex
    track[leafAbove, gapBelow] = activeLeaf1ndex

@numba.jit(nopython=True, cache=True, fastmath=True)
def doWhileOne(
    activeGap1ndex: numpy.uint8,
    activeLeaf1ndex: numpy.uint8,
    connectionGraph: numpy.ndarray,
    dimensionsTotal: numpy.uint8,
    leavesTotal: numpy.uint8,
    potentialGaps: numpy.ndarray,
    track: numpy.ndarray):
    """Compute the full job with values passed to the function.
    `leavesTotal: numpy.uint8` is a limitation: be cautious, especially [2,2,2,2,2,2,2,2]"""

    dimensionsUnconstrained = numpy.uint8(0)
    dimension1ndex = numpy.uint8(1)
    gap1ndexLowerBound = numpy.uint8(0)
    leaf1ndexConnectee = numpy.uint8(0)
    indexMiniGap = numpy.uint8(0)

    foldsTotal = numpy.uint64(0)

    while activeLeaf1ndex > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
            if activeLeaf1ndex > leavesTotal:
                foldsTotal = foldsTotal + leavesTotal
            else:
                dimensionsUnconstrained = 0
                gap1ndexLowerBound = track[gapRangeStart, activeLeaf1ndex - 1]
                dimension1ndex = 1
                while dimension1ndex <= dimensionsTotal:
                    if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                        dimensionsUnconstrained = dimensionsUnconstrained + 1
                    else:
                        leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                        while leaf1ndexConnectee != activeLeaf1ndex:
                            potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                            if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
                                gap1ndexLowerBound = gap1ndexLowerBound + 1
                            track[countDimensionsGapped, leaf1ndexConnectee] = track[countDimensionsGapped, leaf1ndexConnectee] + 1
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                    dimension1ndex = dimension1ndex + 1
                indexMiniGap = activeGap1ndex
                while indexMiniGap < gap1ndexLowerBound:
                    potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                    if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                        activeGap1ndex = activeGap1ndex + 1
                    track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
                    indexMiniGap = indexMiniGap + 1
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
            activeLeaf1ndex = activeLeaf1ndex - 1
            updateLinkedTrackParallel(track, activeLeaf1ndex)
        if activeLeaf1ndex > 0:
            activeGap1ndex = activeGap1ndex - 1
            updateLeafConnections(track, activeLeaf1ndex, potentialGaps, activeGap1ndex)
            track[gapRangeStart, activeLeaf1ndex] = activeGap1ndex
            activeLeaf1ndex = activeLeaf1ndex + 1
    return foldsTotal
