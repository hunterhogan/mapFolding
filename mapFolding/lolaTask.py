"""A functional but untenable implementation of the Run Lola Run concept. Untenable because of excessive code duplication."""
from mapFolding import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart
import numba
import numpy

@numba.jit(nopython=True, cache=True, fastmath=True)
def doWhileTask(activeGap1ndex: numpy.uint8,
    activeLeaf1ndex: numpy.uint8,
    connectionGraph: numpy.ndarray,
    dimensionsTotal: numpy.uint8,
    leavesTotal: numpy.uint8,
    gaps: numpy.ndarray,
    taskIndex: numpy.uint8,
    s: numpy.ndarray):
    """Compute one task with values passed to the function.
    `leavesTotal: numpy.uint8` is a limitation: be cautious, especially [2,2,2,2,2,2,2,2]"""

    potentialGaps = gaps.copy()
    track = s.copy()

    foldsSubTotal = numpy.uint64(0)
    while activeLeaf1ndex > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
            if activeLeaf1ndex > leavesTotal:
                foldsSubTotal += numpy.uint64(leavesTotal)
            else:
                dimensionsUnconstrained = numpy.uint8(0)
                gap1ndexLowerBound: numpy.uint8 = track[gapRangeStart, activeLeaf1ndex - 1]
                dimension1ndex = numpy.uint8(1)
                while dimension1ndex <= dimensionsTotal:
                    if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                        dimensionsUnconstrained += 1
                    else:
                        leaf1ndexConnectee: numpy.uint8 = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                        while leaf1ndexConnectee != activeLeaf1ndex:
                            if activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
                                potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                                if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
                                    gap1ndexLowerBound += 1
                                track[countDimensionsGapped, leaf1ndexConnectee] += 1
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                    dimension1ndex += 1
                indexMiniGap: numpy.uint8 = activeGap1ndex
                while indexMiniGap < gap1ndexLowerBound:
                    potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                    if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                        activeGap1ndex += 1
                    track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
                    indexMiniGap += 1
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
            activeLeaf1ndex -= 1
            track[leafBelow, track[leafAbove, activeLeaf1ndex]] = track[leafBelow, activeLeaf1ndex]
            track[leafAbove, track[leafBelow, activeLeaf1ndex]] = track[leafAbove, activeLeaf1ndex]
        if activeLeaf1ndex > 0:
            activeGap1ndex -= 1
            track[leafAbove, activeLeaf1ndex] = potentialGaps[activeGap1ndex]
            track[leafBelow, activeLeaf1ndex] = track[leafBelow, track[leafAbove, activeLeaf1ndex]]
            track[leafBelow, track[leafAbove, activeLeaf1ndex]] = activeLeaf1ndex
            track[leafAbove, track[leafBelow, activeLeaf1ndex]] = activeLeaf1ndex
            track[gapRangeStart, activeLeaf1ndex] = activeGap1ndex
            activeLeaf1ndex += 1
    return foldsSubTotal
