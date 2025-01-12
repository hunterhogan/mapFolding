from mapFolding import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart
import numba
import numba.extending
import numpy
from typing import Optional

# TODO learn how to dynamically set the integer sizes instead of hardcoding them into type hints and numba types

@numba.jit(nopython=True, cache=True, fastmath=True)
def ifComputationDivisions(taskDivisions: numpy.uint8, activeLeaf1ndex: numpy.uint8, leaf1ndexConnectee: numpy.uint8, taskIndex: Optional[numpy.uint8]) -> bool:
    """This function (allegedly) allows numba to compile two different versions based on the value of `taskDivisions`. The benefit is one less conditional check in the main loop, which is important. As I write this, I have a CPU process that I estimate is about half way through counting folds: "Total Time: 20:35:45.593; Cycles: 135,149,788,103,967", so removing a statement that might account for only .1% of 135 trillion cycles, would save 135 billion cycles. That's a lot of cycles.
    """
    """
    `taskIndex: Optional[numpy.uint8] = numpy.uint8(0)` vs `taskIndex: Optional[numpy.uint8] ...` no default value
    If taskIndex should have an int value, but for some reason it doesn't, I want the code to fail fast.
    """
    if not taskDivisions: # If `taskDivisions` is 0, return True. Or, if for some reason, `taskDivisions` is None or False, return True.
        return True
    return activeLeaf1ndex != taskDivisions or leaf1ndexConnectee % taskDivisions == taskIndex

@numba.jit(nopython=True, cache=True, fastmath=True, parallel=True)
def doWhileConcurrent(
        activeGap1ndex: numpy.uint8,
        activeLeaf1ndex: numpy.uint8,
        connectionGraph: numpy.ndarray,
        dimensionsTotal: numpy.uint8,
        leavesTotal: numpy.uint8,
        potentialGaps: numpy.ndarray,
        track: numpy.ndarray,
        taskDivisions: numpy.uint8,
        ):

    foldsRunningTotal = numpy.uint64(0)

    for taskIndex in numba.prange(taskDivisions):
        foldsRunningTotal = foldsRunningTotal + int(
            doWhile(
                activeGap1ndex,
                activeLeaf1ndex,
                connectionGraph,
                dimensionsTotal,
                leavesTotal,
                potentialGaps.copy(),
                track.copy(),
                taskDivisions,
                numpy.uint8(taskIndex),
                )
            )

    return foldsRunningTotal

@numba.jit(nopython=True, cache=True, fastmath=True)
def doWhile(
    activeGap1ndex: numpy.uint8,
    activeLeaf1ndex: numpy.uint8,
    connectionGraph: numpy.ndarray,
    dimensionsTotal: numpy.uint8,
    leavesTotal: numpy.uint8,
    potentialGaps: numpy.ndarray,
    track: numpy.ndarray,
    taskDivisions: numpy.uint8 = numpy.uint8(0),
    taskIndex: Optional[numpy.uint8] = None,
    ):
    """Compute the full job with values passed to the function.
    If taskIndex is None, computes all tasks.
    If taskIndex has a value, computes only that task.
    `leavesTotal: numpy.uint8` is a limitation: be cautious, especially [2,2,2,2,2,2,2,2]"""

    def updateLinkedTrackParallel(track: numpy.ndarray, activeLeaf1ndex: numpy.uint8) -> None:
        # Pre-fetch values to avoid multiple array accesses
        valueAbove = track[leafAbove, activeLeaf1ndex]
        valueBelow = track[leafBelow, activeLeaf1ndex]
        trackLeafBelow = track[leafBelow, activeLeaf1ndex]
        trackLeafAbove = track[leafAbove, activeLeaf1ndex]

        # Perform updates
        track[leafBelow, valueAbove] = trackLeafBelow
        track[leafAbove, valueBelow] = trackLeafAbove

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
                            if ifComputationDivisions(taskDivisions, activeLeaf1ndex, leaf1ndexConnectee, taskIndex):
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
