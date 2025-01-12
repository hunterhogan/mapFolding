from mapFolding import t
from mapFolding import activeGap1ndex, activeLeaf1ndex, dimension1ndex, dimensionsUnconstrained, gap1ndexLowerBound, indexMiniGap, leaf1ndexConnectee
import numba
import numba.extending
import numpy
from typing import Optional

# TODO learn how to dynamically set the integer sizes instead of hardcoding them into type hints and numba types

@numba.jit(nopython=True, cache=True, fastmath=True)
def ifComputationDivisions(taskDivisions: numpy.uint8, my: numpy.ndarray, taskIndex: Optional[numpy.uint8]) -> bool:
    """This function (allegedly) allows numba to compile two different versions based on the value of `taskDivisions`. The benefit is one less conditional check in the main loop, which is important. As I write this, I have a CPU process that I estimate is about half way through counting folds: "Total Time: 20:35:45.593; Cycles: 135,149,788,103,967", so removing a statement that might account for only .1% of 135 trillion cycles, would save 135 billion cycles. That's a lot of cycles.
    """
    """
    `taskIndex: Optional[numpy.uint8] = numpy.uint8(0)` vs `taskIndex: Optional[numpy.uint8] ...` no default value
    If taskIndex should have an int value, but for some reason it doesn't, I want the code to fail fast.
    """
    if not taskDivisions: # If `taskDivisions` is 0, return True. Or, if for some reason, `taskDivisions` is None or False, return True.
        return True
    return my[activeLeaf1ndex] != taskDivisions or my[leaf1ndexConnectee] % taskDivisions == taskIndex

@numba.jit(nopython=True, cache=True, fastmath=True, parallel=True)
def doWhileConcurrent(
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

    my = track[t.my.value]
    foldsTotal = numpy.uint64(0)

    while my[activeLeaf1ndex] > 0:
        if my[activeLeaf1ndex] <= 1 or track[t.leafBelow.value, 0] == 1:
            if my[activeLeaf1ndex] > leavesTotal:
                foldsTotal = foldsTotal + leavesTotal
            else:
                my[dimensionsUnconstrained] = 0
                my[gap1ndexLowerBound] = track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]
                my[dimension1ndex] = 1
                while my[dimension1ndex] <= dimensionsTotal:
                    if connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
                        my[dimensionsUnconstrained] = my[dimensionsUnconstrained] + 1
                    else:
                        my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]]
                        while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                            if ifComputationDivisions(taskDivisions, my, taskIndex):
                                potentialGaps[my[gap1ndexLowerBound]] = my[leaf1ndexConnectee]
                                if track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] == 0:
                                    my[gap1ndexLowerBound] = my[gap1ndexLowerBound] + 1
                                track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] = track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] + 1
                            my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], track[t.leafBelow.value, my[leaf1ndexConnectee]]]
                    my[dimension1ndex] = my[dimension1ndex] + 1
                my[indexMiniGap] = my[activeGap1ndex]
                while my[indexMiniGap] < my[gap1ndexLowerBound]:
                    potentialGaps[my[activeGap1ndex]] = potentialGaps[my[indexMiniGap]]
                    if track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] == dimensionsTotal - my[dimensionsUnconstrained]:
                        my[activeGap1ndex] = my[activeGap1ndex] + 1
                    track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] = 0
                    my[indexMiniGap] = my[indexMiniGap] + 1
        while my[activeLeaf1ndex] > 0 and my[activeGap1ndex] == track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]:
            my[activeLeaf1ndex] = my[activeLeaf1ndex] - 1
            valueAbove = track[t.leafAbove.value, my[activeLeaf1ndex]]
            valueBelow = track[t.leafBelow.value, my[activeLeaf1ndex]]
            trackLeafBelow = track[t.leafBelow.value, my[activeLeaf1ndex]]
            trackLeafAbove = track[t.leafAbove.value, my[activeLeaf1ndex]]

            # Perform updates
            track[t.leafBelow.value, valueAbove] = trackLeafBelow
            track[t.leafAbove.value, valueBelow] = trackLeafAbove
        if my[activeLeaf1ndex] > 0:
            my[activeGap1ndex] = my[activeGap1ndex] - 1
            # Pre-fetch gap value
            gapValue = potentialGaps[my[activeGap1ndex]]
            # Update above connections
            track[t.leafAbove.value, my[activeLeaf1ndex]] = gapValue
            # Pre-fetch below connection
            gapBelow = track[t.leafBelow.value, gapValue]
            # Update remaining connections
            track[t.leafBelow.value, my[activeLeaf1ndex]] = gapBelow
            track[t.leafBelow.value, gapValue] = my[activeLeaf1ndex]
            track[t.leafAbove.value, gapBelow] = my[activeLeaf1ndex]
            track[t.gapRangeStart.value, my[activeLeaf1ndex]] = my[activeGap1ndex]
            my[activeLeaf1ndex] = my[activeLeaf1ndex] + 1
    return foldsTotal
