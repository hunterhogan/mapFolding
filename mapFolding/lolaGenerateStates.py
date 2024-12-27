from numba import njit
import copy
from typing import TypedDict, Dict
import numpy

from mapFolding.lolaIndices import leavesTotal, dimensionsTotal, dimensionsPlus1
from mapFolding.lovelaceIndices import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart
from mapFolding.lolaIndices import (
    activeGap1ndex as activeGap1ndexMyIndex,
    activeLeaf1ndex as activeLeaf1ndexMyIndex, 
    dimension1ndex as dimension1ndexMyIndex,
    foldingsSubtotal,
    gap1ndexLowerBound as gap1ndexLowerBoundMyIndex,
    leaf1ndexConnectee as leaf1ndexConnecteeMyIndex,
    taskIndex,
    unconstrainedLeaf as unconstrainedLeafMyIndex,
                                    )

class TaskState(TypedDict):
    track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]
    potentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]
    my: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]

# @njit(cache=True, fastmath=False)
def generateStates(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    potentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    my: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    the: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
                    connectionGraph: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]):

    dictionaryStates: Dict[int, TaskState] = {}
    foldingsTotal: int = 0
    activeLeaf1ndex: int = 1
    activeGap1ndex: int = 0

    while True:
        if activeLeaf1ndex <= 1 or track[leafBelow][0] == 1:
            if activeLeaf1ndex > the[leavesTotal]:
                foldingsTotal += the[leavesTotal]
            else:
                unconstrainedLeaf: int = 0
                # Track possible gaps
                gap1ndexLowerBound: int = track[gapRangeStart][activeLeaf1ndex - 1]
                # Reset gap index
                activeGap1ndex = gap1ndexLowerBound

                # Count possible gaps for leaf l in each section
                for dimension1ndex in range(1, the[dimensionsPlus1]):
                    if connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex:
                        unconstrainedLeaf += 1
                    else:
                        leaf1ndexConnectee: int = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
                        while leaf1ndexConnectee != activeLeaf1ndex:

                            if activeLeaf1ndex != the[leavesTotal]:
                                pass
                            else:
                                modulo = leaf1ndexConnectee % the[leavesTotal]
                                if modulo not in dictionaryStates:
                                    my[activeGap1ndexMyIndex] = copy.copy(activeGap1ndex)
                                    my[activeLeaf1ndexMyIndex] = copy.copy(activeLeaf1ndex)
                                    my[dimension1ndexMyIndex] = copy.copy(dimension1ndex)
                                    my[gap1ndexLowerBoundMyIndex] = copy.copy(gap1ndexLowerBound)
                                    my[leaf1ndexConnecteeMyIndex] = copy.copy(leaf1ndexConnectee)
                                    my[unconstrainedLeafMyIndex] = copy.copy(unconstrainedLeaf)
                                    my[taskIndex] = copy.copy(modulo)
                                    my[foldingsSubtotal] = 0

                                    taskState: TaskState = {'track': track.copy(), 'potentialGaps': potentialGaps.copy(), 'my': my.copy()}
                                    dictionaryStates[my[taskIndex]] = taskState

                                    if len(dictionaryStates) >= the[leavesTotal]:
                                        dictionaryStates[my[taskIndex]]['my'][foldingsSubtotal] = foldingsTotal
                                        return dictionaryStates

                            potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                            if track[countDimensionsGapped][leaf1ndexConnectee] == 0:
                                gap1ndexLowerBound += 1
                            track[countDimensionsGapped][leaf1ndexConnectee] += 1
                            leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][track[leafBelow][leaf1ndexConnectee]]

                # If leaf l is unconstrained in all sections, it can be inserted anywhere
                if unconstrainedLeaf == the[dimensionsTotal]:
                    for leaf1ndex in range(activeLeaf1ndex):
                        potentialGaps[gap1ndexLowerBound] = leaf1ndex
                        gap1ndexLowerBound += 1

                # Filter gaps that are common to all sections
                for indexMiniGap in range(activeGap1ndex, gap1ndexLowerBound):
                    potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                    if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == the[dimensionsTotal] - unconstrainedLeaf:
                        activeGap1ndex += 1
                    # Reset track[count] for next iteration
                    track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0

        # Recursive backtracking steps
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart][activeLeaf1ndex - 1]:
            activeLeaf1ndex -= 1
            track[leafBelow][track[leafAbove][activeLeaf1ndex]] = track[leafBelow][activeLeaf1ndex]
            track[leafAbove][track[leafBelow][activeLeaf1ndex]] = track[leafAbove][activeLeaf1ndex]

        # Place leaf in valid position
        if activeLeaf1ndex > 0:
            activeGap1ndex -= 1
            track[leafAbove][activeLeaf1ndex] = potentialGaps[activeGap1ndex]
            track[leafBelow][activeLeaf1ndex] = track[leafBelow][track[leafAbove][activeLeaf1ndex]]
            track[leafBelow][track[leafAbove][activeLeaf1ndex]] = activeLeaf1ndex
            track[leafAbove][track[leafBelow][activeLeaf1ndex]] = activeLeaf1ndex
            # Save current gap index
            track[gapRangeStart][activeLeaf1ndex] = activeGap1ndex
            # Move to next leaf
            activeLeaf1ndex += 1
