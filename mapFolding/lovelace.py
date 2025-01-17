from mapFolding import indexMy, indexThe, indexTrack
from numpy import integer
from numpy.typing import NDArray
from typing import Any
import numba
import numpy

def activeGapIncrement(my: NDArray[integer[Any]]):
    my[indexMy.gap1ndex.value] += 1

def activeLeafGreaterThan0Condition(my: NDArray[integer[Any]]):
    return my[indexMy.leaf1ndex.value] > 0

def activeLeafGreaterThanLeavesTotalCondition(my: NDArray[integer[Any]], the: NDArray[integer[Any]]):
    return my[indexMy.leaf1ndex.value] > the[indexThe.leavesTotal.value]

def activeLeafIsTheFirstLeafCondition(my: NDArray[integer[Any]]):
    return my[indexMy.leaf1ndex.value] <= 1

def activeLeafNotEqualToTaskDivisionsCondition(my: NDArray[integer[Any]], the: NDArray[integer[Any]]):
    return my[indexMy.leaf1ndex.value] != the[indexThe.taskDivisions.value]

def allDimensionsAreUnconstrained(my: NDArray[integer[Any]], the: NDArray[integer[Any]]):
    return my[indexMy.dimensionsUnconstrained.value] == the[indexThe.dimensionsTotal.value]

def backtrack(my: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    my[indexMy.leaf1ndex.value] -= 1
    track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]
    track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]

def backtrackCondition(my: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    return my[indexMy.leaf1ndex.value] > 0 and my[indexMy.gap1ndex.value] == track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]

def countGaps(gapsWhere: NDArray[integer[Any]], my: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.leafConnectee.value]
    if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
        gap1ndexCeilingIncrement(my=my)
    track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1

def dimension1ndexIncrement(my: NDArray[integer[Any]]):
    my[indexMy.dimension1ndex.value] += 1

def dimensionsUnconstrainedCondition(connectionGraph: NDArray[integer[Any]], my: NDArray[integer[Any]]):
    return connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]

def dimensionsUnconstrainedIncrement(my: NDArray[integer[Any]]):
    my[indexMy.dimensionsUnconstrained.value] += 1

def filterCommonGaps(gapsWhere: NDArray[integer[Any]], my: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    gapsWhere[my[indexMy.gap1ndex.value]] = gapsWhere[my[indexMy.indexMiniGap.value]]
    if track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] == the[indexThe.dimensionsTotal.value] - my[indexMy.dimensionsUnconstrained.value]:
        activeGapIncrement(my=my)
    track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] = 0

def findGapsInitializeVariables(my: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    my[indexMy.dimensionsUnconstrained.value] = 0
    my[indexMy.gap1ndexCeiling.value] = track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]
    my[indexMy.dimension1ndex.value] = 1

def foldsSubTotalIncrement(foldsSubTotals: NDArray[integer[Any]], my: NDArray[integer[Any]], the: NDArray[integer[Any]]):
    foldsSubTotals[my[indexMy.taskIndex.value]] += the[indexThe.leavesTotal.value]

def gap1ndexCeilingIncrement(my: NDArray[integer[Any]]):
    my[indexMy.gap1ndexCeiling.value] += 1

def indexMiniGapIncrement(my: NDArray[integer[Any]]):
    my[indexMy.indexMiniGap.value] += 1

def indexMiniGapInitialization(my: NDArray[integer[Any]]):
    my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]

def insertUnconstrainedLeaf(gapsWhere: NDArray[integer[Any]], my: NDArray[integer[Any]]):
    my[indexMy.indexLeaf.value] = 0
    while my[indexMy.indexLeaf.value] < my[indexMy.leaf1ndex.value]:
        gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.indexLeaf.value]
        my[indexMy.gap1ndexCeiling.value] += 1
        my[indexMy.indexLeaf.value] += 1

def leafBelowSentinelIs1Condition(track: NDArray[integer[Any]]):
    return track[indexTrack.leafBelow.value, 0] == 1

def leafConnecteeInitialization(connectionGraph: NDArray[integer[Any]], my: NDArray[integer[Any]]):
    my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]]

def leafConnecteeUpdate(connectionGraph: NDArray[integer[Any]], my: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]

def loopingLeavesConnectedToActiveLeaf(my: NDArray[integer[Any]]):
    return my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]

def loopingTheDimensions(my: NDArray[integer[Any]], the: NDArray[integer[Any]]):
    return my[indexMy.dimension1ndex.value] <= the[indexThe.dimensionsTotal.value]

def loopingToActiveGapCeiling(my: NDArray[integer[Any]]):
    return my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexCeiling.value]

def placeLeaf(gapsWhere: NDArray[integer[Any]], my: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    my[indexMy.gap1ndex.value] -= 1
    track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]] = gapsWhere[my[indexMy.gap1ndex.value]]
    track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]] = track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]]
    track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
    track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
    track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value]] = my[indexMy.gap1ndex.value]
    my[indexMy.leaf1ndex.value] += 1

def placeLeafCondition(my: NDArray[integer[Any]]):
    return my[indexMy.leaf1ndex.value] > 0

def taskIndexCondition(my: NDArray[integer[Any]], the: NDArray[integer[Any]]):
    return my[indexMy.leafConnectee.value] % the[indexThe.taskDivisions.value] == my[indexMy.taskIndex.value]

def thereAreComputationDivisionsYouMightSkip(my: NDArray[integer[Any]], the: NDArray[integer[Any]]):
    if activeLeafNotEqualToTaskDivisionsCondition(my=my, the=the):
        return True
    if taskIndexCondition(my=my, the=the):
        return True
    return False

def initialize(connectionGraph: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], my: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    while activeLeafGreaterThan0Condition(my=my):
        if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
            findGapsInitializeVariables(my=my, track=track)
            while loopingTheDimensions(my=my, the=the):
                if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
                    dimensionsUnconstrainedIncrement(my=my)
                else:
                    leafConnecteeInitialization(connectionGraph=connectionGraph, my=my)
                    while loopingLeavesConnectedToActiveLeaf(my=my):
                        countGaps(gapsWhere=gapsWhere, my=my, track=track)
                        leafConnecteeUpdate(connectionGraph=connectionGraph, my=my, track=track)
                dimension1ndexIncrement(my=my)
            if allDimensionsAreUnconstrained(my=my, the=the):
                insertUnconstrainedLeaf(gapsWhere=gapsWhere, my=my)
            indexMiniGapInitialization(my=my)
            while loopingToActiveGapCeiling(my=my):
                filterCommonGaps(gapsWhere=gapsWhere, my=my, the=the, track=track)
                indexMiniGapIncrement(my=my)
        if placeLeafCondition(my=my):
            placeLeaf(gapsWhere=gapsWhere, my=my, track=track)
        if my[indexMy.gap1ndex.value] > 0:
            break

def countParallel(connectionGraph: NDArray[integer[Any]], foldsSubTotals: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], my: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    while activeLeafGreaterThan0Condition(my=my):
        if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
            if activeLeafGreaterThanLeavesTotalCondition(my=my, the=the):
                foldsSubTotalIncrement(foldsSubTotals=foldsSubTotals, my=my, the=the)
            else:
                findGapsInitializeVariables(my=my, track=track)
                while loopingTheDimensions(my=my, the=the):
                    if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
                        dimensionsUnconstrainedIncrement(my=my)
                    else:
                        leafConnecteeInitialization(connectionGraph=connectionGraph, my=my)
                        while loopingLeavesConnectedToActiveLeaf(my=my):
                            if thereAreComputationDivisionsYouMightSkip(my=my, the=the):
                                countGaps(gapsWhere=gapsWhere, my=my, track=track)
                            leafConnecteeUpdate(connectionGraph=connectionGraph, my=my, track=track)
                    dimension1ndexIncrement(my=my)
                indexMiniGapInitialization(my=my)
                while loopingToActiveGapCeiling(my=my):
                    filterCommonGaps(gapsWhere=gapsWhere, my=my, the=the, track=track)
                    indexMiniGapIncrement(my=my)
        while backtrackCondition(my=my, track=track):
            backtrack(my=my, track=track)
        if placeLeafCondition(my=my):
            placeLeaf(gapsWhere=gapsWhere, my=my, track=track)

def countSequential(connectionGraph: NDArray[integer[Any]], foldsSubTotals: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], my: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]):
    while activeLeafGreaterThan0Condition(my=my):
        if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
            if activeLeafGreaterThanLeavesTotalCondition(my=my, the=the):
                foldsSubTotalIncrement(foldsSubTotals=foldsSubTotals, my=my, the=the)
            else:
                findGapsInitializeVariables(my=my, track=track)
                while loopingTheDimensions(my=my, the=the):
                    if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
                        dimensionsUnconstrainedIncrement(my=my)
                    else:
                        leafConnecteeInitialization(connectionGraph=connectionGraph, my=my)
                        while loopingLeavesConnectedToActiveLeaf(my=my):
                            countGaps(gapsWhere=gapsWhere, my=my, track=track)
                            leafConnecteeUpdate(connectionGraph=connectionGraph, my=my, track=track)
                    dimension1ndexIncrement(my=my)
                indexMiniGapInitialization(my=my)
                while loopingToActiveGapCeiling(my=my):
                    filterCommonGaps(gapsWhere=gapsWhere, my=my, the=the, track=track)
                    indexMiniGapIncrement(my=my)
        while backtrackCondition(my=my, track=track):
            backtrack(my=my, track=track)
        if placeLeafCondition(my=my):
            placeLeaf(gapsWhere=gapsWhere, my=my, track=track)

@numba.jit(parallel=True, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def doTaskIndices(connectionGraph: NDArray[integer[Any]], foldsSubTotals: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], my: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]):

    stateGapsWhere = gapsWhere.copy()
    stateMy = my.copy()
    stateTrack = track.copy()

    for indexSherpa in numba.prange(the[indexThe.taskDivisions.value]):
        mySherpa = stateMy.copy()
        mySherpa[indexMy.taskIndex.value] = indexSherpa
        countParallel(connectionGraph=connectionGraph, foldsSubTotals=foldsSubTotals, gapsWhere=stateGapsWhere.copy(), my=mySherpa, the=the, track=stateTrack.copy())

    return foldsSubTotals

def countFoldsCompiled(connectionGraph: NDArray[integer[Any]], foldsSubTotals: NDArray[integer[Any]], gapsWhere: NDArray[integer[Any]], my: NDArray[integer[Any]], the: NDArray[integer[Any]], track: NDArray[integer[Any]]):

    initialize(connectionGraph=connectionGraph, gapsWhere=gapsWhere, my=my, the=the, track=track)

    if the[indexThe.taskDivisions.value] > 0:
        doTaskIndices(connectionGraph=connectionGraph, foldsSubTotals=foldsSubTotals, gapsWhere=gapsWhere, my=my, the=the, track=track)
    else:
        countSequential(connectionGraph=connectionGraph, foldsSubTotals=foldsSubTotals, gapsWhere=gapsWhere, my=my, the=the, track=track)

numba.jit_module(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
