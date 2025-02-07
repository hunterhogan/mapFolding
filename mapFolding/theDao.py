from mapFolding import indexMy, indexTrack
from numpy import integer
from numpy.typing import NDArray
from typing import Any, Tuple
import numba
import numpy

def activeGapIncrement(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.gap1ndex.value] += 1

def activeLeafGreaterThan0Condition(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> Any:
    return my[indexMy.leaf1ndex.value]

def activeLeafGreaterThanLeavesTotalCondition(foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> Any:
    return my[indexMy.leaf1ndex.value] > foldGroups[-1]

def activeLeafIsTheFirstLeafCondition(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> Any:
    return my[indexMy.leaf1ndex.value] <= 1

def allDimensionsAreUnconstrained(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> Any:
    return not my[indexMy.dimensionsUnconstrained.value]

def backtrack(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.leaf1ndex.value] -= 1
    track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]
    track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]

def backtrackCondition(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> Any:
    return my[indexMy.leaf1ndex.value] and my[indexMy.gap1ndex.value] == track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]

def gap1ndexCeilingIncrement(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.gap1ndexCeiling.value] += 1

def countGaps(gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> None:
    gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.leafConnectee.value]
    if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
        gap1ndexCeilingIncrement(my=my)
    track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1

def dimension1ndexIncrement(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.indexDimension.value] += 1

def dimensionsUnconstrainedCondition(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> Any:
    return connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]

def dimensionsUnconstrainedDecrement(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.dimensionsUnconstrained.value] -= 1

def filterCommonGaps(gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> None:
    gapsWhere[my[indexMy.gap1ndex.value]] = gapsWhere[my[indexMy.indexMiniGap.value]]
    if track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] == my[indexMy.dimensionsUnconstrained.value]:
        activeGapIncrement(my=my)
    track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] = 0

def findGapsInitializeVariables(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.dimensionsUnconstrained.value] = my[indexMy.dimensionsTotal.value]
    my[indexMy.gap1ndexCeiling.value] = track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]
    my[indexMy.indexDimension.value] = 0

def indexMiniGapIncrement(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.indexMiniGap.value] += 1

def indexMiniGapInitialization(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]

def insertUnconstrainedLeaf(gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.indexLeaf.value] = 0
    while my[indexMy.indexLeaf.value] < my[indexMy.leaf1ndex.value]:
        gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.indexLeaf.value]
        my[indexMy.gap1ndexCeiling.value] += 1
        my[indexMy.indexLeaf.value] += 1

def leafBelowSentinelIs1Condition(track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> Any:
    return track[indexTrack.leafBelow.value, 0] == 1

def leafConnecteeInitialization(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]]

def leafConnecteeUpdate(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]

def loopingLeavesConnectedToActiveLeaf(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> Any:
    return my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]

def loopingTheDimensions(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> Any:
    return my[indexMy.indexDimension.value] < my[indexMy.dimensionsTotal.value]

def loopingToActiveGapCeiling(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> Any:
    return my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexCeiling.value]

def placeLeaf(gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> None:
    my[indexMy.gap1ndex.value] -= 1
    track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]] = gapsWhere[my[indexMy.gap1ndex.value]]
    track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]] = track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]]
    track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
    track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
    track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value]] = my[indexMy.gap1ndex.value]
    my[indexMy.leaf1ndex.value] += 1

def placeLeafCondition(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> Any:
    return my[indexMy.leaf1ndex.value]

def thereAreComputationDivisionsYouMightSkip(my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]) -> Any:
    return my[indexMy.leaf1ndex.value] != my[indexMy.taskDivisions.value] or my[indexMy.leafConnectee.value] % my[indexMy.taskDivisions.value] == my[indexMy.taskIndex.value]

def countInitialize(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]]
                    , gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                    , my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                    , track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> None:
    while activeLeafGreaterThan0Condition(my=my):
        if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
            findGapsInitializeVariables(my=my, track=track)
            while loopingTheDimensions(my=my):
                if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
                    dimensionsUnconstrainedDecrement(my=my)
                else:
                    leafConnecteeInitialization(connectionGraph=connectionGraph, my=my)
                    while loopingLeavesConnectedToActiveLeaf(my=my):
                        countGaps(gapsWhere=gapsWhere, my=my, track=track)
                        leafConnecteeUpdate(connectionGraph=connectionGraph, my=my, track=track)
                dimension1ndexIncrement(my=my)
            if allDimensionsAreUnconstrained(my=my):
                insertUnconstrainedLeaf(gapsWhere=gapsWhere, my=my)
            indexMiniGapInitialization(my=my)
            while loopingToActiveGapCeiling(my=my):
                filterCommonGaps(gapsWhere=gapsWhere, my=my, track=track)
                indexMiniGapIncrement(my=my)
        if placeLeafCondition(my=my):
            placeLeaf(gapsWhere=gapsWhere, my=my, track=track)
        if my[indexMy.gap1ndex.value] > 0:
            return

def countParallel(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]]
                    , foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                    , gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                    , my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                    , track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> None:
    gapsWherePARALLEL = gapsWhere.copy()
    myPARALLEL = my.copy()
    trackPARALLEL = track.copy()
    taskDivisionsPrange = myPARALLEL[indexMy.taskDivisions.value]
    for indexSherpa in numba.prange(taskDivisionsPrange):
        groupsOfFolds: int = 0
        gapsWhere = gapsWherePARALLEL.copy()
        my = myPARALLEL.copy()
        my[indexMy.taskIndex.value] = indexSherpa
        track = trackPARALLEL.copy()
        while activeLeafGreaterThan0Condition(my=my):
            if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
                if activeLeafGreaterThanLeavesTotalCondition(foldGroups=foldGroups, my=my):
                    groupsOfFolds += 1
                else:
                    findGapsInitializeVariables(my=my, track=track)
                    while loopingTheDimensions(my=my):
                        if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
                            dimensionsUnconstrainedDecrement(my=my)
                        else:
                            leafConnecteeInitialization(connectionGraph=connectionGraph, my=my)
                            while loopingLeavesConnectedToActiveLeaf(my=my):
                                if thereAreComputationDivisionsYouMightSkip(my=my):
                                    countGaps(gapsWhere=gapsWhere, my=my, track=track)
                                leafConnecteeUpdate(connectionGraph=connectionGraph, my=my, track=track)
                        dimension1ndexIncrement(my=my)
                    indexMiniGapInitialization(my=my)
                    while loopingToActiveGapCeiling(my=my):
                        filterCommonGaps(gapsWhere=gapsWhere, my=my, track=track)
                        indexMiniGapIncrement(my=my)
            while backtrackCondition(my=my, track=track):
                backtrack(my=my, track=track)
            if placeLeafCondition(my=my):
                placeLeaf(gapsWhere=gapsWhere, my=my, track=track)
        foldGroups[my[indexMy.taskIndex.value]] = groupsOfFolds

def countSequential(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]], foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]) -> None:
    groupsOfFolds: int = 0
    doFindGaps = True
    while activeLeafGreaterThan0Condition(my=my):
        if ((doFindGaps := activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track))
                and activeLeafGreaterThanLeavesTotalCondition(foldGroups=foldGroups, my=my)):
            groupsOfFolds += 1
        elif doFindGaps:
            findGapsInitializeVariables(my=my, track=track)
            while loopingTheDimensions(my=my):
                if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
                    dimensionsUnconstrainedDecrement(my=my)
                else:
                    leafConnecteeInitialization(connectionGraph=connectionGraph, my=my)
                    while loopingLeavesConnectedToActiveLeaf(my=my):
                        countGaps(gapsWhere=gapsWhere, my=my, track=track)
                        leafConnecteeUpdate(connectionGraph=connectionGraph, my=my, track=track)
                dimension1ndexIncrement(my=my)
            indexMiniGapInitialization(my=my)
            while loopingToActiveGapCeiling(my=my):
                filterCommonGaps(gapsWhere=gapsWhere, my=my, track=track)
                indexMiniGapIncrement(my=my)
        while backtrackCondition(my=my, track=track):
            backtrack(my=my, track=track)
        if placeLeafCondition(my=my):
            placeLeaf(gapsWhere=gapsWhere, my=my, track=track)
    foldGroups[my[indexMy.taskIndex.value]] = groupsOfFolds

def doTheNeedful(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]]
                , foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                , gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                , mapShape: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                , my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                , track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]
                ) -> None:
    countInitialize(connectionGraph, gapsWhere, my, track)

    if my[indexMy.taskDivisions.value] > 0:
        countParallel(connectionGraph, foldGroups, gapsWhere, my, track)
    else:
        countSequential(connectionGraph, foldGroups, gapsWhere, my, track)
