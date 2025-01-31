from mapFolding import indexMy, indexTrack
from typing import Any, Tuple
from numpy import integer
import numpy
import numba

def activeGapIncrement(my):
    my[indexMy.gap1ndex.value] += 1

def activeLeafGreaterThan0Condition(my):
    return my[indexMy.leaf1ndex.value] > 0

def activeLeafGreaterThanLeavesTotalCondition(foldGroups, my):
    return my[indexMy.leaf1ndex.value] > foldGroups[-1]

def activeLeafIsTheFirstLeafCondition(my):
    return my[indexMy.leaf1ndex.value] <= 1

def allDimensionsAreUnconstrained(my):
    return not my[indexMy.dimensionsUnconstrained.value]

def backtrack(my, track):
    my[indexMy.leaf1ndex.value] -= 1
    track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]
    track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]

def backtrackCondition(my, track):
    return my[indexMy.leaf1ndex.value] > 0 and my[indexMy.gap1ndex.value] == track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]

def gap1ndexCeilingIncrement(my):
    my[indexMy.gap1ndexCeiling.value] += 1

def countGaps(gapsWhere, my, track):
    gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.leafConnectee.value]
    if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
        gap1ndexCeilingIncrement(my=my)
    track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1

def dimension1ndexIncrement(my):
    my[indexMy.indexDimension.value] += 1

def dimensionsUnconstrainedCondition(connectionGraph, my):
    return connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]

def dimensionsUnconstrainedDecrement(my):
    my[indexMy.dimensionsUnconstrained.value] -= 1

def filterCommonGaps(gapsWhere, my, track):
    gapsWhere[my[indexMy.gap1ndex.value]] = gapsWhere[my[indexMy.indexMiniGap.value]]
    if track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] == my[indexMy.dimensionsUnconstrained.value]:
        activeGapIncrement(my=my)
    track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] = 0

def findGapsInitializeVariables(my, track):
    my[indexMy.dimensionsUnconstrained.value] = my[indexMy.dimensionsTotal.value]
    my[indexMy.gap1ndexCeiling.value] = track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]
    my[indexMy.indexDimension.value] = 0

def foldsSubTotalIncrement(groupsOfFolds):
    return groupsOfFolds + 1

def indexMiniGapIncrement(my):
    my[indexMy.indexMiniGap.value] += 1

def indexMiniGapInitialization(my):
    my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]

def insertUnconstrainedLeaf(gapsWhere, my):
    my[indexMy.indexLeaf.value] = 0
    while my[indexMy.indexLeaf.value] < my[indexMy.leaf1ndex.value]:
        gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.indexLeaf.value]
        my[indexMy.gap1ndexCeiling.value] += 1
        my[indexMy.indexLeaf.value] += 1

def leafBelowSentinelIs1Condition(track):
    return track[indexTrack.leafBelow.value, 0] == 1

def leafConnecteeInitialization(connectionGraph, my):
    my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]]

def leafConnecteeUpdate(connectionGraph, my, track):
    my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]

def loopingLeavesConnectedToActiveLeaf(my):
    return my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]

def loopingTheDimensions(my):
    return my[indexMy.indexDimension.value] < my[indexMy.dimensionsTotal.value]

def loopingToActiveGapCeiling(my):
    return my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexCeiling.value]

def placeLeaf(gapsWhere, my, track):
    my[indexMy.gap1ndex.value] -= 1
    track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]] = gapsWhere[my[indexMy.gap1ndex.value]]
    track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]] = track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]]
    track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
    track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
    track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value]] = my[indexMy.gap1ndex.value]
    my[indexMy.leaf1ndex.value] += 1

def placeLeafCondition(my):
    return my[indexMy.leaf1ndex.value] > 0

def thereAreComputationDivisionsYouMightSkip(my):
    return my[indexMy.leaf1ndex.value] != my[indexMy.taskDivisions.value] or my[indexMy.leafConnectee.value] % my[indexMy.taskDivisions.value] == my[indexMy.taskIndex.value]

def countInitialize(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]]
                    , gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                    , my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                    , track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]):
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
                    , gapsWherePARALLEL: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                    , myPARALLEL: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]]
                    , trackPARALLEL: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]):
    for indexSherpa in numba.prange(myPARALLEL[indexMy.taskDivisions.value]):
        gapsWhere = gapsWherePARALLEL.copy()
        my = myPARALLEL.copy()
        my[indexMy.taskIndex.value] = indexSherpa
        track = trackPARALLEL.copy()
        groupsOfFolds: int = 0
        while activeLeafGreaterThan0Condition(my=my):
            if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
                if activeLeafGreaterThanLeavesTotalCondition(foldGroups=foldGroups, my=my):
                    groupsOfFolds = foldsSubTotalIncrement(groupsOfFolds=groupsOfFolds)
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

def countSequential(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]], foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]):
    doFindGaps = True
    groupsOfFolds: int = 0
    while activeLeafGreaterThan0Condition(my=my):
        if ((doFindGaps := activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track))
                and activeLeafGreaterThanLeavesTotalCondition(foldGroups=foldGroups, my=my)):
            groupsOfFolds = foldsSubTotalIncrement(groupsOfFolds=groupsOfFolds)
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

def doTheNeedful(connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[integer[Any]]], foldGroups: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], gapsWhere: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], mapShape: Tuple[int, ...], my: numpy.ndarray[Tuple[int], numpy.dtype[integer[Any]]], track: numpy.ndarray[Tuple[int, int], numpy.dtype[integer[Any]]]):
    countInitialize(connectionGraph, gapsWhere, my, track)

    if my[indexMy.taskDivisions.value] > 0:
        countParallel(connectionGraph, foldGroups, gapsWhere, my, track)
    else:
        countSequential(connectionGraph, foldGroups, gapsWhere, my, track)
