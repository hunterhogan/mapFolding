from mapFolding import indexMy, indexTrack
import numba

@numba.jit((numba.int64[::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def activeGapIncrement(my):
    my[indexMy.gap1ndex.value] += 1

@numba.jit((numba.int64[::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def activeLeafGreaterThan0Condition(my):
    return my[indexMy.leaf1ndex.value] > 0

@numba.jit((numba.int64[::1],numba.int64[::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def activeLeafGreaterThanLeavesTotalCondition(foldGroups, my):
    return my[indexMy.leaf1ndex.value] > foldGroups[-1] # leavesTotal

@numba.jit((numba.int64[::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def activeLeafIsTheFirstLeafCondition(my):
    return my[indexMy.leaf1ndex.value] <= 1

@numba.jit((numba.int64[::1],numba.int64[::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def allDimensionsAreUnconstrained(my):
    return my[indexMy.dimensionsUnconstrained.value] == my[indexMy.dimensionsTotal.value]

@numba.jit((numba.int64[::1],numba.int64[:,::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def backtrack(my, track):
    my[indexMy.leaf1ndex.value] -= 1
    track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]
    track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]

@numba.jit((numba.int64[::1],numba.int64[:,::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def backtrackCondition(my, track):
    return my[indexMy.leaf1ndex.value] > 0 and my[indexMy.gap1ndex.value] == track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]

@numba.jit((numba.int64[::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def gap1ndexCeilingIncrement(my):
    my[indexMy.gap1ndexCeiling.value] += 1

@numba.jit((numba.int64[::1],numba.int64[::1],numba.int64[:,::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def countGaps(gapsWhere, my, track):
    gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.leafConnectee.value]
    if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
        gap1ndexCeilingIncrement(my=my)
    track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1

@numba.jit((numba.int64[::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def dimension1ndexIncrement(my):
    my[indexMy.indexDimension.value] += 1

@numba.jit((numba.int64[:,:,::1], numba.int64[::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def dimensionsUnconstrainedCondition(connectionGraph, my):
    return connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]

@numba.jit((numba.int64[::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def dimensionsUnconstrainedIncrement(my):
    my[indexMy.dimensionsUnconstrained.value] += 1

@numba.jit((numba.int64[::1],numba.int64[::1],numba.int64[::1],numba.int64[:,::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def filterCommonGaps(gapsWhere, my, track):
    gapsWhere[my[indexMy.gap1ndex.value]] = gapsWhere[my[indexMy.indexMiniGap.value]]
    if track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] == my[indexMy.dimensionsTotal.value] - my[indexMy.dimensionsUnconstrained.value]:
        activeGapIncrement(my=my)
    track[indexTrack.countDimensionsGapped.value, gapsWhere[my[indexMy.indexMiniGap.value]]] = 0

@numba.jit((numba.int64[::1],numba.int64[:,::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def findGapsInitializeVariables(my, track):
    my[indexMy.dimensionsUnconstrained.value] = 0
    my[indexMy.gap1ndexCeiling.value] = track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]
    my[indexMy.indexDimension.value] = 0

@numba.jit((numba.int64[::1],numba.int64[::1],numba.int64[::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def foldsSubTotalIncrement(foldGroups, my):
    foldGroups[my[indexMy.taskIndex.value]] += 1

@numba.jit((numba.int64[::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def indexMiniGapIncrement(my):
    my[indexMy.indexMiniGap.value] += 1

@numba.jit((numba.int64[::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def indexMiniGapInitialization(my):
    my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]

@numba.jit((numba.int64[::1],numba.int64[::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def insertUnconstrainedLeaf(gapsWhere, my):
    my[indexMy.indexLeaf.value] = 0
    while my[indexMy.indexLeaf.value] < my[indexMy.leaf1ndex.value]:
        gapsWhere[my[indexMy.gap1ndexCeiling.value]] = my[indexMy.indexLeaf.value]
        my[indexMy.gap1ndexCeiling.value] += 1
        my[indexMy.indexLeaf.value] += 1

@numba.jit((numba.int64[:,::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def leafBelowSentinelIs1Condition(track):
    return track[indexTrack.leafBelow.value, 0] == 1

@numba.jit((numba.int64[:,:,::1], numba.int64[::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def leafConnecteeInitialization(connectionGraph, my):
    my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]]

@numba.jit((numba.int64[:,:,::1], numba.int64[::1],numba.int64[:,::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def leafConnecteeUpdate(connectionGraph, my, track):
    my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.indexDimension.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]

@numba.jit((numba.int64[::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def loopingLeavesConnectedToActiveLeaf(my):
    return my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]

@numba.jit((numba.int64[::1],numba.int64[::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def loopingTheDimensions(my):
    return my[indexMy.indexDimension.value] < my[indexMy.dimensionsTotal.value]

@numba.jit((numba.int64[::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def loopingToActiveGapCeiling(my):
    return my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexCeiling.value]

@numba.jit((numba.int64[::1],numba.int64[::1],numba.int64[:,::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def placeLeaf(gapsWhere, my, track):
    my[indexMy.gap1ndex.value] -= 1
    track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]] = gapsWhere[my[indexMy.gap1ndex.value]]
    track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]] = track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]]
    track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
    track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
    track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value]] = my[indexMy.gap1ndex.value]
    my[indexMy.leaf1ndex.value] += 1

@numba.jit((numba.int64[::1],), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def placeLeafCondition(my):
    return my[indexMy.leaf1ndex.value] > 0

@numba.jit((numba.int64[::1],numba.int64[::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def thereAreComputationDivisionsYouMightSkip(my):
    return my[indexMy.leaf1ndex.value] != my[indexMy.taskDivisions.value] or my[indexMy.leafConnectee.value] % my[indexMy.taskDivisions.value] == my[indexMy.taskIndex.value]

@numba.jit((numba.int64[:,:,::1], numba.int64[::1], numba.int64[::1], numba.int64[:,::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def countInitialize(connectionGraph, gapsWhere, my, track):
    while activeLeafGreaterThan0Condition(my=my):
        if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
            findGapsInitializeVariables(my=my, track=track)
            while loopingTheDimensions(my=my):
                if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
                    dimensionsUnconstrainedIncrement(my=my)
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

@numba.jit((numba.int64[:,:,::1], numba.int64[::1], numba.int64[::1], numba.int64[::1], numba.int64[:,::1]), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def countSequential(connectionGraph, foldGroups, gapsWhere, my, track):
    while activeLeafGreaterThan0Condition(my=my):
        if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
            if activeLeafGreaterThanLeavesTotalCondition(foldGroups=foldGroups, my=my):
                foldsSubTotalIncrement(foldGroups=foldGroups, my=my)
            else:
                findGapsInitializeVariables(my=my, track=track)
                while loopingTheDimensions(my=my):
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
                    filterCommonGaps(gapsWhere=gapsWhere, my=my, track=track)
                    indexMiniGapIncrement(my=my)
        while backtrackCondition(my=my, track=track):
            backtrack(my=my, track=track)
        if placeLeafCondition(my=my):
            placeLeaf(gapsWhere=gapsWhere, my=my, track=track)

@numba.jit((numba.int64[:,:,::1], numba.int64[::1], numba.int64[::1], numba.int64[::1], numba.int64[:,::1]), parallel=True, boundscheck=False, error_model='numpy', fastmath=True, looplift=False, nogil=True, nopython=True)
def countParallel(connectionGraph, foldGroups, gapsWherePARALLEL, myPARALLEL, trackPARALLEL):
    for indexSherpa in numba.prange(myPARALLEL[indexMy.taskDivisions.value]):
        gapsWhere = gapsWherePARALLEL.copy()
        my = myPARALLEL.copy()
        my[indexMy.taskIndex.value] = indexSherpa
        track = trackPARALLEL.copy()
        while activeLeafGreaterThan0Condition(my=my):
            if activeLeafIsTheFirstLeafCondition(my=my) or leafBelowSentinelIs1Condition(track=track):
                if activeLeafGreaterThanLeavesTotalCondition(foldGroups=foldGroups, my=my):
                    foldsSubTotalIncrement(foldGroups=foldGroups, my=my)
                else:
                    findGapsInitializeVariables(my=my, track=track)
                    while loopingTheDimensions(my=my):
                        if dimensionsUnconstrainedCondition(connectionGraph=connectionGraph, my=my):
                            dimensionsUnconstrainedIncrement(my=my)
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
