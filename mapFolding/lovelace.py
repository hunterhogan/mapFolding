from mapFolding import indexTrack, indexMy, indexThe
from typing import Tuple
import numpy
import numba

# @numba.jit(nopython=True, cache=True, fastmath=True)
@numba.jit(parallel=False, _nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True)
def countFoldsCompiled(connectionGraph: numpy.ndarray, foldsTotal: numpy.ndarray, mapShape: Tuple[int, ...], my: numpy.ndarray, potentialGaps: numpy.ndarray, the: numpy.ndarray, track: numpy.ndarray) -> int:
    def backtrack(my, track):
        # Allegedly, `-=` is an optimized, in-place operation in numpy and likely the best choice.
        my[indexMy.leaf1ndex.value] -= 1
        track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]
        track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]

    def backtrackCondition(my, track):
        return my[indexMy.leaf1ndex.value] > 0 and my[indexMy.gap1ndex.value] == track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]

    def checkActiveLeafGreaterThan0(my):
        return my[indexMy.leaf1ndex.value] > 0

    def checkActiveLeafGreaterThanLeavesTotal(my, the):
        return my[indexMy.leaf1ndex.value] > the[indexThe.leavesTotal.value]

    def checkActiveLeafNotEqualToTaskDivisions(my, the):
        return my[indexMy.leaf1ndex.value] != the[indexThe.taskDivisions.value]

    def checkActiveLeafIs1orLess(my):
        return my[indexMy.leaf1ndex.value] <= 1

    def checkComputationDivisions(the):
        return the[indexThe.taskDivisions.value] == int(False)

    def checkLeafBelowSentinelIs1(track):
        return track[indexTrack.leafBelow.value, 0] == 1

    def checkTaskIndex(my, the):
        return my[indexMy.leafConnectee.value] % the[indexThe.taskDivisions.value] == my[indexMy.taskIndex.value]

    def countGaps(my, potentialGaps, track):
        potentialGaps[my[indexMy.gap1ndexLowerBound.value]] = my[indexMy.leafConnectee.value]
        if track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] == 0:
            my[indexMy.gap1ndexLowerBound.value] += 1
        track[indexTrack.countDimensionsGapped.value, my[indexMy.leafConnectee.value]] += 1

    def filterCommonGaps(my, potentialGaps, the, track):
        potentialGaps[my[indexMy.gap1ndex.value]] = potentialGaps[my[indexMy.indexMiniGap.value]]
        if track[indexTrack.countDimensionsGapped.value, potentialGaps[my[indexMy.indexMiniGap.value]]] == the[indexThe.dimensionsTotal.value] - my[indexMy.dimensionsUnconstrained.value]:
            my[indexMy.gap1ndex.value] += 1
        track[indexTrack.countDimensionsGapped.value, potentialGaps[my[indexMy.indexMiniGap.value]]] = 0

    def findGapsInitialization(my, track):
        my[indexMy.dimensionsUnconstrained.value] = 0
        my[indexMy.gap1ndexLowerBound.value] = track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value] - 1]
        my[indexMy.dimension1ndex.value] = 1

    def initializeLeaf1ndexConnectee(connectionGraph, my):
        my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]]

    def incrementDimensionsUnconstrained(my):
        my[indexMy.dimensionsUnconstrained.value] += 1

    def incrementFoldsTotal(foldsTotal, my, the):
        foldsTotal[my[indexMy.taskIndex.value]] += the[indexThe.leavesTotal.value]

    def incrementIndexMiniGap(my):
        my[indexMy.indexMiniGap.value] += 1

    def insertUnconstrainedLeaf(my, potentialGaps):
        my[indexMy.indexLeaf.value] = 0
        while my[indexMy.indexLeaf.value] < my[indexMy.leaf1ndex.value]:
            potentialGaps[my[indexMy.gap1ndexLowerBound.value]] = my[indexMy.indexLeaf.value]
            my[indexMy.gap1ndexLowerBound.value] += 1
            my[indexMy.indexLeaf.value] += 1

    def placeLeaf(my, potentialGaps, track):
        my[indexMy.gap1ndex.value] -= 1
        track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]] = potentialGaps[my[indexMy.gap1ndex.value]]
        track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]] = track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]]
        track[indexTrack.leafBelow.value, track[indexTrack.leafAbove.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
        track[indexTrack.leafAbove.value, track[indexTrack.leafBelow.value, my[indexMy.leaf1ndex.value]]] = my[indexMy.leaf1ndex.value]
        track[indexTrack.gapRangeStart.value, my[indexMy.leaf1ndex.value]] = my[indexMy.gap1ndex.value]
        my[indexMy.leaf1ndex.value] += 1

    def placeLeafCondition(my):
        return my[indexMy.leaf1ndex.value] > 0

    def updateLeaf1ndexConnectee(connectionGraph, my, track):
        my[indexMy.leafConnectee.value] = connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], track[indexTrack.leafBelow.value, my[indexMy.leafConnectee.value]]]

    def lolaCondition_initializeUnconstrainedLeaf(my):
        if my[indexMy.gap1ndex.value] > 0:
            return int(True)
        return int(False)

    def initializeUnconstrainedLeaf(connectionGraph, foldsTotal, my, potentialGaps, the, track):
        while checkActiveLeafGreaterThan0(my):
            if checkActiveLeafIs1orLess(my) or checkLeafBelowSentinelIs1(track):
                if checkActiveLeafGreaterThanLeavesTotal(my, the):
                    incrementFoldsTotal(foldsTotal, my, the)
                else:
                    findGapsInitialization(my, track)
                    while my[indexMy.dimension1ndex.value] <= the[indexThe.dimensionsTotal.value]:
                        if connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]:
                            incrementDimensionsUnconstrained(my)
                        else:
                            initializeLeaf1ndexConnectee(connectionGraph, my)
                            while my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]:
                                my[indexMy.doCountGaps.value] = int(False)
                                if checkActiveLeafNotEqualToTaskDivisions(my, the):
                                    my[indexMy.doCountGaps.value] = int(True)
                                else:
                                    if checkTaskIndex(my, the):
                                        my[indexMy.doCountGaps.value] = int(True)
                                    else:
                                        if checkComputationDivisions(the):
                                            my[indexMy.doCountGaps.value] = int(True)
                                if my[indexMy.doCountGaps.value]:
                                    countGaps(my, potentialGaps, track)
                                updateLeaf1ndexConnectee(connectionGraph, my, track)
                        my[indexMy.dimension1ndex.value] += 1
                    if my[indexMy.dimensionsUnconstrained.value] == the[indexThe.dimensionsTotal.value]:
                        insertUnconstrainedLeaf(my, potentialGaps)
                    my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]
                    while my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexLowerBound.value]:
                        filterCommonGaps(my, potentialGaps, the, track)
                        incrementIndexMiniGap(my)
            while backtrackCondition(my, track):
                backtrack(my, track)
            if placeLeafCondition(my):
                placeLeaf(my, potentialGaps, track)
            if lolaCondition_initializeUnconstrainedLeaf(my):
                return

    def doWhile(connectionGraph, foldsTotal, my, potentialGaps, the, track):
        while checkActiveLeafGreaterThan0(my):
            if checkActiveLeafIs1orLess(my) or checkLeafBelowSentinelIs1(track):
                if checkActiveLeafGreaterThanLeavesTotal(my, the):
                    incrementFoldsTotal(foldsTotal, my, the)
                else:
                    findGapsInitialization(my, track)
                    while my[indexMy.dimension1ndex.value] <= the[indexThe.dimensionsTotal.value]:
                        if connectionGraph[my[indexMy.dimension1ndex.value], my[indexMy.leaf1ndex.value], my[indexMy.leaf1ndex.value]] == my[indexMy.leaf1ndex.value]:
                            incrementDimensionsUnconstrained(my)
                        else:
                            initializeLeaf1ndexConnectee(connectionGraph, my)
                            while my[indexMy.leafConnectee.value] != my[indexMy.leaf1ndex.value]:
                                my[indexMy.doCountGaps.value] = int(False)
                                if checkActiveLeafNotEqualToTaskDivisions(my, the):
                                    my[indexMy.doCountGaps.value] = int(True)
                                else:
                                    if checkTaskIndex(my, the):
                                        my[indexMy.doCountGaps.value] = int(True)
                                    else:
                                        if checkComputationDivisions(the):
                                            my[indexMy.doCountGaps.value] = int(True)
                                if my[indexMy.doCountGaps.value]:
                                    countGaps(my, potentialGaps, track)
                                updateLeaf1ndexConnectee(connectionGraph, my, track)
                        my[indexMy.dimension1ndex.value] += 1
                    if my[indexMy.dimensionsUnconstrained.value] == the[indexThe.dimensionsTotal.value]:
                        insertUnconstrainedLeaf(my, potentialGaps)
                    my[indexMy.indexMiniGap.value] = my[indexMy.gap1ndex.value]
                    while my[indexMy.indexMiniGap.value] < my[indexMy.gap1ndexLowerBound.value]:
                        filterCommonGaps(my, potentialGaps, the, track)
                        incrementIndexMiniGap(my)
            while backtrackCondition(my, track):
                backtrack(my, track)
            if placeLeafCondition(my):
                placeLeaf(my, potentialGaps, track)

    def doTaskIndices(connectionGraph, foldsTotal, my, potentialGaps, the, track):
        stateMy = my.copy()
        statePotentialGaps = potentialGaps.copy()
        stateTrack = track.copy()
        for indexSherpa in range(the[indexThe.taskDivisions.value]):
            my = stateMy.copy()
            my[indexMy.taskIndex.value] = indexSherpa
            potentialGaps = statePotentialGaps.copy()
            track = stateTrack.copy()
            doWhile(connectionGraph, foldsTotal, my, potentialGaps, the, track)

    initializeUnconstrainedLeaf(connectionGraph, foldsTotal, my, potentialGaps, the, track)

    if the[indexThe.taskDivisions.value] == int(False):
        doWhile(connectionGraph, foldsTotal, my, potentialGaps, the, track)
    else:
        doTaskIndices(connectionGraph, foldsTotal, my, potentialGaps, the, track)

    return numpy.sum(foldsTotal).item()

    # def lolaCondition_initializeTaskIndex():
    #     """
    #     NOTE hey hey hey !
    #     You tested this concept with taskDivisions hardcoded to leavesTotal. In that case,
    #     the Lola Condition was always true at `leavesTotal - 1`. You didn't do the math (because
    #     you can't), but the vibe is consistent with modulo math and the tests always passed.

    #     But, now you are switching back to taskDivisions as a variable. When
    #     `taskDivisions == leavesTotal`, all of the previous tests are relevant. But when
    #     they are not equal, the old tests are not relevant. Furthermore, when they are not
    #     equal, your tests so far suggest that `leavesTotal - 1` sometimes fails and
    #     `taskDivisions - 1` sometimes fails, so you don't have a direct substitute for
    #     `leavesTotal - 1`. Therefore,

    #     TODO prove, adjust, or remove this Lola Condition initialization
    #     Currently, this condition saves a handful of iterations. By itself, it's not much.
    #     """

    #     my[indexMy.doCountGaps.value] = int(True)
    #     # NOTE this is more than a mere condition check: it always forces countGaps to execute
    #     return my[indexMy.leafConnectee.value] % the[indexThe.leavesTotal.value] == the[indexThe.leavesTotal.value] - 1
    # def initializeTaskIndex():
    #                             if lolaCondition_initializeTaskIndex():
    #                                 return
    # initializeTaskIndex()
