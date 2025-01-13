from mapFolding import activeGap1ndex, activeLeaf1ndex, dimension1ndex, dimensionsUnconstrained, gap1ndexLowerBound, indexMiniGap, leaf1ndexConnectee
from mapFolding import outfitFoldings, t
from typing import List, Callable, Any, Final
import numpy
import numba

"""
from Z0Z_tools import dataTabularTOpathFilenameDelimited
from collections import Counter
from functools import wraps
from types import FrameType
import pathlib
import sys
import time
def traceCalls(functionTarget: Callable[..., Any]) -> Callable[..., Any]:
    def decoratorTrace(functionTarget: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(functionTarget)
        def wrapperTrace(*arguments, **keywordArguments):
            pathLog = pathlib.Path("/apps/mapFolding/Z0Z_notes")
            pathFilenameLog = pathLog / 'functionCalls.tab'
            timeStart = time.perf_counter_ns()
            listTraceCalls = []

            pathFilenameHost = pathlib.Path(__file__).resolve()
            def logCall(frame: FrameType, event: str, arg: object):
                if event == 'call':
                    listTraceCalls.append([time.perf_counter_ns(), frame.f_code.co_name, frame.f_code.co_filename])
                return logCall

            oldTrace = sys.gettrace()
            sys.settrace(logCall)
            try:
                return functionTarget(*arguments, **keywordArguments)
            finally:
                sys.settrace(oldTrace)
                # listTraceCalls = [[timeRelativeNS - timeStart, functionName] for timeRelativeNS, functionName, DISCARDpathFilename in listTraceCalls]
                listTraceCalls = [[timeRelativeNS - timeStart, functionName] for timeRelativeNS, functionName, module in listTraceCalls if  pathlib.Path(module).resolve() == pathFilenameHost]
                print(Counter([functionName for timeRelativeNS, functionName in listTraceCalls]))
                dataTabularTOpathFilenameDelimited(pathFilenameLog, listTraceCalls, ['timeRelativeNS', 'functionName'])

        return wrapperTrace
    return decoratorTrace(functionTarget)

"""
# @numba.jit(nopython=True, cache=True, fastmath=True)
def countFolds(listDimensions: List[int], computationDivisions: bool = False):
    def backtrack() -> None:
        nonlocal my
        my[activeLeaf1ndex] -= 1
        track[t.leafBelow.value, track[t.leafAbove.value, my[activeLeaf1ndex]]] = track[t.leafBelow.value, my[activeLeaf1ndex]]
        track[t.leafAbove.value, track[t.leafBelow.value, my[activeLeaf1ndex]]] = track[t.leafAbove.value, my[activeLeaf1ndex]]

    def backtrackCondition():
        nonlocal my
        return my[activeLeaf1ndex] > 0 and my[activeGap1ndex] == track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]

    def checkActiveLeafGreaterThan0():
        nonlocal my
        return my[activeLeaf1ndex] > 0

    def checkActiveLeafGreaterThanLeavesTotal():
        nonlocal my
        return my[activeLeaf1ndex] > leavesTotal

    # TODO taskDivisions
    def checkActiveLeafNotEqualToLeavesTotal():
        nonlocal my
        return my[activeLeaf1ndex] != leavesTotal

    def checkActiveLeafIs1orLess():
        nonlocal my
        return my[activeLeaf1ndex] <= 1

    def checkComputationDivisions():
        return computationDivisions == False

    def checkLeafBelowSentinelIs1():
        return track[t.leafBelow.value, 0] == 1

    # TODO taskDivisions
    def checkTaskIndex():
        nonlocal taskPartition, my
        return (taskPartition := my[leaf1ndexConnectee] % leavesTotal) == taskIndex

    def countGaps():
        nonlocal my
        potentialGaps[my[gap1ndexLowerBound]] = my[leaf1ndexConnectee]
        if track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] == 0:
            incrementGap1ndexLowerBound()
        track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] += 1

    def filterCommonGaps():
        nonlocal my
        potentialGaps[my[activeGap1ndex]] = potentialGaps[my[indexMiniGap]]
        if track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] == dimensionsTotal - my[dimensionsUnconstrained]:
            incrementActiveGap()
        track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] = 0

    def findGapsInitialization():
        nonlocal my
        my[dimensionsUnconstrained] = 0
        my[gap1ndexLowerBound] = track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]
        my[dimension1ndex] = 1

    def initializeLeaf1ndexConnectee():
        nonlocal my
        my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]]

    def incrementActiveGap():
        nonlocal my
        my[activeGap1ndex] += 1

    def incrementDimensionsUnconstrained():
        nonlocal my
        my[dimensionsUnconstrained] += 1

    def incrementFoldsTotal():
        nonlocal foldsTotal
        foldsTotal += leavesTotal

    def incrementGap1ndexLowerBound():
        nonlocal my
        my[gap1ndexLowerBound] += 1

    def incrementIndexMiniGap():
        nonlocal my
        my[indexMiniGap] += 1

    def insertUnconstrainedLeaf():
        nonlocal indexLeaf, my
        indexLeaf = 0
        while indexLeaf < my[activeLeaf1ndex]:
            potentialGaps[my[gap1ndexLowerBound]] = indexLeaf
            my[gap1ndexLowerBound] += 1
            indexLeaf += 1

    def placeLeaf():
        nonlocal my
        my[activeGap1ndex] -= 1
        track[t.leafAbove.value, my[activeLeaf1ndex]] = potentialGaps[my[activeGap1ndex]]
        track[t.leafBelow.value, my[activeLeaf1ndex]] = track[t.leafBelow.value, track[t.leafAbove.value, my[activeLeaf1ndex]]]
        track[t.leafBelow.value, track[t.leafAbove.value, my[activeLeaf1ndex]]] = my[activeLeaf1ndex]
        track[t.leafAbove.value, track[t.leafBelow.value, my[activeLeaf1ndex]]] = my[activeLeaf1ndex]
        track[t.gapRangeStart.value, my[activeLeaf1ndex]] = my[activeGap1ndex]
        my[activeLeaf1ndex] += 1

    def placeLeafCondition():
        nonlocal my
        return my[activeLeaf1ndex] > 0

    def updateLeaf1ndexConnectee():
        nonlocal my
        my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], track[t.leafBelow.value, my[leaf1ndexConnectee]]]

    def lolaCondition_initializeUnconstrainedLeaf():
        nonlocal my
        if my[activeGap1ndex] > 0:
            return True
        return False

    # TODO taskDivisions
    def lolaCondition_initializeTaskIndex():
        # NOTE this is more than a mere condition check: it always forces countGaps to execute
        nonlocal doCountGaps, my
        doCountGaps = True
        if my[leaf1ndexConnectee] % leavesTotal == leavesTotal - 1:
            return True
        return False

    def initializeUnconstrainedLeaf():
        nonlocal my
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    findGapsInitialization()
                    while my[dimension1ndex] <= dimensionsTotal:
                        if connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                                nonlocal doCountGaps
                                doCountGaps = False
                                if checkActiveLeafNotEqualToLeavesTotal():
                                    doCountGaps = True
                                else:
                                    if checkTaskIndex():
                                        doCountGaps = True
                                    else:
                                        if checkComputationDivisions():
                                            doCountGaps = True
                                if doCountGaps:
                                    countGaps()
                                updateLeaf1ndexConnectee()
                        my[dimension1ndex] += 1

                    if my[dimensionsUnconstrained] == dimensionsTotal:
                        insertUnconstrainedLeaf()

                    my[indexMiniGap] = my[activeGap1ndex]
                    while my[indexMiniGap] < my[gap1ndexLowerBound]:
                        filterCommonGaps()
                        incrementIndexMiniGap()
            while backtrackCondition():
                backtrack()
            if placeLeafCondition():
                placeLeaf()

            if lolaCondition_initializeUnconstrainedLeaf():
                return

    def initializeTaskIndex():
        nonlocal my
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    findGapsInitialization()
                    while my[dimension1ndex] <= dimensionsTotal:
                        if connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                                nonlocal doCountGaps
                                doCountGaps = False
                                if checkActiveLeafNotEqualToLeavesTotal():
                                    doCountGaps = True
                                else:
                                    if lolaCondition_initializeTaskIndex():
                                        return
                                    if checkTaskIndex():
                                        doCountGaps = True
                                    else:
                                        if checkComputationDivisions():
                                            doCountGaps = True
                                if doCountGaps:
                                    countGaps()
                                updateLeaf1ndexConnectee()
                        my[dimension1ndex] += 1
                    if my[dimensionsUnconstrained] == dimensionsTotal:
                        insertUnconstrainedLeaf()
                    my[indexMiniGap] = my[activeGap1ndex]
                    while my[indexMiniGap] < my[gap1ndexLowerBound]:
                        filterCommonGaps()
                        incrementIndexMiniGap()
            while backtrackCondition():
                backtrack()
            if placeLeafCondition():
                placeLeaf()

    def doWhile():
        nonlocal my
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    findGapsInitialization()
                    while my[dimension1ndex] <= dimensionsTotal:
                        if connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                                nonlocal doCountGaps
                                doCountGaps = False
                                if checkActiveLeafNotEqualToLeavesTotal():
                                    doCountGaps = True
                                else:
                                    if checkTaskIndex():
                                        doCountGaps = True
                                    else:
                                        if checkComputationDivisions():
                                            doCountGaps = True
                                if doCountGaps:
                                    countGaps()
                                updateLeaf1ndexConnectee()
                        my[dimension1ndex] += 1
                    if my[dimensionsUnconstrained] == dimensionsTotal:
                        insertUnconstrainedLeaf()
                    my[indexMiniGap] = my[activeGap1ndex]
                    while my[indexMiniGap] < my[gap1ndexLowerBound]:
                        filterCommonGaps()
                        incrementIndexMiniGap()
            while backtrackCondition():
                backtrack()
            if placeLeafCondition():
                placeLeaf()

    listDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions)

    my = track[t.my.value]

    connectionGraph: Final[numpy.ndarray]
    dimensionsTotal: Final[int] = len(listDimensions)
    leavesTotal: Final[int]

    my[activeLeaf1ndex] = 1
    doCountGaps: bool = False
    foldsTotal: int = 0
    indexLeaf: int = 0
    taskIndex: int = 0
    taskPartition: int = 0

    initializeUnconstrainedLeaf()
    initializeTaskIndex()

    # I'm confused. If `my` is a view of `track`, then why do I need to copy it? numba.jit is not activated
    # Nevertheless, is it somehow related to numba.jit rejecting tuple indexing?
    state_my = my.copy()
    state_potentialGaps = potentialGaps.copy()
    state_track = track.copy()

    if computationDivisions:
        listTaskIndices = list(range(leavesTotal))
    else:
        listTaskIndices = [taskIndex]

    for taskIndex in listTaskIndices:
        my = state_my.copy()
        potentialGaps = state_potentialGaps.copy()
        track = state_track.copy()
        doWhile()

    return foldsTotal
