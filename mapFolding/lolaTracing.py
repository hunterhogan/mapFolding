from mapFolding import activeGap1ndex, activeLeaf1ndex, dimension1ndex, dimensionsUnconstrained, gap1ndexLowerBound, indexMiniGap, leaf1ndexConnectee
from Z0Z_tools import dataTabularTOpathFilenameDelimited
from collections import Counter
from functools import wraps
from mapFolding import outfitFoldings, t
from types import FrameType
from typing import List, Callable, Any, Final, Literal, Dict
import pathlib
import sys
import time
import numpy
import numba

# """
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

# """
# @numba.jit(nopython=True, cache=True, fastmath=True)
def countFolds(listDimensions: List[int], computationDivisions: bool = False):
    def backtrack() -> None:
        my[activeLeaf1ndex] -= 1
        track[t.leafBelow.value, track[t.leafAbove.value, my[activeLeaf1ndex]]] = track[t.leafBelow.value, my[activeLeaf1ndex]]
        track[t.leafAbove.value, track[t.leafBelow.value, my[activeLeaf1ndex]]] = track[t.leafAbove.value, my[activeLeaf1ndex]]

    def checkActiveLeafGreaterThan0():
        return my[activeLeaf1ndex] > 0

    def checkActiveLeafGreaterThanLeavesTotal():
        return my[activeLeaf1ndex] > leavesTotal

    def checkActiveLeafNotEqualToLeavesTotal():
        return my[activeLeaf1ndex] != leavesTotal

    def checkActiveLeafIs1orLess():
        return my[activeLeaf1ndex] <= 1

    def checkComputationDivisions():
        return computationDivisions == False

    def checkLeafBelowSentinelIs1():
        return track[t.leafBelow.value, 0] == 1

    def checkTaskIndex():
        nonlocal taskPartition
        return (taskPartition := my[leaf1ndexConnectee] % leavesTotal) == taskIndex

    def initializeLeaf1ndexConnectee():
        my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]]

    def incrementActiveGap():
        my[activeGap1ndex] += 1

    def incrementDimensionsUnconstrained():
        my[dimensionsUnconstrained] += 1

    def incrementFoldsTotal():
        nonlocal foldsTotal
        foldsTotal += leavesTotal

    def incrementGap1ndexLowerBound():
        my[gap1ndexLowerBound] += 1

    def incrementIndexMiniGap():
        my[indexMiniGap] += 1

    def insertUnconstrainedLeaf():
        nonlocal indexLeaf
        indexLeaf = 0
        while indexLeaf < my[activeLeaf1ndex]:
            potentialGaps[my[gap1ndexLowerBound]] = indexLeaf
            my[gap1ndexLowerBound] += 1
            indexLeaf += 1

    def placeLeaf() -> None:
        my[activeGap1ndex] -= 1
        track[t.leafAbove.value, my[activeLeaf1ndex]] = potentialGaps[my[activeGap1ndex]]
        track[t.leafBelow.value, my[activeLeaf1ndex]] = track[t.leafBelow.value, track[t.leafAbove.value, my[activeLeaf1ndex]]]
        track[t.leafBelow.value, track[t.leafAbove.value, my[activeLeaf1ndex]]] = my[activeLeaf1ndex]
        track[t.leafAbove.value, track[t.leafBelow.value, my[activeLeaf1ndex]]] = my[activeLeaf1ndex]
        track[t.gapRangeStart.value, my[activeLeaf1ndex]] = my[activeGap1ndex]
        my[activeLeaf1ndex] += 1

    def restoreTaskState(taskNumber):
        potentialGaps = dictionaryTaskState[taskIndex]['potentialGaps'].copy()
        track = dictionaryTaskState[taskIndex]['track'].copy()

    def saveTaskState(taskNumber):
        dictionaryTaskState[taskNumber] = dict(
        potentialGaps = potentialGaps.copy(),
        track = track.copy(),
        )

    def updateLeaf1ndexConnectee():
        my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], track[t.leafBelow.value, my[leaf1ndexConnectee]]]

    def initializeUnconstrainedLeaf():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    my[dimensionsUnconstrained] = 0
                    my[gap1ndexLowerBound] = track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]

                    my[dimension1ndex] = 1
                    while my[dimension1ndex] <= dimensionsTotal:
                        if connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
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
                                    potentialGaps[my[gap1ndexLowerBound]] = my[leaf1ndexConnectee]
                                    if track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] == 0:
                                        incrementGap1ndexLowerBound()
                                    track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] += 1
                                updateLeaf1ndexConnectee()
                        my[dimension1ndex] += 1

                    if my[dimensionsUnconstrained] == dimensionsTotal:
                        insertUnconstrainedLeaf()

                    my[indexMiniGap] = my[activeGap1ndex]
                    while my[indexMiniGap] < my[gap1ndexLowerBound]:
                        potentialGaps[my[activeGap1ndex]] = potentialGaps[my[indexMiniGap]]
                        if track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] == dimensionsTotal - my[dimensionsUnconstrained]:
                            incrementActiveGap()
                        track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] = 0
                        incrementIndexMiniGap()

            while my[activeLeaf1ndex] > 0 and my[activeGap1ndex] == track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]:
                backtrack()

            if my[activeLeaf1ndex] > 0:
                placeLeaf()

            if my[activeGap1ndex] > 0:
                return

    def initializeTaskIndex():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    my[dimensionsUnconstrained] = 0
                    my[gap1ndexLowerBound] = track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]

                    my[dimension1ndex] = 1
                    while my[dimension1ndex] <= dimensionsTotal:
                        if connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                                if not checkActiveLeafNotEqualToLeavesTotal():
                                    if my[leaf1ndexConnectee] % leavesTotal == leavesTotal - 1:
                                        # NOTE Lola condition
                                        return
                                potentialGaps[my[gap1ndexLowerBound]] = my[leaf1ndexConnectee]
                                if track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] == 0:
                                    incrementGap1ndexLowerBound()
                                track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] += 1
                                updateLeaf1ndexConnectee()
                        my[dimension1ndex] += 1

                    if my[dimensionsUnconstrained] == dimensionsTotal:
                        insertUnconstrainedLeaf()

                    my[indexMiniGap] = my[activeGap1ndex]
                    while my[indexMiniGap] < my[gap1ndexLowerBound]:
                        potentialGaps[my[activeGap1ndex]] = potentialGaps[my[indexMiniGap]]
                        if track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] == dimensionsTotal - my[dimensionsUnconstrained]:
                            incrementActiveGap()
                        track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] = 0
                        incrementIndexMiniGap()

            while my[activeLeaf1ndex] > 0 and my[activeGap1ndex] == track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]:
                backtrack()

            if my[activeLeaf1ndex] > 0:
                placeLeaf()

    def mitosis():
        while checkActiveLeafGreaterThan0():
            runLolaRun = False
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    my[dimensionsUnconstrained] = 0
                    my[gap1ndexLowerBound] = track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]

                    my[dimension1ndex] = 1
                    while my[dimension1ndex] <= dimensionsTotal:
                        if connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:

                                doCountGaps = False
                                if checkActiveLeafNotEqualToLeavesTotal():
                                    doCountGaps = True
                                else:
                                    if checkTaskIndex():
                                        doCountGaps = True
                                    else:
                                        # NOTE Lola condition
                                        if modulus[taskPartition] == 0 and runLolaRun == False:
                                            saveTaskState(taskIndex)
                                            runLolaRun = True
                                            doCountGaps = True

                                if doCountGaps:
                                    potentialGaps[my[gap1ndexLowerBound]] = my[leaf1ndexConnectee]
                                    if track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] == 0:
                                        incrementGap1ndexLowerBound()
                                    track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] += 1

                                updateLeaf1ndexConnectee()
                        my[dimension1ndex] += 1

                    if my[dimensionsUnconstrained] == dimensionsTotal:
                        insertUnconstrainedLeaf()

                    my[indexMiniGap] = my[activeGap1ndex]
                    while my[indexMiniGap] < my[gap1ndexLowerBound]:
                        potentialGaps[my[activeGap1ndex]] = potentialGaps[my[indexMiniGap]]
                        if track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] == dimensionsTotal - my[dimensionsUnconstrained]:
                            incrementActiveGap()
                        track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] = 0
                        incrementIndexMiniGap()
            while my[activeLeaf1ndex] > 0 and my[activeGap1ndex] == track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]:
                backtrack()
            if my[activeLeaf1ndex] > 0:
                placeLeaf()

            if runLolaRun:
                saveTaskState(taskPartition)
                restoreTaskState(taskIndex)

            if all(modulus[index] == 1 for index in range(leavesTotal) if index != taskIndex):
                saveTaskState(taskIndex)
                return

    def doWhile():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    my[dimensionsUnconstrained] = 0
                    my[gap1ndexLowerBound] = track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]

                    my[dimension1ndex] = 1
                    while my[dimension1ndex] <= dimensionsTotal:
                        if connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
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
                                    potentialGaps[my[gap1ndexLowerBound]] = my[leaf1ndexConnectee]
                                    if track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] == 0:
                                        incrementGap1ndexLowerBound()
                                    track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] += 1
                                updateLeaf1ndexConnectee()
                        my[dimension1ndex] += 1

                    if my[dimensionsUnconstrained] == dimensionsTotal:
                        insertUnconstrainedLeaf()

                    my[indexMiniGap] = my[activeGap1ndex]
                    while my[indexMiniGap] < my[gap1ndexLowerBound]:
                        potentialGaps[my[activeGap1ndex]] = potentialGaps[my[indexMiniGap]]
                        if track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] == dimensionsTotal - my[dimensionsUnconstrained]:
                            incrementActiveGap()
                        track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] = 0
                        incrementIndexMiniGap()

            while my[activeLeaf1ndex] > 0 and my[activeGap1ndex] == track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]:
                backtrack()

            if my[activeLeaf1ndex] > 0:
                placeLeaf()

    listDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions)
    my = track[t.my.value]
    connectionGraph: Final[numpy.ndarray]
    dimensionsTotal: Final[int] = len(listDimensions)
    leavesTotal: Final[int]

    my[activeLeaf1ndex] = 1
    foldsTotal: int = 0
    indexLeaf: int = 0
    modulus = numpy.zeros(leavesTotal, dtype=numpy.int64)
    taskIndex: int = 0
    taskPartition: int = 0

    dictionaryTaskState = {}

    initializeUnconstrainedLeaf()
    initializeTaskIndex()

    # taskIndex = leavesTotal - 1
    # mitosis()

    # for taskIndex in dictionaryTaskState.keys():
    #     restoreTaskState(taskIndex)
    #     doWhile()

    state_activeGap1ndex = my[activeGap1ndex]
    state_activeLeaf1ndex = my[activeLeaf1ndex]
    state_potentialGaps = potentialGaps.copy()
    state_track = track.copy()

    if computationDivisions:
        listTaskIndices = list(range(leavesTotal))
    else:
        listTaskIndices = [taskIndex]

    for taskIndex in listTaskIndices:
        my[activeGap1ndex] = state_activeGap1ndex
        my[activeLeaf1ndex] = state_activeLeaf1ndex
        potentialGaps = state_potentialGaps.copy()
        track = state_track.copy()
        doWhile()

    # foldsTotal = sum(my[subtotal]) + foldsTotal

    return foldsTotal
