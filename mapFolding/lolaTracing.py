from Z0Z_tools import dataTabularTOpathFilenameDelimited
from collections import Counter
from functools import wraps
from mapFolding import outfitFoldings
from types import FrameType
from typing import List, Callable, Any, Final, Literal, Dict
import pathlib
import sys
import time
import numpy
import numba

"""
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
        nonlocal activeLeaf1ndex
        activeLeaf1ndex -= 1
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = track[leafBelow, activeLeaf1ndex]
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = track[leafAbove, activeLeaf1ndex]

    def checkActiveLeafGreaterThan0():
        return activeLeaf1ndex > 0

    def checkActiveLeafGreaterThanLeavesTotal():
        return activeLeaf1ndex > leavesTotal

    def checkActiveLeafNotEqualToLeavesTotal():
        return activeLeaf1ndex != leavesTotal

    def checkActiveLeafIs1orLess():
        return activeLeaf1ndex <= 1

    def checkComputationDivisions():
        if computationDivisions == False:
            return True

    def checkLeafBelowSentinelIs1():
        return track[leafBelow, 0] == 1

    def checkTaskIndex():
        return leaf1ndexConnectee % leavesTotal == taskIndex

    def initializeLeaf1ndexConnectee():
        nonlocal leaf1ndexConnectee
        leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]

    def incrementActiveGap():
        nonlocal activeGap1ndex
        activeGap1ndex += 1

    def incrementDimensionsUnconstrained():
        nonlocal dimensionsUnconstrained
        dimensionsUnconstrained += 1

    def incrementFoldsTotal():
        nonlocal foldsTotal
        foldsTotal += leavesTotal

    def incrementGap1ndexLowerBound():
        nonlocal gap1ndexLowerBound
        gap1ndexLowerBound += 1

    def incrementIndexMiniGap():
        nonlocal indexMiniGap
        indexMiniGap += 1

    def insertUnconstrainedLeaf():
        nonlocal gap1ndexLowerBound, indexLeaf
        indexLeaf = 0
        while indexLeaf < activeLeaf1ndex:
            potentialGaps[gap1ndexLowerBound] = indexLeaf
            gap1ndexLowerBound += 1
            indexLeaf += 1

    def placeLeaf() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        activeGap1ndex -= 1
        track[leafAbove, activeLeaf1ndex] = potentialGaps[activeGap1ndex]
        track[leafBelow, activeLeaf1ndex] = track[leafBelow, track[leafAbove, activeLeaf1ndex]]
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = activeLeaf1ndex
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = activeLeaf1ndex
        track[gapRangeStart, activeLeaf1ndex] = activeGap1ndex
        activeLeaf1ndex += 1

    def updateLeaf1ndexConnectee():
        nonlocal leaf1ndexConnectee
        leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]

    def initializeUnconstrainedLeaf():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    nonlocal activeGap1ndex, dimensionsUnconstrained, gap1ndexLowerBound, dimension1ndex

                    dimensionsUnconstrained = 0
                    gap1ndexLowerBound = track[gapRangeStart, activeLeaf1ndex - 1]

                    dimension1ndex = 1
                    while dimension1ndex <= dimensionsTotal:
                        if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while leaf1ndexConnectee != activeLeaf1ndex:
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
                                    potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                                    if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
                                        incrementGap1ndexLowerBound()
                                    track[countDimensionsGapped, leaf1ndexConnectee] += 1
                                updateLeaf1ndexConnectee()
                        dimension1ndex += 1

                    if dimensionsUnconstrained == dimensionsTotal:
                        insertUnconstrainedLeaf()

                    nonlocal indexMiniGap
                    indexMiniGap = activeGap1ndex
                    while indexMiniGap < gap1ndexLowerBound:
                        potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                        if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                            incrementActiveGap()
                        track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
                        incrementIndexMiniGap()

            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()

            if activeLeaf1ndex > 0:
                placeLeaf()

            if activeGap1ndex > 0:
                return

    def initializeTaskIndex():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    nonlocal activeGap1ndex, dimensionsUnconstrained, gap1ndexLowerBound, dimension1ndex

                    dimensionsUnconstrained = 0
                    gap1ndexLowerBound = track[gapRangeStart, activeLeaf1ndex - 1]

                    dimension1ndex = 1
                    while dimension1ndex <= dimensionsTotal:
                        if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while leaf1ndexConnectee != activeLeaf1ndex:
                                if not checkActiveLeafNotEqualToLeavesTotal():
                                    if leaf1ndexConnectee % leavesTotal == leavesTotal - 1:
                                        # NOTE Lola condition
                                        return
                                potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                                if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
                                    incrementGap1ndexLowerBound()
                                track[countDimensionsGapped, leaf1ndexConnectee] += 1
                                updateLeaf1ndexConnectee()
                        dimension1ndex += 1

                    if dimensionsUnconstrained == dimensionsTotal:
                        insertUnconstrainedLeaf()

                    nonlocal indexMiniGap
                    indexMiniGap = activeGap1ndex
                    while indexMiniGap < gap1ndexLowerBound:
                        potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                        if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                            incrementActiveGap()
                        track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
                        incrementIndexMiniGap()

            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()

            if activeLeaf1ndex > 0:
                placeLeaf()

    def changeling():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    nonlocal activeGap1ndex, dimensionsUnconstrained, gap1ndexLowerBound, dimension1ndex, taskIndex, foldsTotal

                    dimensionsUnconstrained = 0
                    gap1ndexLowerBound = track[gapRangeStart, activeLeaf1ndex - 1]

                    dimension1ndex = 1
                    while dimension1ndex <= dimensionsTotal:
                        if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while leaf1ndexConnectee != activeLeaf1ndex:
                                if not checkActiveLeafNotEqualToLeavesTotal():
                                    taskPartition: int = leaf1ndexConnectee % leavesTotal
                                    if taskPartition == taskIndex:
                                        pass
                                    else:
                                        # NOTE Lola condition
                                        if track[modulus, taskPartition] == 0:
                                            track[modulus, taskIndex] = 1
                                            taskIndex = taskPartition
                                potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                                if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
                                    incrementGap1ndexLowerBound()
                                track[countDimensionsGapped, leaf1ndexConnectee] += 1
                                updateLeaf1ndexConnectee()
                        dimension1ndex += 1

                    if dimensionsUnconstrained == dimensionsTotal:
                        insertUnconstrainedLeaf()

                    nonlocal indexMiniGap
                    indexMiniGap = activeGap1ndex
                    while indexMiniGap < gap1ndexLowerBound:
                        potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                        if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                            incrementActiveGap()
                        track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
                        incrementIndexMiniGap()

            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()

            if activeLeaf1ndex > 0:
                placeLeaf()

            if all(track[modulus, index] == 1 for index in range(leavesTotal) if index != taskIndex):
                return

    def doWhile():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    nonlocal activeGap1ndex, dimensionsUnconstrained, gap1ndexLowerBound, dimension1ndex

                    dimensionsUnconstrained = 0
                    gap1ndexLowerBound = track[gapRangeStart, activeLeaf1ndex - 1]

                    dimension1ndex = 1
                    while dimension1ndex <= dimensionsTotal:
                        if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while leaf1ndexConnectee != activeLeaf1ndex:
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
                                    potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
                                    if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
                                        incrementGap1ndexLowerBound()
                                    track[countDimensionsGapped, leaf1ndexConnectee] += 1
                                updateLeaf1ndexConnectee()
                        dimension1ndex += 1

                    if dimensionsUnconstrained == dimensionsTotal:
                        insertUnconstrainedLeaf()

                    nonlocal indexMiniGap
                    indexMiniGap = activeGap1ndex
                    while indexMiniGap < gap1ndexLowerBound:
                        potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                        if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                            incrementActiveGap()
                        track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
                        incrementIndexMiniGap()

            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()

            if activeLeaf1ndex > 0:
                placeLeaf()

    listDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions)

    connectionGraph: Final[numpy.ndarray]
    countDimensionsGapped: Final[Literal[2]] = 2
    dimensionsTotal: Final[int] = len(listDimensions)
    gapRangeStart: Final[Literal[3]] = 3
    leafAbove: Final[Literal[0]] = 0
    leafBelow: Final[Literal[1]] = 1
    leavesTotal: Final[int]
    modulus: Final[Literal[4]] = 4
    subtotal: Final[Literal[4]] = 4

    activeGap1ndex: int = 0
    activeLeaf1ndex: int = 1
    dimension1ndex: int = 1
    dimensionsUnconstrained: int = 0
    foldsTotal: int = 0
    gap1ndexLowerBound: int = 0
    indexLeaf: int = 0
    indexMiniGap: int = 0
    leaf1ndexConnectee: int = 0
    taskIndex: int = 0

    initializeUnconstrainedLeaf()
    initializeTaskIndex()

    taskIndex = leavesTotal - 1
    changeling()
    """NOTE At the moment, changeling is very interesting, but after limited testing, computationDivisions does not work"""
    doWhile()

    # track[subtotal] = 0
    # track[subtotal, taskIndex] = foldsTotal
    # print(foldsTotal)
    # foldsTotal = 0

    # state_activeGap1ndex = activeGap1ndex
    # state_activeLeaf1ndex = activeLeaf1ndex
    # state_potentialGaps = potentialGaps.copy()
    # state_track = track.copy()

    # if computationDivisions:
    #     listTaskIndices = list(range(leavesTotal))
    # else:
    #     listTaskIndices = [taskIndex]

    # for taskIndex in listTaskIndices:
    #     print(foldsTotal)
    #     activeGap1ndex = state_activeGap1ndex
    #     activeLeaf1ndex = state_activeLeaf1ndex
    #     potentialGaps = state_potentialGaps.copy()
    #     track = state_track.copy()
    #     doWhile()

    # # print(foldsTotal)
    # foldsTotal = sum(track[subtotal]) + foldsTotal

    return foldsTotal
