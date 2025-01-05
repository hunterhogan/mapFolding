from collections import Counter
from functools import wraps
from mapFolding import outfitFoldings
from numpy.typing import NDArray
from types import FrameType
from typing import List, Callable, Any
from Z0Z_tools import dataTabularTOpathFilenameDelimited
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

def countFolds(listDimensions: List[int], computationDivisions: bool = False) -> int:
    listDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions)
    computationIndex = 0
    dimensionsTotal = len(listDimensions)
    return countFoldings(track, potentialGaps, connectionGraph, leavesTotal, dimensionsTotal, int(computationDivisions), computationIndex)

def countFoldings(track: NDArray, potentialGaps: NDArray, connectionGraph: NDArray, leavesTotal: int, dimensionsTotal: int, taskDivisions: int, taskIndex: int) -> int:
    def backtrack() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        activeLeaf1ndex -= 1
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = track[leafBelow, activeLeaf1ndex]
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = track[leafAbove, activeLeaf1ndex]

    def checkActiveLeafGreaterThan0():
        return activeLeaf1ndex > 0

    def checkActiveLeafGreaterThanLeavesTotal():
        return activeLeaf1ndex > leavesTotal

    def checkActiveLeafIs1orLess():
        if activeLeaf1ndex <= 1:
            return True
        else:
            return checkLeafBelowSentinelIs1()

    def checkLeafBelowSentinelIs1():
        return track[leafBelow, 0] == 1

    def countGaps():
        nonlocal gap1ndexLowerBound
        potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
        if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
            incrementGap1ndexLowerBound()
        track[countDimensionsGapped, leaf1ndexConnectee] += 1

    def filterCommonGaps() -> None:
        def incrementIndexMiniGap():
            nonlocal indexMiniGap
            indexMiniGap += 1
        nonlocal activeGap1ndex, dimensionsUnconstrained, gap1ndexLowerBound
        indexMiniGap = activeGap1ndex
        while indexMiniGap < gap1ndexLowerBound:
            potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
            if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                incrementActiveGap()
            track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
            incrementIndexMiniGap()

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

    def insertUnconstrainedLeaf(gapNumberLowerBound: int) -> int:
        index = 0
        while index < activeLeaf1ndex:
            potentialGaps[gapNumberLowerBound] = index
            gapNumberLowerBound += 1
            index += 1
        return gapNumberLowerBound

    def placeLeaf() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        activeGap1ndex -= 1
        track[leafAbove, activeLeaf1ndex] = potentialGaps[activeGap1ndex]
        track[leafBelow, activeLeaf1ndex] = track[leafBelow, track[leafAbove, activeLeaf1ndex]]
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = activeLeaf1ndex
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = activeLeaf1ndex
        track[gapRangeStart, activeLeaf1ndex] = activeGap1ndex
        activeLeaf1ndex += 1

    def initialize_insertUnconstrainedLeaf():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess():
                if checkActiveLeafGreaterThanLeavesTotal(): incrementFoldsTotal()
                else:
                    nonlocal activeGap1ndex, dimensionsUnconstrained, gap1ndexLowerBound, dimension1ndex

                    dimensionsUnconstrained= 0
                    gap1ndexLowerBound = track[gapRangeStart, activeLeaf1ndex - 1]
                    activeGap1ndex = gap1ndexLowerBound

                    while dimension1ndex <= dimensionsTotal:
                        if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                            incrementDimensionsUnconstrained()
                        else:
                            nonlocal leaf1ndexConnectee
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                            while leaf1ndexConnectee != activeLeaf1ndex:
                                if taskDivisions == 0 or activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
                                    if leaf1ndexConnectee % leavesTotal == leavesTotal - 1:
                                        nonlocal taskIndexInitialized
                                        taskIndexInitialized = True
                                        return
                                    countGaps()
                                leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]

                        dimension1ndex += 1

                    dimension1ndex = 1 # NOTE this initializes the next loop
                    if dimensionsUnconstrained == dimensionsTotal: gap1ndexLowerBound = insertUnconstrainedLeaf(gap1ndexLowerBound)

                    filterCommonGaps()

            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]: backtrack()

            if activeLeaf1ndex > 0:
                placeLeaf()

            # NOTE Lola condition
            # if activeGap1ndex > 0:
            #     return

    def initialize_taskIndex():
        """foldsTotal will increment on the iteration following this return."""
        while checkActiveLeafGreaterThan0(): # NOTE placeholder

            # NOTE Lola condition
            if taskIndexInitialized:
                return

    def middleGame():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess():
                if checkActiveLeafGreaterThanLeavesTotal(): incrementFoldsTotal()
                else:
                    nonlocal activeGap1ndex, dimensionsUnconstrained, gap1ndexLowerBound, dimension1ndex

                    dimensionsUnconstrained= 0
                    gap1ndexLowerBound = track[gapRangeStart, activeLeaf1ndex - 1]
                    activeGap1ndex = gap1ndexLowerBound

                    while dimension1ndex <= dimensionsTotal:
                        if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                            incrementDimensionsUnconstrained()
                        else:
                            nonlocal leaf1ndexConnectee
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                            while leaf1ndexConnectee != activeLeaf1ndex:
                                if taskDivisions == 0 or activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
                                    countGaps()
                                    if leaf1ndexConnectee % leavesTotal == leavesTotal - 1:
                                        nonlocal taskIndexInitialized
                                        taskIndexInitialized = True
                                leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]

                        dimension1ndex += 1

                    dimension1ndex = 1 # NOTE this initializes the next loop
                    if dimensionsUnconstrained == dimensionsTotal: gap1ndexLowerBound = insertUnconstrainedLeaf(gap1ndexLowerBound)

                    filterCommonGaps()

            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]: backtrack()

            if activeLeaf1ndex > 0:
                placeLeaf()

    leafAbove, leafBelow, countDimensionsGapped, gapRangeStart = 0, 1, 2, 3

    activeLeaf1ndex: int = 1
    activeGap1ndex: int = 0
    foldsTotal: int = 0
    taskIndexInitialized: bool = False
    dimensionsUnconstrained: int = 0
    gap1ndexLowerBound:int = 0
    dimension1ndex: int = 1 # NOTE This initialization is correct.
    leaf1ndexConnectee: int = 0

    initialize_insertUnconstrainedLeaf() # NOTE This state can initialize all task indices.
    # initialize_taskIndex()

    statePreserved = (activeLeaf1ndex, activeGap1ndex, track.copy(), potentialGaps.copy())

    for taskIndex in range(leavesTotal):
        activeLeaf1ndex, activeGap1ndex, track, potentialGaps = statePreserved
        # _0, _1, foldsTotal, track, potentialGaps = statePreserved
        # activeLeaf1ndex = 1
        # activeGap1ndex = 0
        # taskIndex = index
        middleGame()

    # middleGame()

    return foldsTotal
