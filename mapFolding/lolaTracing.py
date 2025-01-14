from mapFolding import Z0Z_outfitFoldings, indexTrack, Z0Z_computationState, indexMy, indexThe
from typing import List, Callable, Any, Final, Optional, Union, Sequence, Tuple
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
# @numba.jit(_nrt=True, boundscheck=False, error_model='numpy', fastmath=True, forceinline=True, looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nogil=True, nopython=True, parallel=False)
# @numba.jit(nopython=True, cache=True, fastmath=True, forceinline=True)
def countFolds(listDimensions: Sequence[int], CPUlimit: Optional[Union[int, float, bool]] = None, computationDivisions = None):
    def backtrack() -> None:
        # Allegedly, `-=` is an optimized, in-place operation in numpy and likely the best choice.
        my[indexMy.leaf1ndex] -= 1
        track[indexTrack.leafBelow, track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]] = track[indexTrack.leafBelow, my[indexMy.leaf1ndex]]
        track[indexTrack.leafAbove, track[indexTrack.leafBelow, my[indexMy.leaf1ndex]]] = track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]

    def backtrackCondition():
        return my[indexMy.leaf1ndex] > 0 and my[indexMy.gap1ndex] == track[indexTrack.gapRangeStart, my[indexMy.leaf1ndex] - 1]

    def checkActiveLeafGreaterThan0():
        return my[indexMy.leaf1ndex] > 0

    def checkActiveLeafGreaterThanLeavesTotal():
        return my[indexMy.leaf1ndex] > the[indexThe.leavesTotal]

    # TODO taskDivisions
    def checkActiveLeafNotEqualToLeavesTotal():
        return my[indexMy.leaf1ndex] != the[indexThe.leavesTotal]

    def checkActiveLeafIs1orLess():
        return my[indexMy.leaf1ndex] <= 1

    def checkComputationDivisions():
        return the[indexThe.taskDivisions] == int(False)

    def checkLeafBelowSentinelIs1():
        return track[indexTrack.leafBelow, 0] == 1

    # TODO taskDivisions
    def checkTaskIndex():
        return my[indexMy.leafConnectee] % the[indexThe.leavesTotal] == my[indexMy.taskIndex]

    def countGaps():
        potentialGaps[my[indexMy.gap1ndexLowerBound]] = my[indexMy.leafConnectee]
        if track[indexTrack.countDimensionsGapped, my[indexMy.leafConnectee]] == 0:
            incrementGap1ndexLowerBound()
        track[indexTrack.countDimensionsGapped, my[indexMy.leafConnectee]] += 1

    def filterCommonGaps():
        potentialGaps[my[indexMy.gap1ndex]] = potentialGaps[my[indexMy.indexMiniGap]]
        if track[indexTrack.countDimensionsGapped, potentialGaps[my[indexMy.indexMiniGap]]] == the[indexThe.dimensionsTotal] - my[indexMy.dimensionsUnconstrained]:
            incrementActiveGap()
        track[indexTrack.countDimensionsGapped, potentialGaps[my[indexMy.indexMiniGap]]] = 0

    def findGapsInitialization():
        my[indexMy.dimensionsUnconstrained] = 0
        my[indexMy.gap1ndexLowerBound] = track[indexTrack.gapRangeStart, my[indexMy.leaf1ndex] - 1]
        my[indexMy.dimension1ndex] = 1

    def initializeLeaf1ndexConnectee():
        my[indexMy.leafConnectee] = connectionGraph[my[indexMy.dimension1ndex], my[indexMy.leaf1ndex], my[indexMy.leaf1ndex]]

    def incrementActiveGap():
        my[indexMy.gap1ndex] += 1

    def incrementDimensionsUnconstrained():
        my[indexMy.dimensionsUnconstrained] += 1

    def incrementFoldsTotal():
        foldsTotal[my[indexMy.taskIndex]] += the[indexThe.leavesTotal]

    def incrementGap1ndexLowerBound():
        my[indexMy.gap1ndexLowerBound] += 1

    def incrementIndexMiniGap():
        my[indexMy.indexMiniGap] += 1

    def insertUnconstrainedLeaf():
        my[indexMy.indexLeaf] = 0
        while my[indexMy.indexLeaf] < my[indexMy.leaf1ndex]:
            potentialGaps[my[indexMy.gap1ndexLowerBound]] = my[indexMy.indexLeaf]
            my[indexMy.gap1ndexLowerBound] += 1
            my[indexMy.indexLeaf] += 1

    def placeLeaf():
        my[indexMy.gap1ndex] -= 1
        track[indexTrack.leafAbove, my[indexMy.leaf1ndex]] = potentialGaps[my[indexMy.gap1ndex]]
        track[indexTrack.leafBelow, my[indexMy.leaf1ndex]] = track[indexTrack.leafBelow, track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]]
        track[indexTrack.leafBelow, track[indexTrack.leafAbove, my[indexMy.leaf1ndex]]] = my[indexMy.leaf1ndex]
        track[indexTrack.leafAbove, track[indexTrack.leafBelow, my[indexMy.leaf1ndex]]] = my[indexMy.leaf1ndex]
        track[indexTrack.gapRangeStart, my[indexMy.leaf1ndex]] = my[indexMy.gap1ndex]
        my[indexMy.leaf1ndex] += 1

    def placeLeafCondition():
        return my[indexMy.leaf1ndex] > 0

    def updateLeaf1ndexConnectee():
        my[indexMy.leafConnectee] = connectionGraph[my[indexMy.dimension1ndex], my[indexMy.leaf1ndex], track[indexTrack.leafBelow, my[indexMy.leafConnectee]]]

    def lolaCondition_initializeUnconstrainedLeaf():
        if my[indexMy.gap1ndex] > 0:
            return int(True)
        return int(False)

    # TODO taskDivisions
    def lolaCondition_initializeTaskIndex():
        # NOTE this is more than a mere condition check: it always forces countGaps to execute
        my[indexMy.doCountGaps] = int(True)
        my[indexMy.lolaCondition] = int(my[indexMy.leafConnectee] % the[indexThe.leavesTotal] == the[indexThe.leavesTotal] - 1)

    def initializeUnconstrainedLeaf():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    findGapsInitialization()
                    while my[indexMy.dimension1ndex] <= the[indexThe.dimensionsTotal]:
                        if connectionGraph[my[indexMy.dimension1ndex], my[indexMy.leaf1ndex], my[indexMy.leaf1ndex]] == my[indexMy.leaf1ndex]:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while my[indexMy.leafConnectee] != my[indexMy.leaf1ndex]:
                                my[indexMy.doCountGaps] = int(False)
                                if checkActiveLeafNotEqualToLeavesTotal():
                                    my[indexMy.doCountGaps] = int(True)
                                else:
                                    if checkTaskIndex():
                                        my[indexMy.doCountGaps] = int(True)
                                    else:
                                        if checkComputationDivisions():
                                            my[indexMy.doCountGaps] = int(True)
                                if my[indexMy.doCountGaps]:
                                    countGaps()
                                updateLeaf1ndexConnectee()
                        my[indexMy.dimension1ndex] += 1

                    if my[indexMy.dimensionsUnconstrained] == the[indexThe.dimensionsTotal]:
                        insertUnconstrainedLeaf()

                    my[indexMy.indexMiniGap] = my[indexMy.gap1ndex]
                    while my[indexMy.indexMiniGap] < my[indexMy.gap1ndexLowerBound]:
                        filterCommonGaps()
                        incrementIndexMiniGap()
            while backtrackCondition():
                backtrack()
            if placeLeafCondition():
                placeLeaf()

            if lolaCondition_initializeUnconstrainedLeaf():
                return

    def initializeTaskIndex():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    findGapsInitialization()
                    while my[indexMy.dimension1ndex] <= the[indexThe.dimensionsTotal]:
                        if connectionGraph[my[indexMy.dimension1ndex], my[indexMy.leaf1ndex], my[indexMy.leaf1ndex]] == my[indexMy.leaf1ndex]:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while my[indexMy.leafConnectee] != my[indexMy.leaf1ndex]:
                                my[indexMy.doCountGaps] = int(False)
                                if checkActiveLeafNotEqualToLeavesTotal():
                                    my[indexMy.doCountGaps] = int(True)
                                else:
                                    lolaCondition_initializeTaskIndex()
                                    if my[indexMy.lolaCondition]:
                                        my[indexMy.lolaCondition] = int(False)
                                        return
                                    if checkTaskIndex():
                                        my[indexMy.doCountGaps] = int(True)
                                    else:
                                        if checkComputationDivisions():
                                            my[indexMy.doCountGaps] = int(True)
                                if my[indexMy.doCountGaps]:
                                    countGaps()
                                updateLeaf1ndexConnectee()
                        my[indexMy.dimension1ndex] += 1
                    if my[indexMy.dimensionsUnconstrained] == the[indexThe.dimensionsTotal]:
                        insertUnconstrainedLeaf()
                    my[indexMy.indexMiniGap] = my[indexMy.gap1ndex]
                    while my[indexMy.indexMiniGap] < my[indexMy.gap1ndexLowerBound]:
                        filterCommonGaps()
                        incrementIndexMiniGap()
            while backtrackCondition():
                backtrack()
            if placeLeafCondition():
                placeLeaf()

    def doWhile():
        while checkActiveLeafGreaterThan0():
            if checkActiveLeafIs1orLess() or checkLeafBelowSentinelIs1():
                if checkActiveLeafGreaterThanLeavesTotal():
                    incrementFoldsTotal()
                else:
                    findGapsInitialization()
                    while my[indexMy.dimension1ndex] <= the[indexThe.dimensionsTotal]:
                        if connectionGraph[my[indexMy.dimension1ndex], my[indexMy.leaf1ndex], my[indexMy.leaf1ndex]] == my[indexMy.leaf1ndex]:
                            incrementDimensionsUnconstrained()
                        else:
                            initializeLeaf1ndexConnectee()
                            while my[indexMy.leafConnectee] != my[indexMy.leaf1ndex]:
                                my[indexMy.doCountGaps] = int(False)
                                if checkActiveLeafNotEqualToLeavesTotal():
                                    my[indexMy.doCountGaps] = int(True)
                                else:
                                    if checkTaskIndex():
                                        my[indexMy.doCountGaps] = int(True)
                                    else:
                                        if checkComputationDivisions():
                                            my[indexMy.doCountGaps] = int(True)
                                if my[indexMy.doCountGaps]:
                                    countGaps()
                                updateLeaf1ndexConnectee()
                        my[indexMy.dimension1ndex] += 1
                    if my[indexMy.dimensionsUnconstrained] == the[indexThe.dimensionsTotal]:
                        insertUnconstrainedLeaf()
                    my[indexMy.indexMiniGap] = my[indexMy.gap1ndex]
                    while my[indexMy.indexMiniGap] < my[indexMy.gap1ndexLowerBound]:
                        filterCommonGaps()
                        incrementIndexMiniGap()
            while backtrackCondition():
                backtrack()
            if placeLeafCondition():
                placeLeaf()

    stateStart = Z0Z_outfitFoldings(listDimensions, computationDivisions=computationDivisions, CPUlimit=CPUlimit)
    connectionGraph: Final[numpy.ndarray] = stateStart['connectionGraph']
    foldsTotal = stateStart['foldsTotal']
    mapShape: Final[Tuple] = stateStart['mapShape']
    my = stateStart['my']
    potentialGaps = stateStart['potentialGaps']
    the: Final[numpy.ndarray] = stateStart['the']
    track = stateStart['track']

    # connectionGraph, foldsTotal, my, potentialGaps, the, track = Z0Z_outfitFoldings(listDimensions, computationDivisions)

    my[indexMy.doCountGaps] = int(False)
    my[indexMy.leaf1ndex] = 1

    initializeUnconstrainedLeaf()
    initializeTaskIndex()
    doWhile()

    return numpy.sum(foldsTotal).item()
