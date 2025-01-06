from mapFolding import outfitFoldings
from typing import List, Final, Literal
import numpy
import numba

@numba.jit(nopython=True, cache=True, fastmath=False)
def countFolds(listDimensions: List[int], computationDivisions: bool = False):

    dtypeDefault: Final = numpy.uint8
    dtypeMaximum: Final = numpy.uint16

    listDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions, dtypeDefault, dtypeMaximum)

    dimensionsTotal: Final[int] = len(listDimensions)

    leafAbove: Final[Literal[0]] = 0
    leafBelow: Final[Literal[1]] = 1
    countDimensionsGapped: Final[Literal[2]] = 2
    gapRangeStart: Final[Literal[3]] = 3

    taskIndex: int = 0

    activeLeaf1ndex: int = 1
    activeGap1ndex: int = 0

    foldsTotal: int = 0

    def incrementFoldsTotal():
        nonlocal foldsTotal
        foldsTotal += leavesTotal

    def insertUnconstrainedLeaf(gap1ndexLowerBound):
        indexLeaf: int = 0
        while indexLeaf < activeLeaf1ndex:
            potentialGaps[gap1ndexLowerBound] = indexLeaf
            gap1ndexLowerBound += 1
            indexLeaf += 1
        return gap1ndexLowerBound

    def filterCommonGaps(dimensionsUnconstrained, gap1ndexLowerBound) -> None:
        nonlocal activeGap1ndex
        indexMiniGap = activeGap1ndex
        while indexMiniGap < gap1ndexLowerBound:
            potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
            if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                activeGap1ndex += 1
            track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
            indexMiniGap += 1

    def backtrack() -> None:
        nonlocal activeLeaf1ndex
        activeLeaf1ndex -= 1
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = track[leafBelow, activeLeaf1ndex]
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = track[leafAbove, activeLeaf1ndex]

    def countGaps(gap1ndexLowerBound, leaf1ndexConnectee):
        potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
        if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
            gap1ndexLowerBound += 1
        track[countDimensionsGapped, leaf1ndexConnectee] += 1
        return gap1ndexLowerBound

    def placeLeaf() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        activeGap1ndex -= 1
        track[leafAbove, activeLeaf1ndex] = potentialGaps[activeGap1ndex]
        track[leafBelow, activeLeaf1ndex] = track[leafBelow, track[leafAbove, activeLeaf1ndex]]
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = activeLeaf1ndex
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = activeLeaf1ndex
        track[gapRangeStart, activeLeaf1ndex] = activeGap1ndex
        activeLeaf1ndex += 1

    def initializeState():
        nonlocal foldsTotal
        while activeLeaf1ndex > 0:
            if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
                if activeLeaf1ndex > leavesTotal:
                    incrementFoldsTotal()
                else:
                    dimensionsUnconstrained: int = 0
                    gap1ndexLowerBound: int = track[gapRangeStart, activeLeaf1ndex - 1]
                    dimension1ndex: int = 1
                    while dimension1ndex <= dimensionsTotal:
                        if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                            dimensionsUnconstrained += 1
                        else:
                            leaf1ndexConnectee: int = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                            while leaf1ndexConnectee != activeLeaf1ndex:
                                # if computationDivisions == False or activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
                                gap1ndexLowerBound = countGaps(gap1ndexLowerBound, leaf1ndexConnectee)
                                leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                        dimension1ndex += 1
                    if dimensionsUnconstrained == dimensionsTotal:
                        gap1ndexLowerBound = insertUnconstrainedLeaf(gap1ndexLowerBound)
                    filterCommonGaps(dimensionsUnconstrained, gap1ndexLowerBound)
            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()
            if activeLeaf1ndex > 0:
                placeLeaf()
            if activeGap1ndex > 0:
                return

    def doWhile():
        nonlocal foldsTotal
        while activeLeaf1ndex > 0:
            if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
                if activeLeaf1ndex > leavesTotal:
                    incrementFoldsTotal()
                else:
                    dimensionsUnconstrained: int = 0
                    gap1ndexLowerBound: int = track[gapRangeStart, activeLeaf1ndex - 1]
                    dimension1ndex: int = 1
                    while dimension1ndex <= dimensionsTotal:
                        if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                            dimensionsUnconstrained += 1
                        else:
                            leaf1ndexConnectee: int = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                            while leaf1ndexConnectee != activeLeaf1ndex:
                                if computationDivisions == False or activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
                                    gap1ndexLowerBound = countGaps(gap1ndexLowerBound, leaf1ndexConnectee)
                                leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                        dimension1ndex += 1
                    if dimensionsUnconstrained == dimensionsTotal:
                        gap1ndexLowerBound = insertUnconstrainedLeaf(gap1ndexLowerBound)
                    filterCommonGaps(dimensionsUnconstrained, gap1ndexLowerBound)
            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()
            if activeLeaf1ndex > 0:
                placeLeaf()

    initializeState()

    if computationDivisions:
        state_activeGap1ndex = activeGap1ndex
        state_activeLeaf1ndex = activeLeaf1ndex
        state_potentialGaps = potentialGaps.copy()
        state_track = track.copy()
        listTaskIndices = list(range(leavesTotal))
        for taskIndex in listTaskIndices:
            activeGap1ndex = state_activeGap1ndex
            activeLeaf1ndex = state_activeLeaf1ndex
            potentialGaps = state_potentialGaps.copy()
            track = state_track.copy()
            doWhile()
    else:
        doWhile()

    return foldsTotal
