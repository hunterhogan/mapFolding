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

    def countGaps(gap1ndexLowerBound, leaf1ndexConnectee):
        if computationDivisions == False or activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
            potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
            if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
                gap1ndexLowerBound += 1
            track[countDimensionsGapped, leaf1ndexConnectee] += 1
        return gap1ndexLowerBound

    def inspectConnectees(gap1ndexLowerBound, dimension1ndex):
        leaf1ndexConnectee: int = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
        while leaf1ndexConnectee != activeLeaf1ndex:
            gap1ndexLowerBound = countGaps(gap1ndexLowerBound, leaf1ndexConnectee)
            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
        return gap1ndexLowerBound

    def findGaps():
        nonlocal activeGap1ndex

        dimensionsUnconstrained: int = 0
        gap1ndexLowerBound: int = track[gapRangeStart, activeLeaf1ndex - 1]
        activeGap1ndex = gap1ndexLowerBound
        dimension1ndex: int = 1

        while dimension1ndex <= dimensionsTotal:
            if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                dimensionsUnconstrained += 1
            else:
                gap1ndexLowerBound = inspectConnectees(gap1ndexLowerBound, dimension1ndex)
            dimension1ndex += 1

        return dimensionsUnconstrained, gap1ndexLowerBound

    def insertUnconstrainedLeaf(unconstrainedCount, gapNumberLowerBound):
        if unconstrainedCount == dimensionsTotal:
            indexLeaf: int = 0
            while indexLeaf < activeLeaf1ndex:
                potentialGaps[gapNumberLowerBound] = indexLeaf
                gapNumberLowerBound += 1
                indexLeaf += 1
        return gapNumberLowerBound

    def filterCommonGaps(unconstrainedCount, gapNumberLowerBound) -> None:
        nonlocal activeGap1ndex
        indexMiniGap = activeGap1ndex
        while indexMiniGap < gapNumberLowerBound:
            potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
            if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - unconstrainedCount:
                activeGap1ndex += 1
            track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
            indexMiniGap += 1

    def backtrack() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
            activeLeaf1ndex -= 1
            track[leafBelow, track[leafAbove, activeLeaf1ndex]] = track[leafBelow, activeLeaf1ndex]
            track[leafAbove, track[leafBelow, activeLeaf1ndex]] = track[leafAbove, activeLeaf1ndex]

    def placeLeaf() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        activeGap1ndex -= 1
        track[leafAbove, activeLeaf1ndex] = potentialGaps[activeGap1ndex]
        track[leafBelow, activeLeaf1ndex] = track[leafBelow, track[leafAbove, activeLeaf1ndex]]
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = activeLeaf1ndex
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = activeLeaf1ndex
        track[gapRangeStart, activeLeaf1ndex] = activeGap1ndex
        activeLeaf1ndex += 1

    def doWhile():
        nonlocal foldsTotal
        while activeLeaf1ndex > 0:
            if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
                if activeLeaf1ndex > leavesTotal:
                    foldsTotal += leavesTotal
                else:
                    dimensionsUnconstrained, gap1ndexLowerBound = findGaps()
                    gap1ndexLowerBound = insertUnconstrainedLeaf(dimensionsUnconstrained, gap1ndexLowerBound)
                    filterCommonGaps(dimensionsUnconstrained, gap1ndexLowerBound)

            backtrack()
            if activeLeaf1ndex > 0:
                placeLeaf()

    doWhile()

    return foldsTotal
