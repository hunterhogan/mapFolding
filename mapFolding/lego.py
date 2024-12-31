from mapFolding import outfitFoldings, validateTaskDivisions
from typing import List, Final, Tuple
import numba
import numpy

leafAbove = 0
leafBelow = 1
countDimensionsGapped = 2
gapRangeStart = 3

@numba.jit(cache=True, fastmath=False)
def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    listDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions)
    computationDivisions, computationIndex = validateTaskDivisions(computationDivisions, computationIndex, leavesTotal)

    dimensionsTotal: int = len(listDimensions)

    foldingsTotal = countFoldings(
        track, potentialGaps, connectionGraph, leavesTotal, dimensionsTotal,
        computationDivisions, computationIndex)

    return foldingsTotal

@numba.njit(cache=True, fastmath=False)
def countFoldings(
    track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
    potentialGaps: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
    D: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]],
    n: int,
    d: int,
    computationDivisions: int,
    computationIndex: int,
    ) -> int:

    connectionGraph: Final = D
    leavesTotal: Final = n
    dimensionsTotal: Final = d
    taskDivisions: Final = computationDivisions
    # NOTE don't forget about this new `Final` type thing
    taskIndex: Final = computationIndex
    
    foldingsTotal: int = 0
    activeLeaf1ndex: int = 1
    activeGap1ndex: int = 0

    def countGaps(gap1ndexLowerBound: int, leaf1ndexConnectee: int) -> int:
        if taskDivisions == 0 or activeLeaf1ndex != taskDivisions or leaf1ndexConnectee % taskDivisions == taskIndex:
            potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
            if track[countDimensionsGapped][leaf1ndexConnectee] == 0:
                gap1ndexLowerBound += 1
            track[countDimensionsGapped][leaf1ndexConnectee] += 1
        return gap1ndexLowerBound

    def inspectConnectees(gap1ndexLowerBound: int, dimension1ndex: int) -> int:
        leaf1ndexConnectee: int = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
        while leaf1ndexConnectee != activeLeaf1ndex:
            gap1ndexLowerBound = countGaps(gap1ndexLowerBound, leaf1ndexConnectee)
            leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][track[leafBelow][leaf1ndexConnectee]]
        return gap1ndexLowerBound

    def findGaps() -> Tuple[int, int]:
        nonlocal activeGap1ndex

        dimensionsUnconstrained: int = 0
        gap1ndexLowerBound: int = track[gapRangeStart][activeLeaf1ndex - 1]
        activeGap1ndex = gap1ndexLowerBound
        dimension1ndex = 1 

        while dimension1ndex <= dimensionsTotal:
            if connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex:
                dimensionsUnconstrained += 1
            else:
                gap1ndexLowerBound = inspectConnectees(gap1ndexLowerBound, dimension1ndex)
            dimension1ndex += 1

        return dimensionsUnconstrained, gap1ndexLowerBound

    def insertUnconstrainedLeaf(unconstrainedCount: int, gapNumberLowerBound: int) -> int:
        # NOTE I suspect this is really an initialization function that should not be in the main loop
        """If activeLeaf1ndex is unconstrained in all dimensions, it can be inserted anywhere"""
        if unconstrainedCount == dimensionsTotal:
            for index in range(activeLeaf1ndex):
                potentialGaps[gapNumberLowerBound] = index
                gapNumberLowerBound += 1
        return gapNumberLowerBound

    def filterCommonGaps(unconstrainedCount, gapNumberLowerBound) -> None:
        nonlocal activeGap1ndex
        for indexMiniGap in range(activeGap1ndex, gapNumberLowerBound):
            potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
            if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == dimensionsTotal - unconstrainedCount:
                activeGap1ndex += 1
            track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0

    def backtrack() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart][activeLeaf1ndex - 1]:
            activeLeaf1ndex -= 1
            track[leafBelow][track[leafAbove][activeLeaf1ndex]] = track[leafBelow][activeLeaf1ndex]
            track[leafAbove][track[leafBelow][activeLeaf1ndex]] = track[leafAbove][activeLeaf1ndex]

    def placeLeaf() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        activeGap1ndex -= 1
        track[leafAbove][activeLeaf1ndex] = potentialGaps[activeGap1ndex]
        track[leafBelow][activeLeaf1ndex] = track[leafBelow][track[leafAbove][activeLeaf1ndex]]
        track[leafBelow][track[leafAbove][activeLeaf1ndex]] = activeLeaf1ndex
        track[leafAbove][track[leafBelow][activeLeaf1ndex]] = activeLeaf1ndex
        track[gapRangeStart][activeLeaf1ndex] = activeGap1ndex
        activeLeaf1ndex += 1

    while activeLeaf1ndex > 0:
        if activeLeaf1ndex <= 1 or track[leafBelow][0] == 1:
            if activeLeaf1ndex > leavesTotal:
                foldingsTotal += leavesTotal
            else:
                dimensionsUnconstrained, gap1ndexLowerBound = findGaps()
                gap1ndexLowerBound = insertUnconstrainedLeaf(dimensionsUnconstrained, gap1ndexLowerBound)
                filterCommonGaps(dimensionsUnconstrained, gap1ndexLowerBound)

        backtrack()
        if activeLeaf1ndex > 0:
            placeLeaf()

    return foldingsTotal
