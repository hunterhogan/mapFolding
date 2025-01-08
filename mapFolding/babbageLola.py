from typing import Any, Final, Literal
import numba
import numpy
import numpy.typing

@numba.jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _countFolds(n: int, d: int, D: numpy.typing.NDArray[numpy.integer[Any]], taskDivisions: bool,
    track: numpy.typing.NDArray[numpy.integer[Any]], potentialGaps: numpy.typing.NDArray[numpy.integer[Any]],
    )-> int:

    leavesTotal: Final[int] = n
    dimensionsTotal: Final[int] = d
    connectionGraph: Final[numpy.ndarray] = D
    computationDivisions: Final[bool] = taskDivisions

    leafAbove: Final[Literal[0]] = 0
    leafBelow: Final[Literal[1]] = 1
    countDimensionsGapped: Final[Literal[2]] = 2
    gapRangeStart: Final[Literal[3]] = 3

    taskIndex: int = 0

    activeLeaf1ndex: int = 1
    activeGap1ndex: int = 0

    def backtrack() -> None:
        nonlocal activeLeaf1ndex
        activeLeaf1ndex -= 1
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = track[leafBelow, activeLeaf1ndex]
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = track[leafAbove, activeLeaf1ndex]

    def countGaps(gap1ndexLowerBound: int, leaf1ndexConnectee: int) -> int:
        potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
        if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
            gap1ndexLowerBound += 1
        track[countDimensionsGapped, leaf1ndexConnectee] += 1
        return gap1ndexLowerBound

    def filterCommonGaps(dimensionsUnconstrained: int, gap1ndexLowerBound: int) -> None:
        nonlocal activeGap1ndex
        indexMiniGap = activeGap1ndex
        while indexMiniGap < gap1ndexLowerBound:
            potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
            if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                activeGap1ndex += 1
            track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
            indexMiniGap += 1

    def placeLeaf() -> None:
        nonlocal activeLeaf1ndex, activeGap1ndex
        activeGap1ndex -= 1
        track[leafAbove, activeLeaf1ndex] = potentialGaps[activeGap1ndex]
        track[leafBelow, activeLeaf1ndex] = track[leafBelow, track[leafAbove, activeLeaf1ndex]]
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = activeLeaf1ndex
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = activeLeaf1ndex
        track[gapRangeStart, activeLeaf1ndex] = activeGap1ndex
        activeLeaf1ndex += 1

    def initializeState() -> None:
        while activeLeaf1ndex > 0:
            if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
                dimensionsUnconstrained: int = 0
                gap1ndexLowerBound: int = track[gapRangeStart, activeLeaf1ndex - 1]
                dimension1ndex: int = 1
                while dimension1ndex <= dimensionsTotal:
                    if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                        dimensionsUnconstrained += 1
                    else:
                        leaf1ndexConnectee: int = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                        while leaf1ndexConnectee != activeLeaf1ndex:
                            if not activeLeaf1ndex != leavesTotal and leaf1ndexConnectee % leavesTotal == leavesTotal - 1:
                                return
                            gap1ndexLowerBound = countGaps(gap1ndexLowerBound, leaf1ndexConnectee)
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                    dimension1ndex += 1
                if dimensionsUnconstrained == dimensionsTotal:
                    indexLeaf: int = 0
                    while indexLeaf < activeLeaf1ndex:
                        potentialGaps[gap1ndexLowerBound] = indexLeaf
                        gap1ndexLowerBound += 1
                        indexLeaf += 1
                filterCommonGaps(dimensionsUnconstrained, gap1ndexLowerBound)
            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()
            if activeLeaf1ndex > 0:
                placeLeaf()

    def doWhile():
        nonlocal taskIndex
        foldsSubTotal: int = 0
        while activeLeaf1ndex > 0:
            if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
                if activeLeaf1ndex > leavesTotal:
                    foldsSubTotal += leavesTotal
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
                    filterCommonGaps(dimensionsUnconstrained, gap1ndexLowerBound)
            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()
            if activeLeaf1ndex > 0:
                placeLeaf()
        return foldsSubTotal

    initializeState()

    if computationDivisions:
        foldsTotal: int = 0
        state_activeGap1ndex = activeGap1ndex
        state_activeLeaf1ndex = activeLeaf1ndex
        state_potentialGaps = potentialGaps.copy()
        state_track = track.copy()
        for taskIndex in numba.prange(leavesTotal):
            activeGap1ndex = state_activeGap1ndex
            activeLeaf1ndex = state_activeLeaf1ndex
            potentialGaps = state_potentialGaps.copy()
            track = state_track.copy()
            foldsTotal += doWhile()
    else:
        foldsTotal = doWhile()

    return foldsTotal
