from .lolaBeDRY import countFoldsTask
from typing import List, Final, Literal, Any, Union
import numba
import numba.extending
import numba.typed
import numba.types
import numpy
import os
import pathlib
import pickle

def doCountFoldsTask(pathFilenameTask: Union[str, os.PathLike[Any]]):

    dictionaryState: countFoldsTask = pickle.loads(pathlib.Path(pathFilenameTask).read_bytes())
    activeGap1ndex = dictionaryState["activeGap1ndex"]
    activeLeaf1ndex = dictionaryState["activeLeaf1ndex"]
    connectionGraph = dictionaryState["connectionGraph"]
    countDimensionsGapped = dictionaryState["countDimensionsGapped"]
    dimensionsTotal = dictionaryState["dimensionsTotal"]
    foldsSubTotal = dictionaryState["foldsTotal"]
    gapRangeStart = dictionaryState["gapRangeStart"]
    leafAbove = dictionaryState["leafAbove"]
    leafBelow = dictionaryState["leafBelow"]
    leavesTotal = dictionaryState["leavesTotal"]
    listDimensions = dictionaryState["listDimensions"]
    potentialGaps = dictionaryState["potentialGaps"]
    taskIndex = dictionaryState["taskIndex"]
    track = dictionaryState["track"]

    connectionGraph: Final[numpy.ndarray]
    countDimensionsGapped: Final[Literal[2]]
    dimensionsTotal: Final[int]
    gapRangeStart: Final[Literal[3]]
    leafAbove: Final[Literal[0]]
    leafBelow: Final[Literal[1]]
    listDimensions: Final[List[int]]
    taskIndex: Final[int]
    foldsSubTotal = _doCountFoldsTask(activeGap1ndex, activeLeaf1ndex, connectionGraph, countDimensionsGapped, dimensionsTotal, foldsSubTotal, gapRangeStart, leafAbove, leafBelow, leavesTotal, listDimensions, potentialGaps, taskIndex, track)
    pathFilenameFoldsSubTotal = pathlib.Path(pathFilenameTask).with_suffix(".foldsSubTotal").write_text(str(foldsSubTotal))

@numba.jit(nopython=True, cache=True, fastmath=False)
def _doCountFoldsTask(activeGap1ndex, activeLeaf1ndex, connectionGraph, countDimensionsGapped, dimensionsTotal, foldsSubTotal, gapRangeStart, leafAbove, leafBelow, leavesTotal, listDimensions, potentialGaps, taskIndex, track):

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

    def doWhile():
        nonlocal foldsSubTotal
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
                                if activeLeaf1ndex != leavesTotal or leaf1ndexConnectee % leavesTotal == taskIndex:
                                    gap1ndexLowerBound = countGaps(gap1ndexLowerBound, leaf1ndexConnectee)
                                leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                        dimension1ndex += 1
                    filterCommonGaps(dimensionsUnconstrained, gap1ndexLowerBound)
            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()
            if activeLeaf1ndex > 0:
                placeLeaf()

    doWhile()

    return foldsSubTotal