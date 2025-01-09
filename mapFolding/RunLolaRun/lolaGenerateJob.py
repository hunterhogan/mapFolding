from mapFolding import outfitFoldings
from .lolaBeDRY import countFoldsTask, leafBelow, leafAbove, countDimensionsGapped, gapRangeStart
from typing import List, Final, Literal, Any, Union, TypedDict
import numpy
import pathlib
import os
import pickle

def makeJob(listDimensions: List[int], pathJobs: Union[str, os.PathLike[Any]] = "jobs"):

    dtypeDefault: Final = numpy.uint8
    dtypeMaximum: Final = numpy.uint16

    listDimensions, n, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions, dtypeDefault, dtypeMaximum)
    leavesTotal: Final[numpy.uint8] = numpy.uint8(n)
    dimensionsTotal: Final[numpy.uint8] = numpy.uint8(len(listDimensions))

    activeLeaf1ndex: numpy.uint8 = numpy.uint8(1)
    activeGap1ndex: numpy.uint8 = numpy.uint8(0)
    # activeLeaf1ndex: int = 1
    # activeGap1ndex: int = 0

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
        while activeLeaf1ndex > 0:
            if activeLeaf1ndex <= 1 or track[leafBelow, 0] == 1:
                dimensionsUnconstrained: numpy.uint8 = numpy.uint8(0)
                gap1ndexLowerBound: int = track[gapRangeStart, activeLeaf1ndex - 1]
                dimension1ndex: numpy.uint8 = numpy.uint8(1)
                while dimension1ndex <= dimensionsTotal:
                    if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                        dimensionsUnconstrained += 1
                    else:
                        leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                        while leaf1ndexConnectee != activeLeaf1ndex:
                            if not activeLeaf1ndex != leavesTotal and leaf1ndexConnectee % leavesTotal == leavesTotal - 1:
                                return
                            gap1ndexLowerBound = countGaps(gap1ndexLowerBound, leaf1ndexConnectee)
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                    dimension1ndex += 1
                if dimensionsUnconstrained == dimensionsTotal:
                    indexLeaf: numpy.uint8 = numpy.uint8(0)
                    while indexLeaf < activeLeaf1ndex:
                        potentialGaps[gap1ndexLowerBound] = indexLeaf
                        gap1ndexLowerBound += 1
                        indexLeaf += 1
                filterCommonGaps(dimensionsUnconstrained, gap1ndexLowerBound)
            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart, activeLeaf1ndex - 1]:
                backtrack()
            if activeLeaf1ndex > 0:
                placeLeaf()

    initializeState()

    pathJobBase = pathlib.Path(pathJobs).resolve()
    pathJobWriteStream = pathJobBase / f"[{','.join(map(str, listDimensions))}]"
    pathJobWriteStream.mkdir(parents=True, exist_ok=True)

    for index in list(range(leavesTotal)):
        dictionaryState = countFoldsTask(
            activeGap1ndex=activeGap1ndex,
            activeLeaf1ndex=activeLeaf1ndex,
            connectionGraph=connectionGraph,
            countDimensionsGapped=countDimensionsGapped,
            dimensionsTotal=dimensionsTotal,
            foldsTotal=0,
            gapRangeStart=gapRangeStart,
            leafAbove=leafAbove,
            leafBelow=leafBelow,
            leavesTotal=leavesTotal,
            listDimensions=listDimensions,
            potentialGaps=potentialGaps,
            taskIndex=numpy.uint8(index),
            track=track
        )
        zeroPadding = len(str(leavesTotal))
        pathFilenameJobWriteStream = pathlib.Path(pathJobWriteStream, f"{str(index).zfill(zeroPadding)}.pkl").write_bytes(pickle.dumps(dictionaryState))
