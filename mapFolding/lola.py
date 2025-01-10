"""A functional but untenable implementation of the Run Lola Run concept. Untenable because of excessive code duplication."""
from mapFolding import outfitFoldings, leafAbove, leafBelow, gapRangeStart, countDimensionsGapped, setCPUlimit, getTaskDivisions
from typing import Any, Final, List, Optional, Tuple, Union
import numpy
import numpy.typing
import pathlib
import os

def countFolds(listDimensions: List[int], computationDivisions: bool = False, CPUlimit: Optional[Union[int, float, bool]] = None, pathJob: Optional[Union[str, os.PathLike[Any]]] = None):
    """
    Count the distinct ways to fold a multi-dimensional map.
    Parameters:
        listDimensions: list of integers, the dimensions of the map.
        computationDivisions: whether to divide the computation into tasks. Dividing into tasks is NEVER* faster and is often many times slower. (*: that I have seen)
        CPUlimit: This is only relevant if `computationDivisions` is `True`. It sets whether and how to limit the CPU usage. See notes for details.
        pathJob: If you pass a path, instead of computing the job, the function will save the job to that path. To compute the job, use a different function that can process the saved values. See `mapFolding.lolaOne`.

    Limits on CPU usage `CPUlimit`
        - `False`, `None`, or `0`: No limits on CPU usage; uses all available CPUs. All other values will potentially limit CPU usage.
        - `True`: Yes, limit the CPU usage; limits to 1 CPU.
        - Integer `>= 1`: Limits usage to the specified number of CPUs.
        - Decimal value (`float`) between 0 and 1: Fraction of total CPUs to use.
        - Decimal value (`float`) between -1 and 0: Fraction of CPUs to *not* use.
        - Integer `<= -1`: Subtract the absolute value from total CPUs.
    """

    taskDivisions = getTaskDivisions(computationDivisions)

    dtypeDefault: Final = numpy.uint8
    dtypeMaximum: Final = numpy.uint16

    validatedDimensions, n, D, track, potentialGaps = outfitFoldings(listDimensions, dtypeDefault, dtypeMaximum)

    dimensionsTotal: Final[numpy.uint8] = numpy.uint8(len(validatedDimensions))

    connectionGraph: Final[numpy.ndarray] = D
    leavesTotal: Final[numpy.uint8] = numpy.uint8(n)

    activeGap1ndex = numpy.uint8(0)
    activeLeaf1ndex = numpy.uint8(1)

    def backtrack():
        nonlocal activeLeaf1ndex
        activeLeaf1ndex -= 1
        track[leafBelow, track[leafAbove, activeLeaf1ndex]] = track[leafBelow, activeLeaf1ndex]
        track[leafAbove, track[leafBelow, activeLeaf1ndex]] = track[leafAbove, activeLeaf1ndex]

    def countGaps(gap1ndexLowerBound: numpy.uint8, leaf1ndexConnectee: numpy.uint8):
        potentialGaps[gap1ndexLowerBound] = leaf1ndexConnectee
        if track[countDimensionsGapped, leaf1ndexConnectee] == 0:
            gap1ndexLowerBound += 1
        track[countDimensionsGapped, leaf1ndexConnectee] += 1
        return gap1ndexLowerBound

    def filterCommonGaps(dimensionsUnconstrained: numpy.uint8, gap1ndexLowerBound: numpy.uint8):
        nonlocal activeGap1ndex
        indexMiniGap = activeGap1ndex
        while indexMiniGap < gap1ndexLowerBound:
            potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
            if track[countDimensionsGapped, potentialGaps[indexMiniGap]] == dimensionsTotal - dimensionsUnconstrained:
                activeGap1ndex += 1
            track[countDimensionsGapped, potentialGaps[indexMiniGap]] = 0
            indexMiniGap += 1

    def placeLeaf():
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
                dimensionsUnconstrained = numpy.uint8(0)
                gap1ndexLowerBound: numpy.uint8 = track[gapRangeStart, activeLeaf1ndex - 1]
                dimension1ndex = numpy.uint8(1)
                while dimension1ndex <= dimensionsTotal:
                    if connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex] == activeLeaf1ndex:
                        dimensionsUnconstrained += 1
                    else:
                        leaf1ndexConnectee: numpy.uint8 = connectionGraph[dimension1ndex, activeLeaf1ndex, activeLeaf1ndex]
                        while leaf1ndexConnectee != activeLeaf1ndex:
                            if not activeLeaf1ndex != leavesTotal and leaf1ndexConnectee % leavesTotal == leavesTotal - 1:
                                return
                            gap1ndexLowerBound = countGaps(gap1ndexLowerBound, leaf1ndexConnectee)
                            leaf1ndexConnectee = connectionGraph[dimension1ndex, activeLeaf1ndex, track[leafBelow, leaf1ndexConnectee]]
                    dimension1ndex += 1
                if dimensionsUnconstrained == dimensionsTotal:
                    indexLeaf = numpy.uint8(0)
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

    tupleLolaState = (activeGap1ndex, activeLeaf1ndex, connectionGraph, dimensionsTotal, leavesTotal, potentialGaps, track)
    if pathJob is not None:
        pathJob = pathlib.Path(pathJob, f"[{','.join(map(str, listDimensions))}]")

    return lolaDispatcher(taskDivisions, CPUlimit, pathJob, tupleLolaState)

def saveState(pathState: Union[str, os.PathLike[Any]],
    activeGap1ndex: numpy.uint8,
    activeLeaf1ndex: numpy.uint8,
    connectionGraph: numpy.ndarray,
    dimensionsTotal: numpy.uint8,
    leavesTotal: numpy.uint8,
    potentialGaps: numpy.ndarray,
    track: numpy.ndarray):
    import pathlib
    pathFilenameState = pathlib.Path(pathState, "stateJob.npz")
    pathFilenameState.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez_compressed(pathFilenameState,
        activeGap1ndex=activeGap1ndex,
        activeLeaf1ndex=activeLeaf1ndex,
        connectionGraph=connectionGraph,
        dimensionsTotal=dimensionsTotal,
        leavesTotal=leavesTotal,
        potentialGaps=potentialGaps,
        track=track
    )
    return pathFilenameState

def loadState(pathFilenameState: Union[str, os.PathLike[Any]]) -> Tuple:
    tupleLolaState = (
        numpy.uint8(numpy.load(pathFilenameState)['activeGap1ndex']),
        numpy.uint8(numpy.load(pathFilenameState)['activeLeaf1ndex']),
        numpy.load(pathFilenameState)['connectionGraph'],
        numpy.uint8(numpy.load(pathFilenameState)['dimensionsTotal']),
        numpy.uint8(numpy.load(pathFilenameState)['leavesTotal']),
        numpy.load(pathFilenameState)['potentialGaps'],
        numpy.load(pathFilenameState)['track']
    )
    return tupleLolaState

def doJob(pathFilenameState: Union[str, os.PathLike[Any]], computationDivisions: bool = False, CPUlimit: Optional[Union[int, float, bool]] = None, gpu: bool = False):
    taskDivisions = getTaskDivisions(computationDivisions)
    foldsTotal = lolaDispatcher(taskDivisions, CPUlimit, pathJob=None, tupleLolaState=loadState(pathFilenameState), gpu=gpu)
    print(foldsTotal)
    pathFilenameFoldsTotal = pathlib.Path(pathFilenameState).with_name("foldsTotal.txt")
    pathFilenameFoldsTotal.write_text(str(foldsTotal))
    return pathFilenameFoldsTotal

def lolaDispatcher(taskDivisions, CPUlimit, pathJob, tupleLolaState, gpu: Optional[bool] = False):
    if pathJob is not None:
        return saveState(pathJob, *tupleLolaState)
    elif gpu:
        from mapFolding.lolaGPU import doWhileGPU
        foldsTotal = int(doWhileGPU(*tupleLolaState))
    elif taskDivisions:
        # NOTE `set_num_threads` only affects "jitted" functions that have _not_ yet been "imported"
        setCPUlimit(CPUlimit)
        from mapFolding.lolaTask import doWhileConcurrent
        foldsTotal = int(doWhileConcurrent(*tupleLolaState))
    else:
        from mapFolding.lolaOne import doWhileOne
        foldsTotal = int(doWhileOne(*tupleLolaState))
    return foldsTotal
