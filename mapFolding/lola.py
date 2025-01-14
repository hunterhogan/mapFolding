"""A functional but untenable implementation of the Run Lola Run concept. Untenable because of excessive code duplication."""
from mapFolding import outfitFoldings, indexTrack as t, setCPUlimit, getTaskDivisions
from mapFolding import activeGap1ndex, activeLeaf1ndex, dimension1ndex, dimensionsUnconstrained, gap1ndexLowerBound, indexMiniGap, leaf1ndexConnectee
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
        computationDivisions: whether to divide the computation into tasks.
        CPUlimit: This is only relevant if `computationDivisions` is `True`: it sets whether and how to limit the CPU usage. See notes for details.
        pathJob (prototype): If you set a path in this parameter, instead of computing the job, the function will save the job to `pathJob`. To compute the job, use a different function that can process the saved values. See `mapFolding.lolaRun` (or whatever name I have selected for the module).

    Returns:
        foldsTotal (or pathFilenameState): The total number of distinct ways to fold the map described by `listDimensions`. (Or the path and filename to a datafile that can be used to initiate the computation.)

    If you want to compute a large `foldsTotal`, dividing the computation into tasks is usually a bad idea. Dividing the algorithm into tasks is inherently inefficient: efficient division into tasks means there would be no overlap in the work performed by each task. When dividing this algorithm, the amount of overlap is between 50% and 90% by all tasks: at least 50% of the work done by every task must be done by _all_ tasks. If you improve the computation time, it will only change by -10 to -50% depending on (at the very least) the ratio of the map dimensions and the number of leaves. If an undivided computation would take 10 hours on your computer, for example, the computation will still take at least 5 hours but you might reduce the time to 9 hours. Most of the time, however, you will increase the computation time. If logicalCores >= leavesTotal, it will probably be faster. If logicalCores <= 2 * leavesTotal, it will almost certainly be slower for all map dimensions.

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
    my=track[t.my.value]
    dimensionsTotal: Final[numpy.uint8] = numpy.uint8(len(validatedDimensions))

    connectionGraph: Final[numpy.ndarray] = D
    leavesTotal: Final[numpy.uint8] = numpy.uint8(n)

    my[activeLeaf1ndex] = numpy.uint8(1)

    def backtrack():
        my[activeLeaf1ndex] -= 1
        track[t.leafBelow.value, track[t.leafAbove.value, my[activeLeaf1ndex]]] = track[t.leafBelow.value, my[activeLeaf1ndex]]
        track[t.leafAbove.value, track[t.leafBelow.value, my[activeLeaf1ndex]]] = track[t.leafAbove.value, my[activeLeaf1ndex]]

    def countGaps():
        potentialGaps[my[gap1ndexLowerBound]] = my[leaf1ndexConnectee]
        if track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] == 0:
            my[gap1ndexLowerBound] += 1
        track[t.countDimensionsGapped.value, my[leaf1ndexConnectee]] += 1
        return my[gap1ndexLowerBound]

    def filterCommonGaps():
        my[indexMiniGap] = my[activeGap1ndex]
        while my[indexMiniGap] < my[gap1ndexLowerBound]:
            potentialGaps[my[activeGap1ndex]] = potentialGaps[my[indexMiniGap]]
            if track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] == dimensionsTotal - my[dimensionsUnconstrained]:
                my[activeGap1ndex] += 1
            track[t.countDimensionsGapped.value, potentialGaps[my[indexMiniGap]]] = 0
            my[indexMiniGap] += 1

    def placeLeaf():
        my[activeGap1ndex] -= 1
        track[t.leafAbove.value, my[activeLeaf1ndex]] = potentialGaps[my[activeGap1ndex]]
        track[t.leafBelow.value, my[activeLeaf1ndex]] = track[t.leafBelow.value, track[t.leafAbove.value, my[activeLeaf1ndex]]]
        track[t.leafBelow.value, track[t.leafAbove.value, my[activeLeaf1ndex]]] = my[activeLeaf1ndex]
        track[t.leafAbove.value, track[t.leafBelow.value, my[activeLeaf1ndex]]] = my[activeLeaf1ndex]
        track[t.gapRangeStart.value, my[activeLeaf1ndex]] = my[activeGap1ndex]
        my[activeLeaf1ndex] += 1

    def initializeState():
        while my[activeLeaf1ndex] > 0:
            if my[activeLeaf1ndex] <= 1 or track[t.leafBelow.value, 0] == 1:
                my[dimensionsUnconstrained] = 0
                my[gap1ndexLowerBound] = track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]
                my[dimension1ndex] = 1
                while my[dimension1ndex] <= dimensionsTotal:
                    if connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]] == my[activeLeaf1ndex]:
                        my[dimensionsUnconstrained] += 1
                    else:
                        my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], my[activeLeaf1ndex]]
                        while my[leaf1ndexConnectee] != my[activeLeaf1ndex]:
                            if not my[activeLeaf1ndex] != leavesTotal and my[leaf1ndexConnectee] % leavesTotal == leavesTotal - 1:
                            # if not my[activeLeaf1ndex] != taskDivisions and my[leaf1ndexConnectee] % taskDivisions == taskDivisions - 1:
                                return
                            my[gap1ndexLowerBound] = countGaps()
                            my[leaf1ndexConnectee] = connectionGraph[my[dimension1ndex], my[activeLeaf1ndex], track[t.leafBelow.value, my[leaf1ndexConnectee]]]
                    my[dimension1ndex] += 1
                if my[dimensionsUnconstrained] == dimensionsTotal:
                    indexLeaf = numpy.uint8(0)
                    while indexLeaf < my[activeLeaf1ndex]:
                        potentialGaps[my[gap1ndexLowerBound]] = indexLeaf
                        my[gap1ndexLowerBound] += 1
                        indexLeaf += 1
                filterCommonGaps()
            while my[activeLeaf1ndex] > 0 and my[activeGap1ndex] == track[t.gapRangeStart.value, my[activeLeaf1ndex] - 1]:
                backtrack()
            if my[activeLeaf1ndex] > 0:
                placeLeaf()

    initializeState()

    tupleLolaState = (connectionGraph, dimensionsTotal, leavesTotal, potentialGaps, track)
    if pathJob is not None:
        # TODO SSOT for where I am putting information!
        pathRelativeJob = str(listDimensions).replace(' ', '')
        pathJob = pathlib.Path(pathJob, pathRelativeJob)

    return lolaDispatcher(taskDivisions, CPUlimit, pathJob, tupleLolaState)

def saveState(pathState: Union[str, os.PathLike[Any]],
    connectionGraph: numpy.ndarray,
    dimensionsTotal: numpy.uint8,
    leavesTotal: numpy.uint8,
    potentialGaps: numpy.ndarray,
    track: numpy.ndarray):
    pathFilenameState = pathlib.Path(pathState, "stateJob.npz")
    pathFilenameState.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez_compressed(pathFilenameState,
        connectionGraph=connectionGraph,
        dimensionsTotal=dimensionsTotal,
        leavesTotal=leavesTotal,
        potentialGaps=potentialGaps,
        track=track
    )
    return pathFilenameState

def loadState(pathFilenameState: Union[str, os.PathLike[Any]]) -> Tuple:
    tupleLolaState = (
        numpy.load(pathFilenameState)['connectionGraph'],
        numpy.uint8(numpy.load(pathFilenameState)['dimensionsTotal']),
        numpy.uint8(numpy.load(pathFilenameState)['leavesTotal']),
        numpy.load(pathFilenameState)['potentialGaps'],
        numpy.load(pathFilenameState)['track']
    )
    return tupleLolaState

def doJob(pathFilenameState: Union[str, os.PathLike[Any]], computationDivisions: bool = False, CPUlimit: Optional[Union[int, float, bool]] = None):
    taskDivisions = getTaskDivisions(computationDivisions)
    # TODO figure out how to pass state information around. Am I having fun yet?!
    tupleLolaState=loadState(pathFilenameState)
    foldsTotal = lolaDispatcher(taskDivisions, CPUlimit, pathJob=None, tupleLolaState=tupleLolaState)
    print(foldsTotal)
    # TODO SSOT for where I am putting information, ffs!
    filenameFoldsTotal = str(tupleLolaState[2]).replace(' ', '') + ".foldsTotal"
    pathFilenameFoldsTotal = pathlib.Path(pathFilenameState).with_name(filenameFoldsTotal)
    pathFilenameFoldsTotal.write_text(str(foldsTotal))
    return pathFilenameFoldsTotal

def lolaDispatcher(computationDivisions, CPUlimit, pathJob, tupleLolaState):
    if pathJob is not None:
        return saveState(pathJob, *tupleLolaState)
    elif computationDivisions:
        # NOTE `set_num_threads` only affects "jitted" functions that have _not_ yet been "imported"
        setCPUlimit(CPUlimit)
        # taskDivisions = numpy.uint8(min(numba.get_num_threads(), tupleLolaState[2])) # leavesTotal
        """TODO something is wrong with `taskDivisions = numpy.uint8(min(numba.get_num_threads(), tupleLolaState[2])) # leavesTotal` but setting `taskDivisions = numpy.uint8(tupleLolaState[2]) # leavesTotal` still works as expected. Therefore, while restructuring the entire computationDivisions-to-taskDivisions system, look for the reason(s) why taskDivisions != leavesTotal if leavesTotal < numba.get_num_threads() calculates incorrectly."""
        taskDivisions = numpy.uint8(tupleLolaState[2]) # leavesTotal
        from mapFolding.lolaDispatcher import doWhileConcurrent
        foldsTotal = int(doWhileConcurrent(*tupleLolaState, taskDivisions)) # type: ignore
    else:
        from mapFolding.lolaDispatcher import doWhile
        foldsTotal = int(doWhile(*tupleLolaState))
    return foldsTotal
