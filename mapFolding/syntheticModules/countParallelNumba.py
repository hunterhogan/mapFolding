from __future__ import annotations

from concurrent.futures import Future as ConcurrentFuture, ProcessPoolExecutor
from copy import deepcopy
from mapFolding.dataBaskets import (
	Array1DElephino, Array1DLeavesTotal, Array3DLeavesTotal, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal,
	ParallelMapFoldingState)
from multiprocessing import set_start_method as multiprocessing_set_start_method
from numba import jit

if __name__ == '__main__':
    multiprocessing_set_start_method('spawn')

@jit(cache=True, error_model='numpy', fastmath=True, forceinline=True)
def count(groupsOfFolds: DatatypeFoldsTotal, gap1ndex: DatatypeElephino, gap1ndexCeiling: DatatypeElephino, 次Dimension: DatatypeLeavesTotal, 次Leaf: DatatypeLeavesTotal, 次MiniGap: DatatypeElephino, leaf1ndex: DatatypeLeavesTotal, leafConnectee: DatatypeLeavesTotal, dimensionsUnconstrained: DatatypeLeavesTotal, countDimensionsGapped: Array1DLeavesTotal, gapRangeStart: Array1DElephino, gapsWhere: Array1DLeavesTotal, leafAbove: Array1DLeavesTotal, leafBelow: Array1DLeavesTotal, connectionGraph: Array3DLeavesTotal, dimensionsTotal: DatatypeLeavesTotal, leavesTotal: DatatypeLeavesTotal, taskDivisions: DatatypeLeavesTotal, task次: DatatypeLeavesTotal) -> tuple[DatatypeFoldsTotal, DatatypeElephino, DatatypeElephino, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeElephino, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal, Array1DLeavesTotal, Array1DElephino, Array1DLeavesTotal, Array1DLeavesTotal, Array1DLeavesTotal, Array3DLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal]:
    while leaf1ndex > 0:
        if leaf1ndex <= 1 or leafBelow[0] == 1:
            if leaf1ndex > leavesTotal:
                groupsOfFolds += 1
            else:
                dimensionsUnconstrained = dimensionsTotal
                gap1ndexCeiling = gapRangeStart[leaf1ndex - 1]
                次Dimension = 0
                while 次Dimension < dimensionsTotal:
                    leafConnectee = connectionGraph[次Dimension, leaf1ndex, leaf1ndex]
                    if leafConnectee == leaf1ndex:
                        dimensionsUnconstrained -= 1
                    else:
                        while leafConnectee != leaf1ndex:
                            if leaf1ndex != taskDivisions or leafConnectee % taskDivisions == task次:
                                gapsWhere[gap1ndexCeiling] = leafConnectee
                                if countDimensionsGapped[leafConnectee] == 0:
                                    gap1ndexCeiling += 1
                                countDimensionsGapped[leafConnectee] += 1
                            leafConnectee = connectionGraph[次Dimension, leaf1ndex, leafBelow[leafConnectee]]
                    次Dimension += 1
                if not dimensionsUnconstrained:
                    次Leaf = 0
                    while 次Leaf < leaf1ndex:
                        gapsWhere[gap1ndexCeiling] = 次Leaf
                        gap1ndexCeiling += 1
                        次Leaf += 1
                次MiniGap = gap1ndex
                while 次MiniGap < gap1ndexCeiling:
                    gapsWhere[gap1ndex] = gapsWhere[次MiniGap]
                    if countDimensionsGapped[gapsWhere[次MiniGap]] == dimensionsUnconstrained:
                        gap1ndex += 1
                    countDimensionsGapped[gapsWhere[次MiniGap]] = 0
                    次MiniGap += 1
        while leaf1ndex > 0 and gap1ndex == gapRangeStart[leaf1ndex - 1]:
            leaf1ndex -= 1
            leafBelow[leafAbove[leaf1ndex]] = leafBelow[leaf1ndex]
            leafAbove[leafBelow[leaf1ndex]] = leafAbove[leaf1ndex]
        if leaf1ndex > 0:
            gap1ndex -= 1
            leafAbove[leaf1ndex] = gapsWhere[gap1ndex]
            leafBelow[leaf1ndex] = leafBelow[leafAbove[leaf1ndex]]
            leafBelow[leafAbove[leaf1ndex]] = leaf1ndex
            leafAbove[leafBelow[leaf1ndex]] = leaf1ndex
            gapRangeStart[leaf1ndex] = gap1ndex
            leaf1ndex += 1
    return (groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次Leaf, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal, taskDivisions, task次)

def unRepackParallelMapFoldingState(state: ParallelMapFoldingState) -> ParallelMapFoldingState:
    mapShape: tuple[DatatypeLeavesTotal, ...] = state.mapShape
    groupsOfFolds: DatatypeFoldsTotal = state.groupsOfFolds
    gap1ndex: DatatypeElephino = state.gap1ndex
    gap1ndexCeiling: DatatypeElephino = state.gap1ndexCeiling
    次Dimension: DatatypeLeavesTotal = state.次Dimension
    次Leaf: DatatypeLeavesTotal = state.次Leaf
    次MiniGap: DatatypeElephino = state.次MiniGap
    leaf1ndex: DatatypeLeavesTotal = state.leaf1ndex
    leafConnectee: DatatypeLeavesTotal = state.leafConnectee
    dimensionsUnconstrained: DatatypeLeavesTotal = state.dimensionsUnconstrained
    countDimensionsGapped: Array1DLeavesTotal = state.countDimensionsGapped
    gapRangeStart: Array1DElephino = state.gapRangeStart
    gapsWhere: Array1DLeavesTotal = state.gapsWhere
    leafAbove: Array1DLeavesTotal = state.leafAbove
    leafBelow: Array1DLeavesTotal = state.leafBelow
    connectionGraph: Array3DLeavesTotal = state.connectionGraph
    dimensionsTotal: DatatypeLeavesTotal = state.dimensionsTotal
    leavesTotal: DatatypeLeavesTotal = state.leavesTotal
    taskDivisions: DatatypeLeavesTotal = state.taskDivisions
    task次: DatatypeLeavesTotal = state.task次
    groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次Leaf, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal, taskDivisions, task次 = count(groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次Leaf, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal, taskDivisions, task次)
    state = ParallelMapFoldingState(mapShape=mapShape, groupsOfFolds=groupsOfFolds, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, 次Dimension=次Dimension, 次Leaf=次Leaf, 次MiniGap=次MiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, dimensionsUnconstrained=dimensionsUnconstrained, countDimensionsGapped=countDimensionsGapped, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow, taskDivisions=taskDivisions, task次=task次)
    return state

def doTheNeedful(state: ParallelMapFoldingState, concurrencyLimit: int) -> tuple[int, list[ParallelMapFoldingState]]:
    stateParallel = deepcopy(state)
    boxOfStatesParallel: list[ParallelMapFoldingState] = [stateParallel] * stateParallel.taskDivisions
    groupsOfFoldsTotal: int = 0
    dictionaryConcurrency: dict[int, ConcurrentFuture[ParallelMapFoldingState]] = {}
    with ProcessPoolExecutor(concurrencyLimit) as concurrencyManager:
        for indexSherpa in range(stateParallel.taskDivisions):
            state = deepcopy(stateParallel)
            state.task次 = indexSherpa
            dictionaryConcurrency[indexSherpa] = concurrencyManager.submit(unRepackParallelMapFoldingState, state)
        for indexSherpa in range(stateParallel.taskDivisions):
            boxOfStatesParallel[indexSherpa] = dictionaryConcurrency[indexSherpa].result()
            groupsOfFoldsTotal += boxOfStatesParallel[indexSherpa].groupsOfFolds
    foldsTotal: int = groupsOfFoldsTotal * stateParallel.leavesTotal
    return (foldsTotal, boxOfStatesParallel)
