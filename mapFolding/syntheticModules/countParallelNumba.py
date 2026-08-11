from __future__ import annotations

from concurrent.futures import Future as ConcurrentFuture, ProcessPoolExecutor
from copy import deepcopy
from mapFolding.dataBaskets import (
	ParallelMapFoldingState, 形Array1DElephino, 形Array1DTotalLeaves, 形Array3DTotalLeaves, 形Elephino, 形TotalFolds, 形TotalLeaves)
from multiprocessing import set_start_method as multiprocessing_set_start_method
from numba import jit

if __name__ == '__main__':
    multiprocessing_set_start_method('spawn')

@jit(cache=True, error_model='numpy', fastmath=True, forceinline=True)
def count(groupsOfFolds: 形TotalFolds, gap1ndex: 形Elephino, gap1ndexCeiling: 形Elephino, 次Dimension: 形TotalLeaves, 次Leaf: 形TotalLeaves, 次MiniGap: 形Elephino, leaf1ndex: 形TotalLeaves, leafConnectee: 形TotalLeaves, dimensionsUnconstrained: 形TotalLeaves, countDimensionsGapped: 形Array1DTotalLeaves, gapRangeStart: 形Array1DElephino, gapsWhere: 形Array1DTotalLeaves, leafAbove: 形Array1DTotalLeaves, leafBelow: 形Array1DTotalLeaves, connectionGraph: 形Array3DTotalLeaves, totalDimensions: 形TotalLeaves, totalLeaves: 形TotalLeaves, taskDivisions: 形TotalLeaves, task次: 形TotalLeaves) -> tuple[形TotalFolds, 形Elephino, 形Elephino, 形TotalLeaves, 形TotalLeaves, 形Elephino, 形TotalLeaves, 形TotalLeaves, 形TotalLeaves, 形Array1DTotalLeaves, 形Array1DElephino, 形Array1DTotalLeaves, 形Array1DTotalLeaves, 形Array1DTotalLeaves, 形Array3DTotalLeaves, 形TotalLeaves, 形TotalLeaves, 形TotalLeaves, 形TotalLeaves]:
    while leaf1ndex > 0:
        if leaf1ndex <= 1 or leafBelow[0] == 1:
            if leaf1ndex > totalLeaves:
                groupsOfFolds += 1
            else:
                dimensionsUnconstrained = totalDimensions
                gap1ndexCeiling = gapRangeStart[leaf1ndex - 1]
                次Dimension = 0
                while 次Dimension < totalDimensions:
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
    return (groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次Leaf, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, totalDimensions, totalLeaves, taskDivisions, task次)

def unRepackParallelMapFoldingState(state: ParallelMapFoldingState) -> ParallelMapFoldingState:
    mapShape: tuple[形TotalLeaves, ...] = state.mapShape
    groupsOfFolds: 形TotalFolds = state.groupsOfFolds
    gap1ndex: 形Elephino = state.gap1ndex
    gap1ndexCeiling: 形Elephino = state.gap1ndexCeiling
    次Dimension: 形TotalLeaves = state.次Dimension
    次Leaf: 形TotalLeaves = state.次Leaf
    次MiniGap: 形Elephino = state.次MiniGap
    leaf1ndex: 形TotalLeaves = state.leaf1ndex
    leafConnectee: 形TotalLeaves = state.leafConnectee
    dimensionsUnconstrained: 形TotalLeaves = state.dimensionsUnconstrained
    countDimensionsGapped: 形Array1DTotalLeaves = state.countDimensionsGapped
    gapRangeStart: 形Array1DElephino = state.gapRangeStart
    gapsWhere: 形Array1DTotalLeaves = state.gapsWhere
    leafAbove: 形Array1DTotalLeaves = state.leafAbove
    leafBelow: 形Array1DTotalLeaves = state.leafBelow
    connectionGraph: 形Array3DTotalLeaves = state.connectionGraph
    totalDimensions: 形TotalLeaves = state.totalDimensions
    totalLeaves: 形TotalLeaves = state.totalLeaves
    taskDivisions: 形TotalLeaves = state.taskDivisions
    task次: 形TotalLeaves = state.task次
    groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次Leaf, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, totalDimensions, totalLeaves, taskDivisions, task次 = count(groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次Leaf, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, totalDimensions, totalLeaves, taskDivisions, task次)
    state = ParallelMapFoldingState(mapShape=mapShape, groupsOfFolds=groupsOfFolds, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, 次Dimension=次Dimension, 次Leaf=次Leaf, 次MiniGap=次MiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, dimensionsUnconstrained=dimensionsUnconstrained, countDimensionsGapped=countDimensionsGapped, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow, taskDivisions=taskDivisions, task次=task次)
    return state

def doTheNeedful(state: ParallelMapFoldingState, concurrencyLimit: int) -> tuple[int, list[ParallelMapFoldingState]]:
    stateParallel = deepcopy(state)
    boxOfStatesParallel: list[ParallelMapFoldingState] = [stateParallel] * stateParallel.taskDivisions
    groupsOfTotalFolds: int = 0
    dictionaryConcurrency: dict[int, ConcurrentFuture[ParallelMapFoldingState]] = {}
    with ProcessPoolExecutor(concurrencyLimit) as concurrencyManager:
        for indexSherpa in range(stateParallel.taskDivisions):
            state = deepcopy(stateParallel)
            state.task次 = indexSherpa
            dictionaryConcurrency[indexSherpa] = concurrencyManager.submit(unRepackParallelMapFoldingState, state)
        for indexSherpa in range(stateParallel.taskDivisions):
            boxOfStatesParallel[indexSherpa] = dictionaryConcurrency[indexSherpa].result()
            groupsOfTotalFolds += boxOfStatesParallel[indexSherpa].groupsOfFolds
    totalFolds: int = groupsOfTotalFolds * stateParallel.totalLeaves
    return (totalFolds, boxOfStatesParallel)
