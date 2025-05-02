from concurrent.futures import Future as ConcurrentFuture, ProcessPoolExecutor
from copy import deepcopy
from mapFolding.dataBaskets import Array1DElephino, Array1DLeavesTotal, Array3D, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal, ParallelMapFoldingState
from multiprocessing import set_start_method as multiprocessing_set_start_method
from numba import jit
if __name__ == '__main__':
    multiprocessing_set_start_method('spawn')

@jit(cache=True, error_model='numpy', fastmath=True, forceinline=True)
def count(groupsOfFolds: DatatypeFoldsTotal, gap1ndex: DatatypeElephino, gap1ndexCeiling: DatatypeElephino, indexDimension: DatatypeLeavesTotal, indexLeaf: DatatypeLeavesTotal, indexMiniGap: DatatypeElephino, leaf1ndex: DatatypeLeavesTotal, leafConnectee: DatatypeLeavesTotal, dimensionsUnconstrained: DatatypeLeavesTotal, countDimensionsGapped: Array1DLeavesTotal, gapRangeStart: Array1DElephino, gapsWhere: Array1DLeavesTotal, leafAbove: Array1DLeavesTotal, leafBelow: Array1DLeavesTotal, connectionGraph: Array3D, dimensionsTotal: DatatypeLeavesTotal, leavesTotal: DatatypeLeavesTotal, taskDivisions: DatatypeLeavesTotal, taskIndex: DatatypeLeavesTotal) -> tuple[DatatypeFoldsTotal, DatatypeElephino, DatatypeElephino, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeElephino, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal, Array1DLeavesTotal, Array1DElephino, Array1DLeavesTotal, Array1DLeavesTotal, Array1DLeavesTotal, Array3D, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal]:
    while leaf1ndex > 0:
        if leaf1ndex <= 1 or leafBelow[0] == 1:
            if leaf1ndex > leavesTotal:
                groupsOfFolds += 1
            else:
                dimensionsUnconstrained = dimensionsTotal
                gap1ndexCeiling = gapRangeStart[leaf1ndex - 1]
                indexDimension = 0
                while indexDimension < dimensionsTotal:
                    leafConnectee = connectionGraph[indexDimension, leaf1ndex, leaf1ndex]
                    if leafConnectee == leaf1ndex:
                        dimensionsUnconstrained -= 1
                    else:
                        while leafConnectee != leaf1ndex:
                            if leaf1ndex != taskDivisions or leafConnectee % taskDivisions == taskIndex:
                                gapsWhere[gap1ndexCeiling] = leafConnectee
                                if countDimensionsGapped[leafConnectee] == 0:
                                    gap1ndexCeiling += 1
                                countDimensionsGapped[leafConnectee] += 1
                            leafConnectee = connectionGraph[indexDimension, leaf1ndex, leafBelow[leafConnectee]]
                    indexDimension += 1
                if not dimensionsUnconstrained:
                    indexLeaf = 0
                    while indexLeaf < leaf1ndex:
                        gapsWhere[gap1ndexCeiling] = indexLeaf
                        gap1ndexCeiling += 1
                        indexLeaf += 1
                indexMiniGap = gap1ndex
                while indexMiniGap < gap1ndexCeiling:
                    gapsWhere[gap1ndex] = gapsWhere[indexMiniGap]
                    if countDimensionsGapped[gapsWhere[indexMiniGap]] == dimensionsUnconstrained:
                        gap1ndex += 1
                    countDimensionsGapped[gapsWhere[indexMiniGap]] = 0
                    indexMiniGap += 1
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
    return (groupsOfFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal, taskDivisions, taskIndex)

def unRepackParallelMapFoldingState(state: ParallelMapFoldingState) -> ParallelMapFoldingState:
    mapShape: tuple[DatatypeLeavesTotal, ...] = state.mapShape
    groupsOfFolds: DatatypeFoldsTotal = state.groupsOfFolds
    gap1ndex: DatatypeElephino = state.gap1ndex
    gap1ndexCeiling: DatatypeElephino = state.gap1ndexCeiling
    indexDimension: DatatypeLeavesTotal = state.indexDimension
    indexLeaf: DatatypeLeavesTotal = state.indexLeaf
    indexMiniGap: DatatypeElephino = state.indexMiniGap
    leaf1ndex: DatatypeLeavesTotal = state.leaf1ndex
    leafConnectee: DatatypeLeavesTotal = state.leafConnectee
    dimensionsUnconstrained: DatatypeLeavesTotal = state.dimensionsUnconstrained
    countDimensionsGapped: Array1DLeavesTotal = state.countDimensionsGapped
    gapRangeStart: Array1DElephino = state.gapRangeStart
    gapsWhere: Array1DLeavesTotal = state.gapsWhere
    leafAbove: Array1DLeavesTotal = state.leafAbove
    leafBelow: Array1DLeavesTotal = state.leafBelow
    connectionGraph: Array3D = state.connectionGraph
    dimensionsTotal: DatatypeLeavesTotal = state.dimensionsTotal
    leavesTotal: DatatypeLeavesTotal = state.leavesTotal
    taskDivisions: DatatypeLeavesTotal = state.taskDivisions
    taskIndex: DatatypeLeavesTotal = state.taskIndex
    groupsOfFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal, taskDivisions, taskIndex = count(groupsOfFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal, taskDivisions, taskIndex)
    state = ParallelMapFoldingState(mapShape=mapShape, groupsOfFolds=groupsOfFolds, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, indexDimension=indexDimension, indexLeaf=indexLeaf, indexMiniGap=indexMiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, dimensionsUnconstrained=dimensionsUnconstrained, countDimensionsGapped=countDimensionsGapped, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow, taskDivisions=taskDivisions, taskIndex=taskIndex)
    return state

def doTheNeedful(state: ParallelMapFoldingState, concurrencyLimit: int) -> tuple[int, list[ParallelMapFoldingState]]:
    stateParallel = deepcopy(state)
    listStatesParallel: list[ParallelMapFoldingState] = [stateParallel] * stateParallel.taskDivisions
    groupsOfFoldsTotal: int = 0
    dictionaryConcurrency: dict[int, ConcurrentFuture[ParallelMapFoldingState]] = {}
    with ProcessPoolExecutor(concurrencyLimit) as concurrencyManager:
        for indexSherpa in range(stateParallel.taskDivisions):
            state = deepcopy(stateParallel)
            state.taskIndex = indexSherpa
            dictionaryConcurrency[indexSherpa] = concurrencyManager.submit(unRepackParallelMapFoldingState, state)
        for indexSherpa in range(stateParallel.taskDivisions):
            listStatesParallel[indexSherpa] = dictionaryConcurrency[indexSherpa].result()
            groupsOfFoldsTotal += listStatesParallel[indexSherpa].groupsOfFolds
    foldsTotal: int = groupsOfFoldsTotal * stateParallel.leavesTotal
    return (foldsTotal, listStatesParallel)