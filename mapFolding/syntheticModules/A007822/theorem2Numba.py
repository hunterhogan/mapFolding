from mapFolding.dataBaskets import Array1DElephino, Array1DLeavesTotal, Array2DLeavesTotal, Array3DLeavesTotal, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal, SymmetricFoldsState
from mapFolding.syntheticModules.A007822.initializeState import transitionOnGroupsOfFolds
from numba import jit
import numpy

@jit(cache=True, error_model='numpy', fastmath=True, forceinline=True)
def count(groupsOfFolds: DatatypeFoldsTotal, gap1ndex: DatatypeElephino, gap1ndexCeiling: DatatypeElephino, indexDimension: DatatypeLeavesTotal, indexLeaf: DatatypeLeavesTotal, indexMiniGap: DatatypeElephino, leaf1ndex: DatatypeLeavesTotal, leafConnectee: DatatypeLeavesTotal, dimensionsUnconstrained: DatatypeLeavesTotal, countDimensionsGapped: Array1DLeavesTotal, gapRangeStart: Array1DElephino, gapsWhere: Array1DLeavesTotal, leafAbove: Array1DLeavesTotal, leafBelow: Array1DLeavesTotal, leafComparison: Array1DLeavesTotal, arrayGroupOfFolds: Array2DLeavesTotal, connectionGraph: Array3DLeavesTotal, dimensionsTotal: DatatypeLeavesTotal, indicesArrayGroupOfFolds: Array2DLeavesTotal, leavesTotal: DatatypeLeavesTotal) -> tuple[DatatypeFoldsTotal, DatatypeElephino, DatatypeElephino, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeElephino, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal, Array1DLeavesTotal, Array1DElephino, Array1DLeavesTotal, Array1DLeavesTotal, Array1DLeavesTotal, Array1DLeavesTotal, Array2DLeavesTotal, Array3DLeavesTotal, DatatypeLeavesTotal, Array2DLeavesTotal, DatatypeLeavesTotal]:
    while leaf1ndex > 4:
        if leafBelow[0] == 1:
            if leaf1ndex > leavesTotal:
                indexLeaf = 1
                leafComparison[0] = 1
                leafConnectee = 1
                while leafConnectee < leavesTotal + 1:
                    indexMiniGap = leafBelow[indexLeaf]
                    leafComparison[leafConnectee] = (indexMiniGap - indexLeaf + leavesTotal) % leavesTotal
                    indexLeaf = indexMiniGap
                    leafConnectee += 1
                arrayGroupOfFolds = numpy.take(leafComparison, indicesArrayGroupOfFolds)
                compared = arrayGroupOfFolds[..., 0:leavesTotal // 2] == arrayGroupOfFolds[..., leavesTotal // 2:None]
                for indexRow in range(len(compared)):
                    groupsOfFolds += compared[indexRow].all()
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
                            gapsWhere[gap1ndexCeiling] = leafConnectee
                            if countDimensionsGapped[leafConnectee] == 0:
                                gap1ndexCeiling += 1
                            countDimensionsGapped[leafConnectee] += 1
                            leafConnectee = connectionGraph[indexDimension, leaf1ndex, leafBelow[leafConnectee]]
                    indexDimension += 1
                indexMiniGap = gap1ndex
                while indexMiniGap < gap1ndexCeiling:
                    gapsWhere[gap1ndex] = gapsWhere[indexMiniGap]
                    if countDimensionsGapped[gapsWhere[indexMiniGap]] == dimensionsUnconstrained:
                        gap1ndex += 1
                    countDimensionsGapped[gapsWhere[indexMiniGap]] = 0
                    indexMiniGap += 1
        while gap1ndex == gapRangeStart[leaf1ndex - 1]:
            leaf1ndex -= 1
            leafBelow[leafAbove[leaf1ndex]] = leafBelow[leaf1ndex]
            leafAbove[leafBelow[leaf1ndex]] = leafAbove[leaf1ndex]
        gap1ndex -= 1
        leafAbove[leaf1ndex] = gapsWhere[gap1ndex]
        leafBelow[leaf1ndex] = leafBelow[leafAbove[leaf1ndex]]
        leafBelow[leafAbove[leaf1ndex]] = leaf1ndex
        leafAbove[leafBelow[leaf1ndex]] = leaf1ndex
        gapRangeStart[leaf1ndex] = gap1ndex
        leaf1ndex += 1
    else:
        groupsOfFolds *= 2
    groupsOfFolds = (groupsOfFolds + 1) // 2
    return (groupsOfFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, leafComparison, arrayGroupOfFolds, connectionGraph, dimensionsTotal, indicesArrayGroupOfFolds, leavesTotal)

def doTheNeedful(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state = transitionOnGroupsOfFolds(state)
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
    leafComparison: Array1DLeavesTotal = state.leafComparison
    arrayGroupOfFolds: Array2DLeavesTotal = state.arrayGroupOfFolds
    connectionGraph: Array3DLeavesTotal = state.connectionGraph
    dimensionsTotal: DatatypeLeavesTotal = state.dimensionsTotal
    indicesArrayGroupOfFolds: Array2DLeavesTotal = state.indicesArrayGroupOfFolds
    leavesTotal: DatatypeLeavesTotal = state.leavesTotal
    groupsOfFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, leafComparison, arrayGroupOfFolds, connectionGraph, dimensionsTotal, indicesArrayGroupOfFolds, leavesTotal = count(groupsOfFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, leafComparison, arrayGroupOfFolds, connectionGraph, dimensionsTotal, indicesArrayGroupOfFolds, leavesTotal)
    state = SymmetricFoldsState(mapShape=mapShape, groupsOfFolds=groupsOfFolds, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, indexDimension=indexDimension, indexLeaf=indexLeaf, indexMiniGap=indexMiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, dimensionsUnconstrained=dimensionsUnconstrained, countDimensionsGapped=countDimensionsGapped, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow, leafComparison=leafComparison)
    return state