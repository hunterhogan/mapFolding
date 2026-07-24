from __future__ import annotations

from mapFolding.dataBaskets import (
	Array1DElephino, Array1DLeavesTotal, Array3DLeavesTotal, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal, SymmetricFoldsState)
from mapFolding.syntheticModules.A007822.initializeState import transitionOnGroupsOfFolds
import codon

@codon.jit
def countSymmetricFoldsState[DatatypeFoldsTotal, DatatypeElephino, DatatypeLeavesTotal, Array1DLeavesTotal, Array1DElephino, Array3DLeavesTotal](symmetricFolds: DatatypeFoldsTotal, gap1ndex: DatatypeElephino, gap1ndexCeiling: DatatypeElephino, indexDimension: DatatypeLeavesTotal, indexLeaf: DatatypeLeavesTotal, indexMiniGap: DatatypeElephino, leaf1ndex: DatatypeLeavesTotal, leafConnectee: DatatypeLeavesTotal, dimensionsUnconstrained: DatatypeLeavesTotal, countDimensionsGapped: Array1DLeavesTotal, gapRangeStart: Array1DElephino, gapsWhere: Array1DLeavesTotal, leafAbove: Array1DLeavesTotal, leafBelow: Array1DLeavesTotal, leafComparison: Array1DLeavesTotal, connectionGraph: Array3DLeavesTotal, dimensionsTotal: DatatypeLeavesTotal, indices: list[list[tuple[int, int]]], leavesTotal: DatatypeLeavesTotal) -> tuple[DatatypeFoldsTotal, DatatypeElephino, DatatypeElephino, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeElephino, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal, Array1DLeavesTotal, Array1DElephino, Array1DLeavesTotal, Array1DLeavesTotal, Array1DLeavesTotal, Array1DLeavesTotal, Array3DLeavesTotal, DatatypeLeavesTotal, list[list[tuple[int, int]]], DatatypeLeavesTotal]:

    def compatibleValue[Reference, Value](_reference: Reference, value: Value) -> Reference:
        return Reference(value)
    while leaf1ndex > compatibleValue(leaf1ndex, 4):
        if leafBelow[0] == compatibleValue(leafBelow[0], 1):
            if leaf1ndex > compatibleValue(leaf1ndex, leavesTotal):
                indexLeaf = compatibleValue(indexLeaf, 1)
                leafComparison[0] = compatibleValue(leafComparison[0], 1)
                leafConnectee = compatibleValue(leafConnectee, 1)
                while leafConnectee < compatibleValue(leafConnectee, leavesTotal + compatibleValue(leavesTotal, 1)):
                    indexMiniGap = compatibleValue(indexMiniGap, leafBelow[indexLeaf.__index__()])
                    leafComparison[leafConnectee.__index__()] = compatibleValue(leafComparison[leafConnectee.__index__()], (indexMiniGap - compatibleValue(indexMiniGap, indexLeaf) + leavesTotal) % leavesTotal)
                    indexLeaf = compatibleValue(indexLeaf, indexMiniGap)
                    leafConnectee += compatibleValue(leafConnectee, 1)
                for listTuples in indices:
                    leafConnectee = compatibleValue(leafConnectee, 1)
                    for indexLeft, indexRight in listTuples:
                        if leafComparison[indexLeft.__index__()] != compatibleValue(leafComparison[indexLeft.__index__()], leafComparison[indexRight.__index__()]):
                            leafConnectee = compatibleValue(leafConnectee, 0)
                            break
                    symmetricFolds += compatibleValue(symmetricFolds, leafConnectee)
            else:
                dimensionsUnconstrained = compatibleValue(dimensionsUnconstrained, dimensionsTotal)
                gap1ndexCeiling = compatibleValue(gap1ndexCeiling, gapRangeStart[(leaf1ndex - compatibleValue(leaf1ndex, 1)).__index__()])
                indexDimension = compatibleValue(indexDimension, 0)
                while indexDimension < compatibleValue(indexDimension, dimensionsTotal):
                    leafConnectee = compatibleValue(leafConnectee, connectionGraph[indexDimension.__index__(), leaf1ndex.__index__(), leaf1ndex.__index__()])
                    if leafConnectee == compatibleValue(leafConnectee, leaf1ndex):
                        dimensionsUnconstrained -= compatibleValue(dimensionsUnconstrained, 1)
                    else:
                        while leafConnectee != compatibleValue(leafConnectee, leaf1ndex):
                            gapsWhere[gap1ndexCeiling.__index__()] = compatibleValue(gapsWhere[gap1ndexCeiling.__index__()], leafConnectee)
                            if countDimensionsGapped[leafConnectee.__index__()] == compatibleValue(countDimensionsGapped[leafConnectee.__index__()], 0):
                                gap1ndexCeiling += compatibleValue(gap1ndexCeiling, 1)
                            countDimensionsGapped[leafConnectee.__index__()] += compatibleValue(countDimensionsGapped[leafConnectee.__index__()], 1)
                            leafConnectee = compatibleValue(leafConnectee, connectionGraph[indexDimension.__index__(), leaf1ndex.__index__(), leafBelow[leafConnectee.__index__()].__index__()])
                    indexDimension += compatibleValue(indexDimension, 1)
                indexMiniGap = compatibleValue(indexMiniGap, gap1ndex)
                while indexMiniGap < compatibleValue(indexMiniGap, gap1ndexCeiling):
                    gapsWhere[gap1ndex.__index__()] = compatibleValue(gapsWhere[gap1ndex.__index__()], gapsWhere[indexMiniGap.__index__()])
                    if countDimensionsGapped[gapsWhere[indexMiniGap.__index__()].__index__()] == compatibleValue(countDimensionsGapped[gapsWhere[indexMiniGap.__index__()].__index__()], dimensionsUnconstrained):
                        gap1ndex += compatibleValue(gap1ndex, 1)
                    countDimensionsGapped[gapsWhere[indexMiniGap.__index__()].__index__()] = compatibleValue(countDimensionsGapped[gapsWhere[indexMiniGap.__index__()].__index__()], 0)
                    indexMiniGap += compatibleValue(indexMiniGap, 1)
        while gap1ndex == compatibleValue(gap1ndex, gapRangeStart[(leaf1ndex - compatibleValue(leaf1ndex, 1)).__index__()]):
            leaf1ndex -= compatibleValue(leaf1ndex, 1)
            leafBelow[leafAbove[leaf1ndex.__index__()].__index__()] = compatibleValue(leafBelow[leafAbove[leaf1ndex.__index__()].__index__()], leafBelow[leaf1ndex.__index__()])
            leafAbove[leafBelow[leaf1ndex.__index__()].__index__()] = compatibleValue(leafAbove[leafBelow[leaf1ndex.__index__()].__index__()], leafAbove[leaf1ndex.__index__()])
        gap1ndex -= compatibleValue(gap1ndex, 1)
        leafAbove[leaf1ndex.__index__()] = compatibleValue(leafAbove[leaf1ndex.__index__()], gapsWhere[gap1ndex.__index__()])
        leafBelow[leaf1ndex.__index__()] = compatibleValue(leafBelow[leaf1ndex.__index__()], leafBelow[leafAbove[leaf1ndex.__index__()].__index__()])
        leafBelow[leafAbove[leaf1ndex.__index__()].__index__()] = compatibleValue(leafBelow[leafAbove[leaf1ndex.__index__()].__index__()], leaf1ndex)
        leafAbove[leafBelow[leaf1ndex.__index__()].__index__()] = compatibleValue(leafAbove[leafBelow[leaf1ndex.__index__()].__index__()], leaf1ndex)
        gapRangeStart[leaf1ndex.__index__()] = compatibleValue(gapRangeStart[leaf1ndex.__index__()], gap1ndex)
        leaf1ndex += compatibleValue(leaf1ndex, 1)
    else:
        symmetricFolds *= compatibleValue(symmetricFolds, 2)
    symmetricFolds = compatibleValue(symmetricFolds, (symmetricFolds + compatibleValue(symmetricFolds, 1)) // 2)
    return (symmetricFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, leafComparison, connectionGraph, dimensionsTotal, indices, leavesTotal)

def doTheNeedful(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state = transitionOnGroupsOfFolds(state)
    mapShape: tuple[DatatypeLeavesTotal, ...] = state.mapShape
    symmetricFolds: DatatypeFoldsTotal = state.symmetricFolds
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
    connectionGraph: Array3DLeavesTotal = state.connectionGraph
    dimensionsTotal: DatatypeLeavesTotal = state.dimensionsTotal
    indices: list[list[tuple[int, int]]] = state.indices
    leavesTotal: DatatypeLeavesTotal = state.leavesTotal
    symmetricFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, leafComparison, connectionGraph, dimensionsTotal, indices, leavesTotal = countSymmetricFoldsState(DatatypeFoldsTotal(symmetricFolds), DatatypeElephino(gap1ndex), DatatypeElephino(gap1ndexCeiling), DatatypeLeavesTotal(indexDimension), DatatypeLeavesTotal(indexLeaf), DatatypeElephino(indexMiniGap), DatatypeLeavesTotal(leaf1ndex), DatatypeLeavesTotal(leafConnectee), DatatypeLeavesTotal(dimensionsUnconstrained), countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, leafComparison, connectionGraph, DatatypeLeavesTotal(dimensionsTotal), indices, DatatypeLeavesTotal(leavesTotal))
    state = SymmetricFoldsState(mapShape=mapShape, symmetricFolds=symmetricFolds, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, indexDimension=indexDimension, indexLeaf=indexLeaf, indexMiniGap=indexMiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, dimensionsUnconstrained=dimensionsUnconstrained, countDimensionsGapped=countDimensionsGapped, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow, leafComparison=leafComparison)
    state.connectionGraph = connectionGraph
    state.dimensionsTotal = dimensionsTotal
    state.indices = indices
    state.leavesTotal = leavesTotal
    return state
