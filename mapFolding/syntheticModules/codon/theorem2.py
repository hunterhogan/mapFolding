from __future__ import annotations

from mapFolding.dataBaskets import (
	Array1DElephino, Array1DLeavesTotal, Array3DLeavesTotal, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal, MapFoldingState)
from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
import codon

@codon.jit
def countMapFoldingState[DatatypeFoldsTotal, DatatypeElephino, DatatypeLeavesTotal, Array1DLeavesTotal, Array1DElephino, Array3DLeavesTotal](groupsOfFolds: DatatypeFoldsTotal, gap1ndex: DatatypeElephino, gap1ndexCeiling: DatatypeElephino, 次Dimension: DatatypeLeavesTotal, 次MiniGap: DatatypeElephino, leaf1ndex: DatatypeLeavesTotal, leafConnectee: DatatypeLeavesTotal, dimensionsUnconstrained: DatatypeLeavesTotal, countDimensionsGapped: Array1DLeavesTotal, gapRangeStart: Array1DElephino, gapsWhere: Array1DLeavesTotal, leafAbove: Array1DLeavesTotal, leafBelow: Array1DLeavesTotal, connectionGraph: Array3DLeavesTotal, dimensionsTotal: DatatypeLeavesTotal, leavesTotal: DatatypeLeavesTotal) -> tuple[DatatypeFoldsTotal, DatatypeElephino, DatatypeElephino, DatatypeLeavesTotal, DatatypeElephino, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal, Array1DLeavesTotal, Array1DElephino, Array1DLeavesTotal, Array1DLeavesTotal, Array1DLeavesTotal, Array3DLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal]:

    def compatibleValue[Reference, Value](_reference: Reference, value: Value) -> Reference:
        return Reference(value)
    while leaf1ndex > compatibleValue(leaf1ndex, 4):
        if leafBelow[0] == compatibleValue(leafBelow[0], 1):
            if leaf1ndex > compatibleValue(leaf1ndex, leavesTotal):
                groupsOfFolds += compatibleValue(groupsOfFolds, 1)
            else:
                dimensionsUnconstrained = compatibleValue(dimensionsUnconstrained, dimensionsTotal)
                gap1ndexCeiling = compatibleValue(gap1ndexCeiling, gapRangeStart[(leaf1ndex - compatibleValue(leaf1ndex, 1)).__index__()])
                次Dimension = compatibleValue(次Dimension, 0)
                while 次Dimension < compatibleValue(次Dimension, dimensionsTotal):
                    leafConnectee = compatibleValue(leafConnectee, connectionGraph[次Dimension.__index__(), leaf1ndex.__index__(), leaf1ndex.__index__()])
                    if leafConnectee == compatibleValue(leafConnectee, leaf1ndex):
                        dimensionsUnconstrained -= compatibleValue(dimensionsUnconstrained, 1)
                    else:
                        while leafConnectee != compatibleValue(leafConnectee, leaf1ndex):
                            gapsWhere[gap1ndexCeiling.__index__()] = compatibleValue(gapsWhere[gap1ndexCeiling.__index__()], leafConnectee)
                            if countDimensionsGapped[leafConnectee.__index__()] == compatibleValue(countDimensionsGapped[leafConnectee.__index__()], 0):
                                gap1ndexCeiling += compatibleValue(gap1ndexCeiling, 1)
                            countDimensionsGapped[leafConnectee.__index__()] += compatibleValue(countDimensionsGapped[leafConnectee.__index__()], 1)
                            leafConnectee = compatibleValue(leafConnectee, connectionGraph[次Dimension.__index__(), leaf1ndex.__index__(), leafBelow[leafConnectee.__index__()].__index__()])
                    次Dimension += compatibleValue(次Dimension, 1)
                次MiniGap = compatibleValue(次MiniGap, gap1ndex)
                while 次MiniGap < compatibleValue(次MiniGap, gap1ndexCeiling):
                    gapsWhere[gap1ndex.__index__()] = compatibleValue(gapsWhere[gap1ndex.__index__()], gapsWhere[次MiniGap.__index__()])
                    if countDimensionsGapped[gapsWhere[次MiniGap.__index__()].__index__()] == compatibleValue(countDimensionsGapped[gapsWhere[次MiniGap.__index__()].__index__()], dimensionsUnconstrained):
                        gap1ndex += compatibleValue(gap1ndex, 1)
                    countDimensionsGapped[gapsWhere[次MiniGap.__index__()].__index__()] = compatibleValue(countDimensionsGapped[gapsWhere[次MiniGap.__index__()].__index__()], 0)
                    次MiniGap += compatibleValue(次MiniGap, 1)
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
        groupsOfFolds *= compatibleValue(groupsOfFolds, 2)
    return (groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal)

def doTheNeedful(state: MapFoldingState) -> MapFoldingState:
    state = transitionOnGroupsOfFolds(state)
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
    groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal = countMapFoldingState(DatatypeFoldsTotal(groupsOfFolds), DatatypeElephino(gap1ndex), DatatypeElephino(gap1ndexCeiling), DatatypeLeavesTotal(次Dimension), DatatypeElephino(次MiniGap), DatatypeLeavesTotal(leaf1ndex), DatatypeLeavesTotal(leafConnectee), DatatypeLeavesTotal(dimensionsUnconstrained), countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, DatatypeLeavesTotal(dimensionsTotal), DatatypeLeavesTotal(leavesTotal))
    state = MapFoldingState(mapShape=mapShape, groupsOfFolds=groupsOfFolds, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, 次Dimension=次Dimension, 次Leaf=次Leaf, 次MiniGap=次MiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, dimensionsUnconstrained=dimensionsUnconstrained, countDimensionsGapped=countDimensionsGapped, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow)
    state.connectionGraph = connectionGraph
    state.dimensionsTotal = dimensionsTotal
    state.leavesTotal = leavesTotal
    return state
