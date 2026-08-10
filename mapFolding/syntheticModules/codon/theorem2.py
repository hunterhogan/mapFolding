from __future__ import annotations

from mapFolding.dataBaskets import (
	MapFoldingState, 形Array1DElephino, 形Array1DLeavesTotal, 形Array3DLeavesTotal, 形Elephino, 形FoldsTotal, 形LeavesTotal)
from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
import codon

@codon.jit
def countMapFoldingState[形FoldsTotal, 形Elephino, 形LeavesTotal, 形Array1DLeavesTotal, 形Array1DElephino, 形Array3DLeavesTotal](groupsOfFolds: 形FoldsTotal, gap1ndex: 形Elephino, gap1ndexCeiling: 形Elephino, 次Dimension: 形LeavesTotal, 次MiniGap: 形Elephino, leaf1ndex: 形LeavesTotal, leafConnectee: 形LeavesTotal, dimensionsUnconstrained: 形LeavesTotal, countDimensionsGapped: 形Array1DLeavesTotal, gapRangeStart: 形Array1DElephino, gapsWhere: 形Array1DLeavesTotal, leafAbove: 形Array1DLeavesTotal, leafBelow: 形Array1DLeavesTotal, connectionGraph: 形Array3DLeavesTotal, dimensionsTotal: 形LeavesTotal, leavesTotal: 形LeavesTotal) -> tuple[形FoldsTotal, 形Elephino, 形Elephino, 形LeavesTotal, 形Elephino, 形LeavesTotal, 形LeavesTotal, 形LeavesTotal, 形Array1DLeavesTotal, 形Array1DElephino, 形Array1DLeavesTotal, 形Array1DLeavesTotal, 形Array1DLeavesTotal, 形Array3DLeavesTotal, 形LeavesTotal, 形LeavesTotal]:

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
    mapShape: tuple[形LeavesTotal, ...] = state.mapShape
    groupsOfFolds: 形FoldsTotal = state.groupsOfFolds
    gap1ndex: 形Elephino = state.gap1ndex
    gap1ndexCeiling: 形Elephino = state.gap1ndexCeiling
    次Dimension: 形LeavesTotal = state.次Dimension
    次Leaf: 形LeavesTotal = state.次Leaf
    次MiniGap: 形Elephino = state.次MiniGap
    leaf1ndex: 形LeavesTotal = state.leaf1ndex
    leafConnectee: 形LeavesTotal = state.leafConnectee
    dimensionsUnconstrained: 形LeavesTotal = state.dimensionsUnconstrained
    countDimensionsGapped: 形Array1DLeavesTotal = state.countDimensionsGapped
    gapRangeStart: 形Array1DElephino = state.gapRangeStart
    gapsWhere: 形Array1DLeavesTotal = state.gapsWhere
    leafAbove: 形Array1DLeavesTotal = state.leafAbove
    leafBelow: 形Array1DLeavesTotal = state.leafBelow
    connectionGraph: 形Array3DLeavesTotal = state.connectionGraph
    dimensionsTotal: 形LeavesTotal = state.dimensionsTotal
    leavesTotal: 形LeavesTotal = state.leavesTotal
    groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal = countMapFoldingState(形FoldsTotal(groupsOfFolds), 形Elephino(gap1ndex), 形Elephino(gap1ndexCeiling), 形LeavesTotal(次Dimension), 形Elephino(次MiniGap), 形LeavesTotal(leaf1ndex), 形LeavesTotal(leafConnectee), 形LeavesTotal(dimensionsUnconstrained), countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, 形LeavesTotal(dimensionsTotal), 形LeavesTotal(leavesTotal))
    state = MapFoldingState(mapShape=mapShape, groupsOfFolds=groupsOfFolds, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, 次Dimension=次Dimension, 次Leaf=次Leaf, 次MiniGap=次MiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, dimensionsUnconstrained=dimensionsUnconstrained, countDimensionsGapped=countDimensionsGapped, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow)
    state.connectionGraph = connectionGraph
    state.dimensionsTotal = dimensionsTotal
    state.leavesTotal = leavesTotal
    return state
