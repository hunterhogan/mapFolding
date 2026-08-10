from __future__ import annotations

from mapFolding.dataBaskets import (
	MapFoldingState, 形Array1DElephino, 形Array1DLeavesTotal, 形Array3DLeavesTotal, 形Elephino, 形FoldsTotal, 形LeavesTotal)
from numba import jit

@jit(cache=True, error_model='numpy', fastmath=True, forceinline=True)
def count(groupsOfFolds: 形FoldsTotal, gap1ndex: 形Elephino, gap1ndexCeiling: 形Elephino, 次Dimension: 形LeavesTotal, 次Leaf: 形LeavesTotal, 次MiniGap: 形Elephino, leaf1ndex: 形LeavesTotal, leafConnectee: 形LeavesTotal, dimensionsUnconstrained: 形LeavesTotal, countDimensionsGapped: 形Array1DLeavesTotal, gapRangeStart: 形Array1DElephino, gapsWhere: 形Array1DLeavesTotal, leafAbove: 形Array1DLeavesTotal, leafBelow: 形Array1DLeavesTotal, connectionGraph: 形Array3DLeavesTotal, dimensionsTotal: 形LeavesTotal, leavesTotal: 形LeavesTotal) -> tuple[形FoldsTotal, 形Elephino, 形Elephino, 形LeavesTotal, 形LeavesTotal, 形Elephino, 形LeavesTotal, 形LeavesTotal, 形LeavesTotal, 形Array1DLeavesTotal, 形Array1DElephino, 形Array1DLeavesTotal, 形Array1DLeavesTotal, 形Array1DLeavesTotal, 形Array3DLeavesTotal, 形LeavesTotal, 形LeavesTotal]:
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
    return (groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次Leaf, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal)

def doTheNeedful(state: MapFoldingState) -> MapFoldingState:
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
    groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次Leaf, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal = count(groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次Leaf, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal)
    state = MapFoldingState(mapShape=mapShape, groupsOfFolds=groupsOfFolds, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, 次Dimension=次Dimension, 次Leaf=次Leaf, 次MiniGap=次MiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, dimensionsUnconstrained=dimensionsUnconstrained, countDimensionsGapped=countDimensionsGapped, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow)
    return state
