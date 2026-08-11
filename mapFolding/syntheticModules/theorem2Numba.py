from __future__ import annotations

from mapFolding.dataBaskets import (
	MapFoldingState, 形Array1DElephino, 形Array1DTotalLeaves, 形Array3DTotalLeaves, 形Elephino, 形TotalFolds, 形TotalLeaves)
from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
from numba import jit

@jit(cache=True, error_model='numpy', fastmath=True, forceinline=True)
def count(groupsOfFolds: 形TotalFolds, gap1ndex: 形Elephino, gap1ndexCeiling: 形Elephino, 次Dimension: 形TotalLeaves, 次MiniGap: 形Elephino, leaf1ndex: 形TotalLeaves, leafConnectee: 形TotalLeaves, dimensionsUnconstrained: 形TotalLeaves, countDimensionsGapped: 形Array1DTotalLeaves, gapRangeStart: 形Array1DElephino, gapsWhere: 形Array1DTotalLeaves, leafAbove: 形Array1DTotalLeaves, leafBelow: 形Array1DTotalLeaves, connectionGraph: 形Array3DTotalLeaves, totalDimensions: 形TotalLeaves, totalLeaves: 形TotalLeaves) -> tuple[形TotalFolds, 形Elephino, 形Elephino, 形TotalLeaves, 形Elephino, 形TotalLeaves, 形TotalLeaves, 形TotalLeaves, 形Array1DTotalLeaves, 形Array1DElephino, 形Array1DTotalLeaves, 形Array1DTotalLeaves, 形Array1DTotalLeaves, 形Array3DTotalLeaves, 形TotalLeaves, 形TotalLeaves]:
    while leaf1ndex > 4:
        if leafBelow[0] == 1:
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
                            gapsWhere[gap1ndexCeiling] = leafConnectee
                            if countDimensionsGapped[leafConnectee] == 0:
                                gap1ndexCeiling += 1
                            countDimensionsGapped[leafConnectee] += 1
                            leafConnectee = connectionGraph[次Dimension, leaf1ndex, leafBelow[leafConnectee]]
                    次Dimension += 1
                次MiniGap = gap1ndex
                while 次MiniGap < gap1ndexCeiling:
                    gapsWhere[gap1ndex] = gapsWhere[次MiniGap]
                    if countDimensionsGapped[gapsWhere[次MiniGap]] == dimensionsUnconstrained:
                        gap1ndex += 1
                    countDimensionsGapped[gapsWhere[次MiniGap]] = 0
                    次MiniGap += 1
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
    return (groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, totalDimensions, totalLeaves)

def doTheNeedful(state: MapFoldingState) -> MapFoldingState:
    state = transitionOnGroupsOfFolds(state)
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
    groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, totalDimensions, totalLeaves = count(groupsOfFolds, gap1ndex, gap1ndexCeiling, 次Dimension, 次MiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, totalDimensions, totalLeaves)
    state = MapFoldingState(mapShape=mapShape, groupsOfFolds=groupsOfFolds, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, 次Dimension=次Dimension, 次Leaf=次Leaf, 次MiniGap=次MiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, dimensionsUnconstrained=dimensionsUnconstrained, countDimensionsGapped=countDimensionsGapped, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow)
    return state
