from mapFolding.dataBaskets import (
	Array1DElephino, Array1DLeavesTotal, Array3DLeavesTotal, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal, MapFoldingState)
from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
import codon

@codon.jit
def countCodon__mapFolding_syntheticModules_codon_theorem2(groupsOfFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal):
    import numpy as np
    while leaf1ndex > 4:
        if int(leafBelow[int(0)]) == 1:
            if leaf1ndex > leavesTotal:
                groupsOfFolds += 1
            else:
                dimensionsUnconstrained = dimensionsTotal
                gap1ndexCeiling = int(gapRangeStart[int(leaf1ndex - 1)])
                indexDimension = 0
                while indexDimension < dimensionsTotal:
                    leafConnectee = int(connectionGraph[int(indexDimension), int(leaf1ndex), int(leaf1ndex)])
                    if leafConnectee == leaf1ndex:
                        dimensionsUnconstrained -= 1
                    else:
                        while leafConnectee != leaf1ndex:
                            gapsWhere[int(gap1ndexCeiling)] = leafConnectee
                            if int(countDimensionsGapped[int(leafConnectee)]) == 0:
                                gap1ndexCeiling += 1
                            countDimensionsGapped[int(leafConnectee)] += np.int64(1)
                            leafConnectee = int(connectionGraph[int(indexDimension), int(leaf1ndex), int(leafBelow[leafConnectee])])
                    indexDimension += 1
                indexMiniGap = gap1ndex
                while indexMiniGap < gap1ndexCeiling:
                    gapsWhere[int(gap1ndex)] = int(gapsWhere[int(indexMiniGap)])
                    if int(countDimensionsGapped[int(gapsWhere[indexMiniGap])]) == dimensionsUnconstrained:
                        gap1ndex += 1
                    countDimensionsGapped[int(int(gapsWhere[indexMiniGap]))] = 0
                    indexMiniGap += 1
        while gap1ndex == int(gapRangeStart[int(leaf1ndex - 1)]):
            leaf1ndex -= 1
            leafBelow[int(int(leafAbove[leaf1ndex]))] = int(leafBelow[int(leaf1ndex)])
            leafAbove[int(int(leafBelow[leaf1ndex]))] = int(leafAbove[int(leaf1ndex)])
        gap1ndex -= 1
        leafAbove[int(leaf1ndex)] = int(gapsWhere[int(gap1ndex)])
        leafBelow[int(leaf1ndex)] = int(leafBelow[int(leafAbove[leaf1ndex])])
        leafBelow[int(int(leafAbove[leaf1ndex]))] = leaf1ndex
        leafAbove[int(int(leafBelow[leaf1ndex]))] = leaf1ndex
        gapRangeStart[int(leaf1ndex)] = gap1ndex
        leaf1ndex += 1
    else:
        groupsOfFolds *= 2
    return (groupsOfFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal)

def doTheNeedful(state: MapFoldingState) -> MapFoldingState:
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
    connectionGraph: Array3DLeavesTotal = state.connectionGraph
    dimensionsTotal: DatatypeLeavesTotal = state.dimensionsTotal
    leavesTotal: DatatypeLeavesTotal = state.leavesTotal
    groupsOfFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal = countCodon__mapFolding_syntheticModules_codon_theorem2(int(groupsOfFolds), int(gap1ndex), int(gap1ndexCeiling), int(indexDimension), int(indexMiniGap), int(leaf1ndex), int(leafConnectee), int(dimensionsUnconstrained), countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, int(dimensionsTotal), int(leavesTotal))
    state = MapFoldingState(mapShape=mapShape, groupsOfFolds=groupsOfFolds, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, indexDimension=indexDimension, indexLeaf=indexLeaf, indexMiniGap=indexMiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, dimensionsUnconstrained=dimensionsUnconstrained, countDimensionsGapped=countDimensionsGapped, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow)
    return state
