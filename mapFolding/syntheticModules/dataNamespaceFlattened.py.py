from mapFolding.theSSOT import ComputationState

def doTheNeedful(mapShape: tuple[DatatypeLeavesTotal, ...], leavesTotal: DatatypeLeavesTotal, taskDivisions: DatatypeLeavesTotal, connectionGraph: Array3D, dimensionsTotal: DatatypeLeavesTotal, countDimensionsGapped: Array1DLeavesTotal, dimensionsUnconstrained: DatatypeLeavesTotal, gapRangeStart: Array1DElephino, gapsWhere: Array1DLeavesTotal, leafAbove: Array1DLeavesTotal, leafBelow: Array1DLeavesTotal, foldGroups: Array1DFoldsTotal, foldsTotal: DatatypeFoldsTotal, gap1ndex: DatatypeLeavesTotal, gap1ndexCeiling: DatatypeElephino, groupsOfFolds: DatatypeFoldsTotal, indexDimension: DatatypeLeavesTotal, indexLeaf: DatatypeLeavesTotal, indexMiniGap: DatatypeElephino, leaf1ndex: DatatypeElephino, leafConnectee: DatatypeElephino, taskIndex: DatatypeLeavesTotal) -> ComputationState:
    computationStateInitialized = countInitialize(computationStateInitialized)
    if computationStateInitialized.taskDivisions > 0:
        return countParallel(computationStateInitialized)
    else:
        return countSequential(computationStateInitialized)

def flattenData(state: ComputationState) -> ComputationState:
    mapShape: tuple[DatatypeLeavesTotal, ...] = state.mapShape
    leavesTotal: DatatypeLeavesTotal = state.leavesTotal
    taskDivisions: DatatypeLeavesTotal = state.taskDivisions
    connectionGraph: Array3D = state.connectionGraph
    dimensionsTotal: DatatypeLeavesTotal = state.dimensionsTotal
    countDimensionsGapped: Array1DLeavesTotal = state.countDimensionsGapped
    dimensionsUnconstrained: DatatypeLeavesTotal = state.dimensionsUnconstrained
    gapRangeStart: Array1DElephino = state.gapRangeStart
    gapsWhere: Array1DLeavesTotal = state.gapsWhere
    leafAbove: Array1DLeavesTotal = state.leafAbove
    leafBelow: Array1DLeavesTotal = state.leafBelow
    foldGroups: Array1DFoldsTotal = state.foldGroups
    foldsTotal: DatatypeFoldsTotal = state.foldsTotal
    gap1ndex: DatatypeLeavesTotal = state.gap1ndex
    gap1ndexCeiling: DatatypeElephino = state.gap1ndexCeiling
    groupsOfFolds: DatatypeFoldsTotal = state.groupsOfFolds
    indexDimension: DatatypeLeavesTotal = state.indexDimension
    indexLeaf: DatatypeLeavesTotal = state.indexLeaf
    indexMiniGap: DatatypeElephino = state.indexMiniGap
    leaf1ndex: DatatypeElephino = state.leaf1ndex
    leafConnectee: DatatypeElephino = state.leafConnectee
    taskIndex: DatatypeLeavesTotal = state.taskIndex
    mapShape, leavesTotal, taskDivisions, connectionGraph, dimensionsTotal, countDimensionsGapped, dimensionsUnconstrained, gapRangeStart, gapsWhere, leafAbove, leafBelow, foldGroups, foldsTotal, gap1ndex, gap1ndexCeiling, groupsOfFolds, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, taskIndex = doTheNeedful(mapShape, leavesTotal, taskDivisions, connectionGraph, dimensionsTotal, countDimensionsGapped, dimensionsUnconstrained, gapRangeStart, gapsWhere, leafAbove, leafBelow, foldGroups, foldsTotal, gap1ndex, gap1ndexCeiling, groupsOfFolds, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, taskIndex)
    return ComputationState(mapShape=mapShape, leavesTotal=leavesTotal, taskDivisions=taskDivisions, countDimensionsGapped=countDimensionsGapped, dimensionsUnconstrained=dimensionsUnconstrained, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow, foldGroups=foldGroups, foldsTotal=foldsTotal, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, groupsOfFolds=groupsOfFolds, indexDimension=indexDimension, indexLeaf=indexLeaf, indexMiniGap=indexMiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, taskIndex=taskIndex)