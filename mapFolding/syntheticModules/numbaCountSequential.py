from mapFolding.theDao import countInitialize, doTheNeedful
from mapFolding.theSSOT import Array1DElephino, Array1DLeavesTotal, Array3D, ComputationState, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal
from numba import jit

@jit(_nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, nopython=True, parallel=False)
def countSequential(
	connectionGraph: Array3D,
	countDimensionsGapped: Array1DLeavesTotal,
	dimensionsTotal: DatatypeLeavesTotal,
	dimensionsUnconstrained: DatatypeLeavesTotal,
	gap1ndex: DatatypeLeavesTotal,
	gap1ndexCeiling: DatatypeElephino,
	gapRangeStart: Array1DElephino,
	gapsWhere: Array1DLeavesTotal,
	groupsOfFolds: DatatypeFoldsTotal,
	indexDimension: DatatypeLeavesTotal,
	indexMiniGap: DatatypeElephino,
	leaf1ndex: DatatypeElephino,
	leafAbove: Array1DLeavesTotal,
	leafBelow: Array1DLeavesTotal,
	leafConnectee: DatatypeElephino,
	leavesTotal: DatatypeLeavesTotal,
	) -> DatatypeFoldsTotal:

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
	return groupsOfFolds

def flattenData(state: ComputationState) -> ComputationState:

	if state.taskDivisions > 0:
		return doTheNeedful(state)

	state = countInitialize(state)

	connectionGraph: Array3D = state.connectionGraph
	countDimensionsGapped: Array1DLeavesTotal = state.countDimensionsGapped
	dimensionsTotal: DatatypeLeavesTotal = state.dimensionsTotal
	dimensionsUnconstrained: DatatypeLeavesTotal = state.dimensionsUnconstrained
	gap1ndex: DatatypeLeavesTotal = state.gap1ndex
	gap1ndexCeiling: DatatypeElephino = state.gap1ndexCeiling
	gapRangeStart: Array1DElephino = state.gapRangeStart
	gapsWhere: Array1DLeavesTotal = state.gapsWhere
	groupsOfFolds: DatatypeFoldsTotal = state.groupsOfFolds
	indexDimension: DatatypeLeavesTotal = state.indexDimension
	indexMiniGap: DatatypeElephino = state.indexMiniGap
	leaf1ndex: DatatypeElephino = state.leaf1ndex
	leafAbove: Array1DLeavesTotal = state.leafAbove
	leafBelow: Array1DLeavesTotal = state.leafBelow
	leafConnectee: DatatypeElephino = state.leafConnectee
	leavesTotal: DatatypeLeavesTotal = state.leavesTotal

	groupsOfFolds = countSequential(
		connectionGraph = connectionGraph,
		countDimensionsGapped = countDimensionsGapped,
		dimensionsTotal = dimensionsTotal,
		dimensionsUnconstrained = dimensionsUnconstrained,
		gap1ndex = gap1ndex,
		gap1ndexCeiling = gap1ndexCeiling,
		gapRangeStart = gapRangeStart,
		gapsWhere = gapsWhere,
		groupsOfFolds = groupsOfFolds,
		indexDimension = indexDimension,
		indexMiniGap = indexMiniGap,
		leaf1ndex = leaf1ndex,
		leafAbove = leafAbove,
		leafBelow = leafBelow,
		leafConnectee = leafConnectee,
		leavesTotal = leavesTotal,
		)

	state.groupsOfFolds = state.foldGroups[state.taskIndex] = groupsOfFolds

	return state
