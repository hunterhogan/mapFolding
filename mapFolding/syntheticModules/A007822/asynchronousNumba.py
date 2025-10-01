"""Asynchronous map folding algorithm with Numba JIT compilation."""

from mapFolding.dataBaskets import (
	Array1DElephino,
	Array1DLeavesTotal,
	Array3DLeavesTotal,
	DatatypeElephino,
	DatatypeFoldsTotal,
	DatatypeLeavesTotal,
	MapFoldingState,
)
from mapFolding.syntheticModules.A007822.asynchronousAnnexNumba import (
	getLeafBelowSender,
	getSymmetricFoldsTotal,
	initializeConcurrencyManager,
)
from mapFolding.syntheticModules.A007822.initializeState import (
	transitionOnGroupsOfFolds,
)
from mapFolding.syntheticModules.A007822.leafBelowSender import LeafBelowSender
from numba import jit

@jit(cache=True, error_model='numpy', fastmath=True, forceinline=True)
def count(
	groupsOfFolds: DatatypeFoldsTotal,
	gap1ndex: DatatypeElephino,
	gap1ndexCeiling: DatatypeElephino,
	indexDimension: DatatypeLeavesTotal,
	indexMiniGap: DatatypeElephino,
	leaf1ndex: DatatypeLeavesTotal,
	leafConnectee: DatatypeLeavesTotal,
	dimensionsUnconstrained: DatatypeLeavesTotal,
	countDimensionsGapped: Array1DLeavesTotal,
	gapRangeStart: Array1DElephino,
	gapsWhere: Array1DLeavesTotal,
	leafAbove: Array1DLeavesTotal,
	leafBelow: Array1DLeavesTotal,
	connectionGraph: Array3DLeavesTotal,
	dimensionsTotal: DatatypeLeavesTotal,
	leavesTotal: DatatypeLeavesTotal,
	leafBelowSender: LeafBelowSender,
) -> tuple[
	DatatypeFoldsTotal,
	DatatypeElephino,
	DatatypeElephino,
	DatatypeLeavesTotal,
	DatatypeElephino,
	DatatypeLeavesTotal,
	DatatypeLeavesTotal,
	DatatypeLeavesTotal,
	Array1DLeavesTotal,
	Array1DElephino,
	Array1DLeavesTotal,
	Array1DLeavesTotal,
	Array1DLeavesTotal,
	Array3DLeavesTotal,
	DatatypeLeavesTotal,
	DatatypeLeavesTotal,
]:
	"""Core counting algorithm for map folding patterns.

	Parameters
	----------
	groupsOfFolds : DatatypeFoldsTotal
		Current count of distinct folding pattern groups
	gap1ndex : DatatypeElephino
		Current 1-indexed position of the gap during computation
	gap1ndexCeiling : DatatypeElephino
		Upper bound of gap1ndex
	indexDimension : DatatypeLeavesTotal
		Current 0-indexed position of the dimension during computation
	indexMiniGap : DatatypeElephino
		Index for mini gap iteration
	leaf1ndex : DatatypeLeavesTotal
		Current 1-indexed leaf position
	leafConnectee : DatatypeLeavesTotal
		Leaf being connected
	dimensionsUnconstrained : DatatypeLeavesTotal
		Number of unconstrained dimensions
	countDimensionsGapped : Array1DLeavesTotal
		Array tracking gapped dimensions count
	gapRangeStart : Array1DElephino
		Array tracking gap range starts
	gapsWhere : Array1DLeavesTotal
		Array tracking gap locations
	leafAbove : Array1DLeavesTotal
		Array tracking leaf above connections
	leafBelow : Array1DLeavesTotal
		Array tracking leaf below connections
	connectionGraph : Array3DLeavesTotal
		3D array representing leaf connections
	dimensionsTotal : DatatypeLeavesTotal
		Total number of dimensions
	leavesTotal : DatatypeLeavesTotal
		Total number of leaves
	leafBelowSender : LeafBelowSender
		Sender for passing leafBelow arrays to processing thread

	Returns
	-------
	tuple
		All state variables for continued computation
	"""
	while leaf1ndex > 4:
		if leafBelow[0] == 1:
			if leaf1ndex > leavesTotal:
				leafBelowSender.push(leafBelow)
			else:
				dimensionsUnconstrained = dimensionsTotal
				gap1ndexCeiling = gapRangeStart[leaf1ndex - 1]
				indexDimension = 0
				while indexDimension < dimensionsTotal:
					leafConnectee = connectionGraph[
						indexDimension, leaf1ndex, leaf1ndex
					]
					if leafConnectee == leaf1ndex:
						dimensionsUnconstrained -= 1
					else:
						while leafConnectee != leaf1ndex:
							gapsWhere[gap1ndexCeiling] = leafConnectee
							if countDimensionsGapped[leafConnectee] == 0:
								gap1ndexCeiling += 1
							countDimensionsGapped[leafConnectee] += 1
							leafConnectee = connectionGraph[
								indexDimension,
								leaf1ndex,
								leafBelow[leafConnectee],
							]
					indexDimension += 1
				indexMiniGap = gap1ndex
				while indexMiniGap < gap1ndexCeiling:
					gapsWhere[gap1ndex] = gapsWhere[indexMiniGap]
					gappedDimensionCount = countDimensionsGapped[
						gapsWhere[indexMiniGap]
					]
					if gappedDimensionCount == dimensionsUnconstrained:
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
	return (
		groupsOfFolds,
		gap1ndex,
		gap1ndexCeiling,
		indexDimension,
		indexMiniGap,
		leaf1ndex,
		leafConnectee,
		dimensionsUnconstrained,
		countDimensionsGapped,
		gapRangeStart,
		gapsWhere,
		leafAbove,
		leafBelow,
		connectionGraph,
		dimensionsTotal,
		leavesTotal,
	)

def doTheNeedful(
	state: MapFoldingState, maxWorkers: int | None = None
) -> MapFoldingState:
	"""Execute the map folding algorithm with asynchronous processing.

	Parameters
	----------
	state : MapFoldingState
		Current computational state
	maxWorkers : int | None
		Number of worker threads (default: 4)

	Returns
	-------
	MapFoldingState
		Updated computational state after processing
	"""
	state = transitionOnGroupsOfFolds(state)
	initializeConcurrencyManager(maxWorkers or 4, 0)
	state.groupsOfFolds = 0
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
	connectionGraph: Array3DLeavesTotal = state.connectionGraph
	dimensionsTotal: DatatypeLeavesTotal = state.dimensionsTotal
	leavesTotal: DatatypeLeavesTotal = state.leavesTotal
	leafBelowSender = getLeafBelowSender()
	(
		groupsOfFolds,
		gap1ndex,
		gap1ndexCeiling,
		indexDimension,
		indexMiniGap,
		leaf1ndex,
		leafConnectee,
		dimensionsUnconstrained,
		countDimensionsGapped,
		gapRangeStart,
		gapsWhere,
		leafAbove,
		leafBelow,
		connectionGraph,
		dimensionsTotal,
		leavesTotal,
	) = count(
		groupsOfFolds,
		gap1ndex,
		gap1ndexCeiling,
		indexDimension,
		indexMiniGap,
		leaf1ndex,
		leafConnectee,
		dimensionsUnconstrained,
		countDimensionsGapped,
		gapRangeStart,
		gapsWhere,
		leafAbove,
		leafBelow,
		connectionGraph,
		dimensionsTotal,
		leavesTotal,
		leafBelowSender,
	)
	groupsOfFolds = getSymmetricFoldsTotal()
	groupsOfFolds = (groupsOfFolds + 1) // 2
	return MapFoldingState(
		mapShape=mapShape,
		groupsOfFolds=groupsOfFolds,
		gap1ndex=gap1ndex,
		gap1ndexCeiling=gap1ndexCeiling,
		indexDimension=indexDimension,
		indexLeaf=indexLeaf,
		indexMiniGap=indexMiniGap,
		leaf1ndex=leaf1ndex,
		leafConnectee=leafConnectee,
		dimensionsUnconstrained=dimensionsUnconstrained,
		countDimensionsGapped=countDimensionsGapped,
		gapRangeStart=gapRangeStart,
		gapsWhere=gapsWhere,
		leafAbove=leafAbove,
		leafBelow=leafBelow,
		leafComparison=leafComparison,
	)
