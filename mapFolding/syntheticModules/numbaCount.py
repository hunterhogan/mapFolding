from concurrent.futures import Future as ConcurrentFuture, ProcessPoolExecutor
from copy import deepcopy
from mapFolding import ComputationState
from mapFolding import Array1DElephino, Array1DFoldsTotal, Array1DLeavesTotal, Array3D, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal
from numba import jit

def countInitialize(state: ComputationState) -> ComputationState:
	while state.gap1ndex == 0:
		if state.leaf1ndex <= 1 or state.leafBelow[0] == 1:
			state.dimensionsUnconstrained = state.dimensionsTotal
			state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
			state.indexDimension = 0
			while state.indexDimension < state.dimensionsTotal:
				state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leaf1ndex]
				if state.leafConnectee == state.leaf1ndex:
					state.dimensionsUnconstrained -= 1
				else:
					while state.leafConnectee != state.leaf1ndex:
						state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
						if state.countDimensionsGapped[state.leafConnectee] == 0:
							state.gap1ndexCeiling += 1
						state.countDimensionsGapped[state.leafConnectee] += 1
						state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
				state.indexDimension += 1
			if not state.dimensionsUnconstrained:
				state.indexLeaf = 0
				while state.indexLeaf < state.leaf1ndex:
					state.gapsWhere[state.gap1ndexCeiling] = state.indexLeaf
					state.gap1ndexCeiling += 1
					state.indexLeaf += 1
			state.indexMiniGap = state.gap1ndex
			while state.indexMiniGap < state.gap1ndexCeiling:
				state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.indexMiniGap]
				if state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] == state.dimensionsUnconstrained:
					state.gap1ndex += 1
				state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] = 0
				state.indexMiniGap += 1
		if state.leaf1ndex > 0:
			state.gap1ndex -= 1
			state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
			state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
			state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
			state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
			state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
			state.leaf1ndex += 1
	return state

@jit(_nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=False, no_cpython_wrapper=False, nopython=True, parallel=False)
def countParallel(leavesTotal: DatatypeLeavesTotal, taskDivisions: DatatypeLeavesTotal, connectionGraph: Array3D, dimensionsTotal: DatatypeLeavesTotal, countDimensionsGapped: Array1DLeavesTotal, dimensionsUnconstrained: DatatypeLeavesTotal, gapRangeStart: Array1DElephino, gapsWhere: Array1DLeavesTotal, leafAbove: Array1DLeavesTotal, leafBelow: Array1DLeavesTotal, foldGroups: Array1DFoldsTotal, gap1ndex: DatatypeElephino, gap1ndexCeiling: DatatypeElephino, groupsOfFolds: DatatypeFoldsTotal, indexDimension: DatatypeLeavesTotal, indexMiniGap: DatatypeElephino, leaf1ndex: DatatypeLeavesTotal, leafConnectee: DatatypeLeavesTotal, taskIndex: DatatypeLeavesTotal) -> DatatypeFoldsTotal:
	while leaf1ndex > 0:
		if leaf1ndex <= 1 or leafBelow[0] == 1:
			if leaf1ndex > leavesTotal:
				groupsOfFolds += 1
			else:
				dimensionsUnconstrained = dimensionsTotal
				gap1ndexCeiling = gapRangeStart[leaf1ndex - 1]
				indexDimension = 0
				while indexDimension < dimensionsTotal:
					if connectionGraph[indexDimension, leaf1ndex, leaf1ndex] == leaf1ndex:
						dimensionsUnconstrained -= 1
					else:
						leafConnectee = connectionGraph[indexDimension, leaf1ndex, leaf1ndex]
						while leafConnectee != leaf1ndex:
							if leaf1ndex != taskDivisions or leafConnectee % taskDivisions == taskIndex:
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
	foldGroups[taskIndex] = groupsOfFolds
	return groupsOfFolds

@jit(_nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=False, no_cpython_wrapper=False, nopython=True, parallel=False)
def countSequential(mapShape: tuple[DatatypeLeavesTotal, ...], leavesTotal: DatatypeLeavesTotal, taskDivisions: DatatypeLeavesTotal, concurrencyLimit: DatatypeElephino, connectionGraph: Array3D, dimensionsTotal: DatatypeLeavesTotal, countDimensionsGapped: Array1DLeavesTotal, dimensionsUnconstrained: DatatypeLeavesTotal, gapRangeStart: Array1DElephino, gapsWhere: Array1DLeavesTotal, leafAbove: Array1DLeavesTotal, leafBelow: Array1DLeavesTotal, foldGroups: Array1DFoldsTotal, foldsTotal: DatatypeFoldsTotal, gap1ndex: DatatypeElephino, gap1ndexCeiling: DatatypeElephino, groupsOfFolds: DatatypeFoldsTotal, indexDimension: DatatypeLeavesTotal, indexLeaf: DatatypeLeavesTotal, indexMiniGap: DatatypeElephino, leaf1ndex: DatatypeLeavesTotal, leafConnectee: DatatypeLeavesTotal, taskIndex: DatatypeLeavesTotal) -> tuple[tuple[DatatypeLeavesTotal, ...], DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeElephino, Array3D, DatatypeLeavesTotal, Array1DLeavesTotal, DatatypeLeavesTotal, Array1DElephino, Array1DLeavesTotal, Array1DLeavesTotal, Array1DLeavesTotal, Array1DFoldsTotal, DatatypeFoldsTotal, DatatypeElephino, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeElephino, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal]:
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
		if leaf1ndex == 3 and groupsOfFolds:
			groupsOfFolds *= 2
			break
		if leaf1ndex > 0:
			gap1ndex -= 1
			leafAbove[leaf1ndex] = gapsWhere[gap1ndex]
			leafBelow[leaf1ndex] = leafBelow[leafAbove[leaf1ndex]]
			leafBelow[leafAbove[leaf1ndex]] = leaf1ndex
			leafAbove[leafBelow[leaf1ndex]] = leaf1ndex
			gapRangeStart[leaf1ndex] = gap1ndex
			leaf1ndex += 1
	foldGroups[taskIndex] = groupsOfFolds
	return (mapShape, leavesTotal, taskDivisions, concurrencyLimit, connectionGraph, dimensionsTotal, countDimensionsGapped, dimensionsUnconstrained, gapRangeStart, gapsWhere, leafAbove, leafBelow, foldGroups, foldsTotal, gap1ndex, gap1ndexCeiling, groupsOfFolds, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, taskIndex)

def doTheNeedful(state: ComputationState) -> ComputationState:
	state = countInitialize(state)
	if state.taskDivisions > 0:
		dictionaryConcurrency: dict[int, ConcurrentFuture[ComputationState]] = {}
		stateParallel = deepcopy(state)
		with ProcessPoolExecutor(stateParallel.concurrencyLimit) as concurrencyManager:
			for indexSherpa in range(stateParallel.taskDivisions):
				state = deepcopy(stateParallel)
				state.taskIndex = indexSherpa
				mapShape: tuple[DatatypeLeavesTotal, ...] = state.mapShape
				leavesTotal: DatatypeLeavesTotal = state.leavesTotal
				taskDivisions: DatatypeLeavesTotal = state.taskDivisions
				concurrencyLimit: DatatypeElephino = state.concurrencyLimit
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
				gap1ndex: DatatypeElephino = state.gap1ndex
				gap1ndexCeiling: DatatypeElephino = state.gap1ndexCeiling
				groupsOfFolds: DatatypeFoldsTotal = state.groupsOfFolds
				indexDimension: DatatypeLeavesTotal = state.indexDimension
				indexLeaf: DatatypeLeavesTotal = state.indexLeaf
				indexMiniGap: DatatypeElephino = state.indexMiniGap
				leaf1ndex: DatatypeLeavesTotal = state.leaf1ndex
				leafConnectee: DatatypeLeavesTotal = state.leafConnectee
				taskIndex: DatatypeLeavesTotal = state.taskIndex
				dictionaryConcurrency[indexSherpa] = concurrencyManager.submit(countParallel, leavesTotal, taskDivisions, connectionGraph, dimensionsTotal, countDimensionsGapped, dimensionsUnconstrained, gapRangeStart, gapsWhere, leafAbove, leafBelow, foldGroups, gap1ndex, gap1ndexCeiling, groupsOfFolds, indexDimension, indexMiniGap, leaf1ndex, leafConnectee, taskIndex)
			for indexSherpa in range(stateParallel.taskDivisions):
				stateParallel.foldGroups[indexSherpa] = dictionaryConcurrency[indexSherpa].result()
		state = stateParallel
	else:
		mapShape: tuple[DatatypeLeavesTotal, ...] = state.mapShape
		leavesTotal: DatatypeLeavesTotal = state.leavesTotal
		taskDivisions: DatatypeLeavesTotal = state.taskDivisions
		concurrencyLimit: DatatypeElephino = state.concurrencyLimit
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
		gap1ndex: DatatypeElephino = state.gap1ndex
		gap1ndexCeiling: DatatypeElephino = state.gap1ndexCeiling
		groupsOfFolds: DatatypeFoldsTotal = state.groupsOfFolds
		indexDimension: DatatypeLeavesTotal = state.indexDimension
		indexLeaf: DatatypeLeavesTotal = state.indexLeaf
		indexMiniGap: DatatypeElephino = state.indexMiniGap
		leaf1ndex: DatatypeLeavesTotal = state.leaf1ndex
		leafConnectee: DatatypeLeavesTotal = state.leafConnectee
		taskIndex: DatatypeLeavesTotal = state.taskIndex
		mapShape, leavesTotal, taskDivisions, concurrencyLimit, connectionGraph, dimensionsTotal, countDimensionsGapped, dimensionsUnconstrained, gapRangeStart, gapsWhere, leafAbove, leafBelow, foldGroups, foldsTotal, gap1ndex, gap1ndexCeiling, groupsOfFolds, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, taskIndex = countSequential(mapShape, leavesTotal, taskDivisions, concurrencyLimit, connectionGraph, dimensionsTotal, countDimensionsGapped, dimensionsUnconstrained, gapRangeStart, gapsWhere, leafAbove, leafBelow, foldGroups, foldsTotal, gap1ndex, gap1ndexCeiling, groupsOfFolds, indexDimension, indexLeaf, indexMiniGap, leaf1ndex, leafConnectee, taskIndex)
		state = ComputationState(mapShape=mapShape, leavesTotal=leavesTotal, taskDivisions=taskDivisions, concurrencyLimit=concurrencyLimit, countDimensionsGapped=countDimensionsGapped, dimensionsUnconstrained=dimensionsUnconstrained, gapRangeStart=gapRangeStart, gapsWhere=gapsWhere, leafAbove=leafAbove, leafBelow=leafBelow, foldGroups=foldGroups, foldsTotal=foldsTotal, gap1ndex=gap1ndex, gap1ndexCeiling=gap1ndexCeiling, groupsOfFolds=groupsOfFolds, indexDimension=indexDimension, indexLeaf=indexLeaf, indexMiniGap=indexMiniGap, leaf1ndex=leaf1ndex, leafConnectee=leafConnectee, taskIndex=taskIndex)
	return state
