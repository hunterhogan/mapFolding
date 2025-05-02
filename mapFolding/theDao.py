"""
Core computational algorithm for map folding counting and enumeration.

This module implements the core algorithms for enumerating and counting the various ways a rectangular map can be
folded. It uses a functional state-transformation approach, where each function performs a specific state mutation and
returns the updated state. The module provides three main counting algorithms:

1. `countInitialize`: Sets up the initial state for computation.
2. `countSequential`: Processes the folding computation sequentially.
3. `countParallel`: Distributes the computation across multiple processes.

All algorithms operate on a `ComputationState` object that tracks the folding process, including:
- A "leaf" is a unit square in the map.
- A "gap" is a potential position where a new leaf can be folded.
- Connections track how leaves can connect above/below each other.
- Leaves are enumerated starting from 1, not 0; hence, `leaf1ndex` not `leafIndex`.

The `doTheNeedful` function is the main entry point that orchestrates the computation strategy based on task divisions and
concurrency parameters.
"""
from concurrent.futures import Future as ConcurrentFuture, ProcessPoolExecutor
from copy import deepcopy
from mapFolding.beDRY import ComputationState
from multiprocessing import set_start_method as multiprocessing_set_start_method

# When to use multiprocessing.set_start_method https://github.com/hunterhogan/mapFolding/issues/6
if __name__ == '__main__':
	multiprocessing_set_start_method('spawn')

def activeLeafConnectedToItself(state: ComputationState) -> bool:
	return state.leafConnectee == state.leaf1ndex

def activeLeafGreaterThan0(state: ComputationState) -> bool:
	return state.leaf1ndex > 0

def activeLeafGreaterThanLeavesTotal(state: ComputationState) -> bool:
	return state.leaf1ndex > state.leavesTotal

def activeLeafIsTheFirstLeaf(state: ComputationState) -> bool:
	return state.leaf1ndex <= 1

def allDimensionsAreUnconstrained(state: ComputationState) -> bool:
	return not state.dimensionsUnconstrained

def countGaps(state: ComputationState) -> ComputationState:
	state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
	if state.countDimensionsGapped[state.leafConnectee] == 0:
		state = incrementGap1ndexCeiling(state)
	state.countDimensionsGapped[state.leafConnectee] += 1
	return state

def decrementDimensionsUnconstrained(state: ComputationState) -> ComputationState:
	state.dimensionsUnconstrained -= 1
	return state

def dimensionsUnconstrainedCondition(state: ComputationState) -> bool:
	return state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leaf1ndex] == state.leaf1ndex

def filterCommonGaps(state: ComputationState) -> ComputationState:
	state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.indexMiniGap]
	if state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] == state.dimensionsUnconstrained:
		state = incrementActiveGap(state)
	state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] = 0
	return state

def incrementActiveGap(state: ComputationState) -> ComputationState:
	state.gap1ndex += 1
	return state

def incrementGap1ndexCeiling(state: ComputationState) -> ComputationState:
	state.gap1ndexCeiling += 1
	return state

def incrementIndexDimension(state: ComputationState) -> ComputationState:
	state.indexDimension += 1
	return state

def incrementIndexMiniGap(state: ComputationState) -> ComputationState:
	state.indexMiniGap += 1
	return state

def initializeIndexMiniGap(state: ComputationState) -> ComputationState:
	state.indexMiniGap = state.gap1ndex
	return state

def initializeLeafConnectee(state: ComputationState) -> ComputationState:
	state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leaf1ndex]
	return state

def initializeVariablesToFindGaps(state: ComputationState) -> ComputationState:
	state.dimensionsUnconstrained = state.dimensionsTotal
	state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
	state.indexDimension = 0
	return state

def insertLeafAtGap(state: ComputationState) -> ComputationState:
	state.gap1ndex -= 1
	state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
	state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
	state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
	state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
	state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
	state.leaf1ndex += 1
	return state

def insertUnconstrainedLeaf(state: ComputationState) -> ComputationState:
	state.indexLeaf = 0
	while state.indexLeaf < state.leaf1ndex:
		state.gapsWhere[state.gap1ndexCeiling] = state.indexLeaf
		state.gap1ndexCeiling += 1
		state.indexLeaf += 1
	return state

def leafBelowSentinelIs1(state: ComputationState) -> bool:
	return state.leafBelow[0] == 1

def loopingLeavesConnectedToActiveLeaf(state: ComputationState) -> bool:
	return state.leafConnectee != state.leaf1ndex

def loopingToActiveGapCeiling(state: ComputationState) -> bool:
	return state.indexMiniGap < state.gap1ndexCeiling

def loopUpToDimensionsTotal(state: ComputationState) -> bool:
	return state.indexDimension < state.dimensionsTotal

def noGapsHere(state: ComputationState) -> bool:
	return (state.leaf1ndex > 0) and (state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1])

def thereIsAnActiveLeaf(state: ComputationState) -> bool:
	return state.leaf1ndex > 0

def thisIsMyTaskIndex(state: ComputationState) -> bool:
	return (state.leaf1ndex != state.taskDivisions) or (state.leafConnectee % state.taskDivisions == state.taskIndex)

def undoLastLeafPlacement(state: ComputationState) -> ComputationState:
	state.leaf1ndex -= 1
	state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
	state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
	return state

def updateLeafConnectee(state: ComputationState) -> ComputationState:
	state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
	return state

def countInitialize(state: ComputationState) -> ComputationState:
	while state.gap1ndex == 0:
		if activeLeafIsTheFirstLeaf(state) or leafBelowSentinelIs1(state):
				state = initializeVariablesToFindGaps(state)
				while loopUpToDimensionsTotal(state):
					state = initializeLeafConnectee(state)
					if activeLeafConnectedToItself(state):
						state = decrementDimensionsUnconstrained(state)
					else:
						while loopingLeavesConnectedToActiveLeaf(state):
							state = countGaps(state)
							state = updateLeafConnectee(state)
					state = incrementIndexDimension(state)
				if allDimensionsAreUnconstrained(state):
					state = insertUnconstrainedLeaf(state)
				state = initializeIndexMiniGap(state)
				while loopingToActiveGapCeiling(state):
					state = filterCommonGaps(state)
					state = incrementIndexMiniGap(state)
		if thereIsAnActiveLeaf(state):
			state = insertLeafAtGap(state)
	return state

def countParallel(state: ComputationState) -> ComputationState:
	while activeLeafGreaterThan0(state):
		if activeLeafIsTheFirstLeaf(state) or leafBelowSentinelIs1(state):
			if activeLeafGreaterThanLeavesTotal(state):
				state.groupsOfFolds += 1
			else:
				state = initializeVariablesToFindGaps(state)
				while loopUpToDimensionsTotal(state):
					if dimensionsUnconstrainedCondition(state):
						state = decrementDimensionsUnconstrained(state)
					else:
						state = initializeLeafConnectee(state)
						while loopingLeavesConnectedToActiveLeaf(state):
							if thisIsMyTaskIndex(state):
								state = countGaps(state)
							state = updateLeafConnectee(state)
					state = incrementIndexDimension(state)
				state = initializeIndexMiniGap(state)
				while loopingToActiveGapCeiling(state):
					state = filterCommonGaps(state)
					state = incrementIndexMiniGap(state)
		while noGapsHere(state):
			state = undoLastLeafPlacement(state)
		if thereIsAnActiveLeaf(state):
			state = insertLeafAtGap(state)
	state.foldGroups[state.taskIndex] = state.groupsOfFolds
	return state

def countSequential(state: ComputationState) -> ComputationState:
	while activeLeafGreaterThan0(state):
		if activeLeafIsTheFirstLeaf(state) or leafBelowSentinelIs1(state):
			if activeLeafGreaterThanLeavesTotal(state):
				state.groupsOfFolds += 1
			else:
				state = initializeVariablesToFindGaps(state)
				while loopUpToDimensionsTotal(state):
					state = initializeLeafConnectee(state)
					if activeLeafConnectedToItself(state):
						state = decrementDimensionsUnconstrained(state)
					else:
						while loopingLeavesConnectedToActiveLeaf(state):
							state = countGaps(state)
							state = updateLeafConnectee(state)
					state = incrementIndexDimension(state)
				state = initializeIndexMiniGap(state)
				while loopingToActiveGapCeiling(state):
					state = filterCommonGaps(state)
					state = incrementIndexMiniGap(state)
		while noGapsHere(state):
			state = undoLastLeafPlacement(state)
		if state.leaf1ndex == 3 and state.groupsOfFolds:
			state.groupsOfFolds *= 2
			# print('break')
			break
		if thereIsAnActiveLeaf(state):
			state = insertLeafAtGap(state)
	state.foldGroups[state.taskIndex] = state.groupsOfFolds
	return state

def doTheNeedful(state: ComputationState) -> ComputationState:
	state = countInitialize(state)
	if state.taskDivisions > 0:
		dictionaryConcurrency: dict[int, ConcurrentFuture[ComputationState]] = {}
		stateParallel = deepcopy(state)
		with ProcessPoolExecutor(stateParallel.concurrencyLimit) as concurrencyManager:
			for indexSherpa in range(stateParallel.taskDivisions):
				state = deepcopy(stateParallel)
				state.taskIndex = indexSherpa
				dictionaryConcurrency[indexSherpa] = concurrencyManager.submit(countParallel, state)
			for indexSherpa in range(stateParallel.taskDivisions):
				stateParallel.foldGroups[indexSherpa] = dictionaryConcurrency[indexSherpa].result().foldGroups[indexSherpa]
		state = stateParallel
	else:
		state = countSequential(state)

	return state
