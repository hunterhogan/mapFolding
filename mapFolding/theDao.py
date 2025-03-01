from mapFolding import ComputationState
from numba import prange
from numpy import dtype, integer, ndarray
from typing import Any

def activeLeafConnectedToItself(state: ComputationState):
	return state.leafConnectee == state.leaf1ndex

def activeLeafGreaterThan0(state: ComputationState):
	return state.leaf1ndex > 0

def activeLeafGreaterThanLeavesTotal(state: ComputationState):
	return state.leaf1ndex > state.leavesTotal

def activeLeafIsTheFirstLeaf(state: ComputationState):
	return state.leaf1ndex <= 1

def allDimensionsAreUnconstrained(state: ComputationState):
	return not state.dimensionsUnconstrained

def backtrack(state: ComputationState):
	state.leaf1ndex -= 1
	state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
	state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
	return state

def countGaps(state: ComputationState):
	state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
	if state.countDimensionsGapped[state.leafConnectee] == 0:
		state = incrementGap1ndexCeiling(state)
	state.countDimensionsGapped[state.leafConnectee] += 1
	return state

def decrementDimensionsUnconstrained(state: ComputationState):
	state.dimensionsUnconstrained -= 1
	return state

def dimensionsUnconstrainedCondition(state: ComputationState):
	return state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leaf1ndex] == state.leaf1ndex

def filterCommonGaps(state: ComputationState):
	state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.indexMiniGap]
	if state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] == state.dimensionsUnconstrained:
		state = incrementActiveGap(state)
	state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] = 0
	return state

def incrementActiveGap(state: ComputationState):
	state.gap1ndex += 1
	return state

def incrementGap1ndexCeiling(state: ComputationState):
	state.gap1ndexCeiling += 1
	return state

def incrementIndexDimension(state: ComputationState):
	state.indexDimension += 1
	return state

def incrementIndexMiniGap(state: ComputationState):
	state.indexMiniGap += 1
	return state

def initializeIndexMiniGap(state: ComputationState):
	state.indexMiniGap = state.gap1ndex
	return state

def initializeLeafConnectee(state: ComputationState):
	state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leaf1ndex]
	return state

def initializeVariablesToFindGaps(state: ComputationState):
	state.dimensionsUnconstrained = state.dimensionsTotal
	state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
	state.indexDimension = 0
	return state

def insertUnconstrainedLeaf(state: ComputationState):
	indexLeaf = 0
	while indexLeaf < state.leaf1ndex:
		state.gapsWhere[state.gap1ndexCeiling] = indexLeaf
		state.gap1ndexCeiling += 1
		indexLeaf += 1
	return state

def leafBelowSentinelIs1(state: ComputationState):
	return state.leafBelow[0] == 1

def loopingLeavesConnectedToActiveLeaf(state: ComputationState):
	return state.leafConnectee != state.leaf1ndex

def loopingToActiveGapCeiling(state: ComputationState):
	return state.indexMiniGap < state.gap1ndexCeiling

def loopUpToDimensionsTotal(state: ComputationState):
	return state.indexDimension < state.dimensionsTotal

def noGapsHere(state: ComputationState):
	return state.leaf1ndex > 0 and state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]

def placeLeaf(state: ComputationState):
	state.gap1ndex -= 1
	state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
	state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
	state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
	state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
	state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
	state.leaf1ndex += 1
	return state

def thereIsAnActiveLeaf(state: ComputationState):
	return state.leaf1ndex > 0

def thisIsMyTaskIndex(state: ComputationState):
	return state.leaf1ndex != state.taskDivisions or state.leafConnectee % state.taskDivisions == state.taskIndex

def updateLeafConnectee(state: ComputationState):
	state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
	return state

def countInitialize(computationStateInitialized: ComputationState) -> ComputationState:
		# if state.gap1ndex > 0:
		# 	return
	return computationStateInitialized

def countParallel():
	# gapsWherePARALLEL = state.gapsWhere.copy(state: ComputationState)
	# myPARALLEL = my.copy(state: ComputationState)
	# trackPARALLEL = track.copy(state: ComputationState)
	# taskDivisionsPrange = myPARALLEL[state.taskDivisions]
	# for indexSherpa in prange(taskDivisionsPrange): # type: ignore
	# 	groupsOfFolds: int = 0
	# 	state.gapsWhere = gapsWherePARALLEL.copy(state: ComputationState)
	# 	my = myPARALLEL.copy(state: ComputationState)
	# 	track = trackPARALLEL.copy(state: ComputationState)
	# 	state.taskIndex = indexSherpa
	pass

def countSequential(state: ComputationState):
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
				if allDimensionsAreUnconstrained(state):
					state = insertUnconstrainedLeaf(state)
				state = initializeIndexMiniGap(state)
				while loopingToActiveGapCeiling(state):
					state = filterCommonGaps(state)
					state = incrementIndexMiniGap(state)
		while noGapsHere(state):
			state = backtrack(state)
		if thereIsAnActiveLeaf(state):
			state = placeLeaf(state)
	state.foldGroups[state.taskIndex] = state.groupsOfFolds
	return state

def doTheNeedful(computationStateInitialized: ComputationState):
	computationStateInitialized = countInitialize(computationStateInitialized)
	if computationStateInitialized.taskDivisions > 0:
		return countParallel()
	else:
		return countSequential(computationStateInitialized)
