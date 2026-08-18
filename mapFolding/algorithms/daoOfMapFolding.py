from __future__ import annotations

from mapFolding.dataBaskets import StateMapFolding

def activeLeafGreaterThan0(state: StateMapFolding) -> bool:
	return state.leaf1ndex > 0

def activeLeafGreaterThanTotalLeaves(state: StateMapFolding) -> bool:
	return state.leaf1ndex > state.totalLeaves

def activeLeafIsTheFirstLeaf(state: StateMapFolding) -> bool:
	return state.leaf1ndex <= 1

def activeLeafIsUnconstrainedInAllDimensions(state: StateMapFolding) -> bool:
	return not state.dimensionsUnconstrained

def activeLeafUnconstrainedInThisDimension(state: StateMapFolding) -> StateMapFolding:
	state.dimensionsUnconstrained -= 1
	return state

def filterCommonGaps(state: StateMapFolding) -> StateMapFolding:
	state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.次MiniGap]
	if state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] == state.dimensionsUnconstrained:
		state = incrementActiveGap(state)
	state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] = 0
	return state

def gapAvailable(state: StateMapFolding) -> bool:
	return state.leaf1ndex > 0

def incrementActiveGap(state: StateMapFolding) -> StateMapFolding:
	state.gap1ndex += 1
	return state

def incrementGap1ndexCeiling(state: StateMapFolding) -> StateMapFolding:
	state.gap1ndexCeiling += 1
	return state

def incrementIndexMiniGap(state: StateMapFolding) -> StateMapFolding:
	state.次MiniGap += 1
	return state

def initializeIndexMiniGap(state: StateMapFolding) -> StateMapFolding:
	state.次MiniGap = state.gap1ndex
	return state

def initializeVariablesToFindGaps(state: StateMapFolding) -> StateMapFolding:
	state.dimensionsUnconstrained = state.totalDimensions
	state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
	state.次Dimension = 0
	return state

def insertActiveLeaf(state: StateMapFolding) -> StateMapFolding:
	state.次Leaf = 0
	while state.次Leaf < state.leaf1ndex:
		state.gapsWhere[state.gap1ndexCeiling] = state.次Leaf
		state.gap1ndexCeiling += 1
		state.次Leaf += 1
	return state

def insertActiveLeafAtGap(state: StateMapFolding) -> StateMapFolding:
	state.gap1ndex -= 1
	state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
	state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
	state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
	state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
	state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
	state.leaf1ndex += 1
	return state

def leafBelowSentinelIs1(state: StateMapFolding) -> bool:
	return state.leafBelow[0] == 1

def leafConnecteeIsActiveLeaf(state: StateMapFolding) -> bool:
	return state.leafConnectee == state.leaf1ndex

def lookForGaps(state: StateMapFolding) -> StateMapFolding:
	state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
	if state.countDimensionsGapped[state.leafConnectee] == 0:
		state = incrementGap1ndexCeiling(state)
	state.countDimensionsGapped[state.leafConnectee] += 1
	return state

def lookupLeafConnecteeInConnectionGraph(state: StateMapFolding) -> StateMapFolding:
	state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leaf1ndex]
	return state

def loopingLeavesConnectedToActiveLeaf(state: StateMapFolding) -> bool:
	return state.leafConnectee != state.leaf1ndex

def loopingThroughTheDimensions(state: StateMapFolding) -> bool:
	return state.次Dimension < state.totalDimensions

def loopingToActiveGapCeiling(state: StateMapFolding) -> bool:
	return state.次MiniGap < state.gap1ndexCeiling

def noGapsHere(state: StateMapFolding) -> bool:
	return (state.leaf1ndex > 0) and (state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1])

def tryAnotherLeafConnectee(state: StateMapFolding) -> StateMapFolding:
	state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
	return state

def tryNextDimension(state: StateMapFolding) -> StateMapFolding:
	state.次Dimension += 1
	return state

def undoLastLeafPlacement(state: StateMapFolding) -> StateMapFolding:
	state.leaf1ndex -= 1
	state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
	state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
	return state

def count(state: StateMapFolding) -> StateMapFolding:
	while activeLeafGreaterThan0(state):
		if activeLeafIsTheFirstLeaf(state) or leafBelowSentinelIs1(state):
			if activeLeafGreaterThanTotalLeaves(state):
				state.groupsOfFolds += 1
			else:
				state = initializeVariablesToFindGaps(state)
				while loopingThroughTheDimensions(state):
					state = lookupLeafConnecteeInConnectionGraph(state)
					if leafConnecteeIsActiveLeaf(state):
						state = activeLeafUnconstrainedInThisDimension(state)
					else:
						while loopingLeavesConnectedToActiveLeaf(state):
							state = lookForGaps(state)
							state = tryAnotherLeafConnectee(state)
					state = tryNextDimension(state)
				if activeLeafIsUnconstrainedInAllDimensions(state):
					state = insertActiveLeaf(state)
				state = initializeIndexMiniGap(state)
				while loopingToActiveGapCeiling(state):
					state = filterCommonGaps(state)
					state = incrementIndexMiniGap(state)
		while noGapsHere(state):
			state = undoLastLeafPlacement(state)
		if gapAvailable(state):
			state = insertActiveLeafAtGap(state)
	return state

def doTheNeedful(state: StateMapFolding) -> StateMapFolding:
	state = count(state)
	return state
