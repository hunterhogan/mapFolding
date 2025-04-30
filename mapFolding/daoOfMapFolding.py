from mapFolding.dataBaskets import MapFoldingState

def activeLeafGreaterThan0(state: MapFoldingState) -> bool:
	return state.leaf1ndex > 0

def activeLeafGreaterThanLeavesTotal(state: MapFoldingState) -> bool:
	return state.leaf1ndex > state.leavesTotal

def activeLeafIsTheFirstLeaf(state: MapFoldingState) -> bool:
	return state.leaf1ndex <= 1

def activeLeafIsUnconstrainedInAllDimensions(state: MapFoldingState) -> bool:
	return not state.dimensionsUnconstrained

def activeLeafUnconstrainedInThisDimension(state: MapFoldingState) -> MapFoldingState:
	state.dimensionsUnconstrained -= 1
	return state

def filterCommonGaps(state: MapFoldingState) -> MapFoldingState:
	state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.indexMiniGap]
	if state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] == state.dimensionsUnconstrained:
		state = incrementActiveGap(state)
	state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] = 0
	return state

def gapAvailable(state: MapFoldingState) -> bool:
	return state.leaf1ndex > 0

def incrementActiveGap(state: MapFoldingState) -> MapFoldingState:
	state.gap1ndex += 1
	return state

def incrementGap1ndexCeiling(state: MapFoldingState) -> MapFoldingState:
	state.gap1ndexCeiling += 1
	return state

def incrementIndexMiniGap(state: MapFoldingState) -> MapFoldingState:
	state.indexMiniGap += 1
	return state

def initializeIndexMiniGap(state: MapFoldingState) -> MapFoldingState:
	state.indexMiniGap = state.gap1ndex
	return state

def initializeVariablesToFindGaps(state: MapFoldingState) -> MapFoldingState:
	state.dimensionsUnconstrained = state.dimensionsTotal
	state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
	state.indexDimension = 0
	return state

def insertActiveLeaf(state: MapFoldingState) -> MapFoldingState:
	state.indexLeaf = 0
	while state.indexLeaf < state.leaf1ndex:
		state.gapsWhere[state.gap1ndexCeiling] = state.indexLeaf
		state.gap1ndexCeiling += 1
		state.indexLeaf += 1
	return state

def insertActiveLeafAtGap(state: MapFoldingState) -> MapFoldingState:
	state.gap1ndex -= 1
	state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
	state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
	state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
	state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
	state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
	state.leaf1ndex += 1
	return state

def leafBelowSentinelIs1(state: MapFoldingState) -> bool:
	return state.leafBelow[0] == 1

def leafConnecteeIsActiveLeaf(state: MapFoldingState) -> bool:
	return state.leafConnectee == state.leaf1ndex

def lookForGaps(state: MapFoldingState) -> MapFoldingState:
	state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
	if state.countDimensionsGapped[state.leafConnectee] == 0:
		state = incrementGap1ndexCeiling(state)
	state.countDimensionsGapped[state.leafConnectee] += 1
	return state

def lookupLeafConnecteeInConnectionGraph(state: MapFoldingState) -> MapFoldingState:
	state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leaf1ndex]
	return state

def loopingLeavesConnectedToActiveLeaf(state: MapFoldingState) -> bool:
	return state.leafConnectee != state.leaf1ndex

def loopingThroughTheDimensions(state: MapFoldingState) -> bool:
	return state.indexDimension < state.dimensionsTotal

def loopingToActiveGapCeiling(state: MapFoldingState) -> bool:
	return state.indexMiniGap < state.gap1ndexCeiling

def noGapsHere(state: MapFoldingState) -> bool:
	return (state.leaf1ndex > 0) and (state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1])

def tryAnotherLeafConnectee(state: MapFoldingState) -> MapFoldingState:
	state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
	return state

def tryNextDimension(state: MapFoldingState) -> MapFoldingState:
	state.indexDimension += 1
	return state

def undoLastLeafPlacement(state: MapFoldingState) -> MapFoldingState:
	state.leaf1ndex -= 1
	state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
	state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
	return state

def count(state: MapFoldingState) -> MapFoldingState:
	while activeLeafGreaterThan0(state):
		if activeLeafIsTheFirstLeaf(state) or leafBelowSentinelIs1(state):
			if activeLeafGreaterThanLeavesTotal(state):
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

def doTheNeedful(state: MapFoldingState) -> MapFoldingState:
	state = count(state)
	return state
