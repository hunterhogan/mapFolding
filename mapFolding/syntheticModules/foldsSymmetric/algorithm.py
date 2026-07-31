from mapFolding.dataBaskets import SymmetricFoldsState

def filterAsymmetricFolds(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.indexLeaf = 1
    state.leafComparison[0] = 1
    state.leafConnectee = 1
    while state.leafConnectee < state.leavesTotal + 1:
        state.indexMiniGap = state.leafBelow[state.indexLeaf]
        state.leafComparison[state.leafConnectee] = (state.indexMiniGap - state.indexLeaf + state.leavesTotal) % state.leavesTotal
        state.indexLeaf = state.indexMiniGap
        state.leafConnectee += 1
    for listTuples in state.indices:
        state.leafConnectee = 1
        for indexLeft, indexRight in listTuples:
            if state.leafComparison[indexLeft] != state.leafComparison[indexRight]:
                state.leafConnectee = 0
                break
        state.symmetricFolds += state.leafConnectee
    return state

def activeLeafGreaterThan0(state: SymmetricFoldsState) -> bool:
    return state.leaf1ndex > 0

def activeLeafGreaterThanLeavesTotal(state: SymmetricFoldsState) -> bool:
    return state.leaf1ndex > state.leavesTotal

def activeLeafIsTheFirstLeaf(state: SymmetricFoldsState) -> bool:
    return state.leaf1ndex <= 1

def activeLeafIsUnconstrainedInAllDimensions(state: SymmetricFoldsState) -> bool:
    return not state.dimensionsUnconstrained

def activeLeafUnconstrainedInThisDimension(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.dimensionsUnconstrained -= 1
    return state

def filterCommonGaps(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.indexMiniGap]
    if state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] == state.dimensionsUnconstrained:
        state = incrementActiveGap(state)
    state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] = 0
    return state

def gapAvailable(state: SymmetricFoldsState) -> bool:
    return state.leaf1ndex > 0

def incrementActiveGap(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.gap1ndex += 1
    return state

def incrementGap1ndexCeiling(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.gap1ndexCeiling += 1
    return state

def incrementIndexMiniGap(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.indexMiniGap += 1
    return state

def initializeIndexMiniGap(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.indexMiniGap = state.gap1ndex
    return state

def initializeVariablesToFindGaps(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.dimensionsUnconstrained = state.dimensionsTotal
    state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
    state.indexDimension = 0
    return state

def insertActiveLeaf(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.indexLeaf = 0
    while state.indexLeaf < state.leaf1ndex:
        state.gapsWhere[state.gap1ndexCeiling] = state.indexLeaf
        state.gap1ndexCeiling += 1
        state.indexLeaf += 1
    return state

def insertActiveLeafAtGap(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.gap1ndex -= 1
    state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
    state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
    state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
    state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
    state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
    state.leaf1ndex += 1
    return state

def leafBelowSentinelIs1(state: SymmetricFoldsState) -> bool:
    return state.leafBelow[0] == 1

def leafConnecteeIsActiveLeaf(state: SymmetricFoldsState) -> bool:
    return state.leafConnectee == state.leaf1ndex

def lookForGaps(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
    if state.countDimensionsGapped[state.leafConnectee] == 0:
        state = incrementGap1ndexCeiling(state)
    state.countDimensionsGapped[state.leafConnectee] += 1
    return state

def lookupLeafConnecteeInConnectionGraph(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leaf1ndex]
    return state

def loopingLeavesConnectedToActiveLeaf(state: SymmetricFoldsState) -> bool:
    return state.leafConnectee != state.leaf1ndex

def loopingThroughTheDimensions(state: SymmetricFoldsState) -> bool:
    return state.indexDimension < state.dimensionsTotal

def loopingToActiveGapCeiling(state: SymmetricFoldsState) -> bool:
    return state.indexMiniGap < state.gap1ndexCeiling

def noGapsHere(state: SymmetricFoldsState) -> bool:
    return state.leaf1ndex > 0 and state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]

def tryAnotherLeafConnectee(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
    return state

def tryNextDimension(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.indexDimension += 1
    return state

def undoLastLeafPlacement(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.leaf1ndex -= 1
    state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
    state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
    return state

def count(state: SymmetricFoldsState) -> SymmetricFoldsState:
    while activeLeafGreaterThan0(state):
        if activeLeafIsTheFirstLeaf(state) or leafBelowSentinelIs1(state):
            if activeLeafGreaterThanLeavesTotal(state):
                state = filterAsymmetricFolds(state)
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
    state.symmetricFolds = (state.symmetricFolds + 1) // 2
    return state

def doTheNeedful(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state = count(state)
    return state
