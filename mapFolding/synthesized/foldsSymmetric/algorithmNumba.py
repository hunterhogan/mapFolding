from __future__ import annotations

from mapFolding.dataBaskets import StateMapFoldingSymmetric
from numba import jit_module

def filterAsymmetricFolds(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次Leaf = 1
    state.leafComparison[0] = 1
    state.leafConnectee = 1
    while state.leafConnectee <= state.totalLeaves:
        state.次MiniGap = state.leafBelow[state.次Leaf]
        state.leafComparison[state.leafConnectee] = (state.totalLeaves + state.次MiniGap - state.次Leaf) % state.totalLeaves
        state.次Leaf = state.次MiniGap
        state.leafConnectee += 1
    for boxOfTuples in state.indices:
        state.leafConnectee = 1
        for 次Left, 次Right in boxOfTuples:
            if state.leafComparison[次Left] != state.leafComparison[次Right]:
                state.leafConnectee = 0
                break
        state.symmetricFolds += state.leafConnectee
    return state

def activeLeafGreaterThan0(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > 0

def activeLeafGreaterThanTotalLeaves(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > state.totalLeaves

def activeLeafIsTheFirstLeaf(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex <= 1

def activeLeafIsUnconstrainedInAllDimensions(state: StateMapFoldingSymmetric) -> bool:
    return not state.dimensionsUnconstrained

def activeLeafUnconstrainedInThisDimension(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.dimensionsUnconstrained -= 1
    return state

def filterCommonGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.次MiniGap]
    if state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] == state.dimensionsUnconstrained:
        state = incrementActiveGap(state)
    state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] = 0
    return state

def gapAvailable(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > 0

def incrementActiveGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gap1ndex += 1
    return state

def incrementGap1ndexCeiling(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gap1ndexCeiling += 1
    return state

def incrementIndexMiniGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次MiniGap += 1
    return state

def initializeIndexMiniGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次MiniGap = state.gap1ndex
    return state

def initializeVariablesToFindGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.dimensionsUnconstrained = state.totalDimensions
    state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
    state.次Dimension = 0
    return state

def insertActiveLeaf(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次Leaf = 0
    while state.次Leaf < state.leaf1ndex:
        state.gapsWhere[state.gap1ndexCeiling] = state.次Leaf
        state.gap1ndexCeiling += 1
        state.次Leaf += 1
    return state

def insertActiveLeafAtGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gap1ndex -= 1
    state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
    state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
    state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
    state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
    state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
    state.leaf1ndex += 1
    return state

def leafBelowSentinelIs1(state: StateMapFoldingSymmetric) -> bool:
    return state.leafBelow[0] == 1

def leafConnecteeIsActiveLeaf(state: StateMapFoldingSymmetric) -> bool:
    return state.leafConnectee == state.leaf1ndex

def lookForGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
    if state.countDimensionsGapped[state.leafConnectee] == 0:
        state = incrementGap1ndexCeiling(state)
    state.countDimensionsGapped[state.leafConnectee] += 1
    return state

def lookupLeafConnecteeInConnectionGraph(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leaf1ndex]
    return state

def loopingLeavesConnectedToActiveLeaf(state: StateMapFoldingSymmetric) -> bool:
    return state.leafConnectee != state.leaf1ndex

def loopingThroughTheDimensions(state: StateMapFoldingSymmetric) -> bool:
    return state.次Dimension < state.totalDimensions

def loopingToActiveGapCeiling(state: StateMapFoldingSymmetric) -> bool:
    return state.次MiniGap < state.gap1ndexCeiling

def noGapsHere(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > 0 and state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]

def tryAnotherLeafConnectee(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
    return state

def tryNextDimension(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次Dimension += 1
    return state

def undoLastLeafPlacement(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.leaf1ndex -= 1
    state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
    state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
    return state

def count(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    while activeLeafGreaterThan0(state):
        if activeLeafIsTheFirstLeaf(state) or leafBelowSentinelIs1(state):
            if activeLeafGreaterThanTotalLeaves(state):
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

def doTheNeedful(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state = count(state)
    return state
jit_module(cache=True, error_model='numpy', fastmath=True, forceinline=True)
