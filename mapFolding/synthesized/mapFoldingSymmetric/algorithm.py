from __future__ import annotations

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
from mapFolding.dataBaskets import StateMapFoldingSymmetric

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def filterAsymmetricFolds(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次Leaf = 1
    state.leafComparison[0] = 1
    state.leafConnectee = 1
    while state.leafConnectee <= state.totalLeaves:
        state.次MiniGap = state.leafBelow[state.次Leaf]
        state.leafComparison[state.leafConnectee] = (state.totalLeaves + state.次MiniGap - state.次Leaf) % state.totalLeaves
        state.次Leaf = state.次MiniGap
        state.leafConnectee += 1
    for arrayOfTuples in state.indices:
        state.leafConnectee = 1
        for 次Left, 次Right in arrayOfTuples:
            if state.leafComparison[次Left] != state.leafComparison[次Right]:
                state.leafConnectee = 0
                break
        # TODO Make the A007822 folding relation explicit in this symmetry terminology.
        state.symmetricFolds += state.leafConnectee
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def activeLeafGreaterThan0(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > 0

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def activeLeafGreaterThanTotalLeaves(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > state.totalLeaves

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def activeLeafIsTheFirstLeaf(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex <= 1

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def activeLeafIsUnconstrainedInAllDimensions(state: StateMapFoldingSymmetric) -> bool:
    return not state.dimensionsUnconstrained

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def activeLeafUnconstrainedInThisDimension(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.dimensionsUnconstrained -= 1
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def filterCommonGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.次MiniGap]
    if state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] == state.dimensionsUnconstrained:
        state = incrementActiveGap(state)
    state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] = 0
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def gapAvailable(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > 0

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def incrementActiveGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gap1ndex += 1
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def incrementGap1ndexCeiling(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gap1ndexCeiling += 1
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def incrementIndexMiniGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次MiniGap += 1
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def initializeIndexMiniGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次MiniGap = state.gap1ndex
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def initializeVariablesToFindGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.dimensionsUnconstrained = state.totalDimensions
    state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
    state.次Dimension = 0
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def insertActiveLeaf(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次Leaf = 0
    while state.次Leaf < state.leaf1ndex:
        state.gapsWhere[state.gap1ndexCeiling] = state.次Leaf
        state.gap1ndexCeiling += 1
        state.次Leaf += 1
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def insertActiveLeafAtGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gap1ndex -= 1
    state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
    state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
    state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
    state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
    state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
    state.leaf1ndex += 1
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def leafBelowSentinelIs1(state: StateMapFoldingSymmetric) -> bool:
    return state.leafBelow[0] == 1

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def leafConnecteeIsActiveLeaf(state: StateMapFoldingSymmetric) -> bool:
    return state.leafConnectee == state.leaf1ndex

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def lookForGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
    if state.countDimensionsGapped[state.leafConnectee] == 0:
        state = incrementGap1ndexCeiling(state)
    state.countDimensionsGapped[state.leafConnectee] += 1
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def lookupLeafConnecteeInConnectionGraph(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leaf1ndex]
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def loopingLeavesConnectedToActiveLeaf(state: StateMapFoldingSymmetric) -> bool:
    return state.leafConnectee != state.leaf1ndex

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def loopingThroughTheDimensions(state: StateMapFoldingSymmetric) -> bool:
    return state.次Dimension < state.totalDimensions

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def loopingToActiveGapCeiling(state: StateMapFoldingSymmetric) -> bool:
    return state.次MiniGap < state.gap1ndexCeiling

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def noGapsHere(state: StateMapFoldingSymmetric) -> bool:
    return state.leaf1ndex > 0 and state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def tryAnotherLeafConnectee(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def tryNextDimension(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.次Dimension += 1
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def undoLastLeafPlacement(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state.leaf1ndex -= 1
    state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
    state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def count(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    while activeLeafGreaterThan0(state):
        if activeLeafIsTheFirstLeaf(state) or leafBelowSentinelIs1(state):
            if activeLeafGreaterThanTotalLeaves(state):
                # TODO Make the A007822 folding relation explicit in this symmetry terminology.
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
    # TODO Make the A007822 folding relation explicit in this symmetry terminology.
    state.symmetricFolds = (state.symmetricFolds + 1) // 2
    return state

# TODO Make the A007822 folding relation explicit in this symmetry terminology.
def doTheNeedful(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    state = count(state)
    return state
