from __future__ import annotations

from mapFolding.dataBaskets import SymmetricFoldsState
from mapFolding.syntheticModules.foldsSymmetric.initializeState import transitionOnGroupsOfFolds

def count(state: SymmetricFoldsState) -> SymmetricFoldsState:
    while state.leaf1ndex > 4:
        if state.leafBelow[0] == 1:
            if state.leaf1ndex > state.leavesTotal:
                state.次Leaf = 1
                state.leafComparison[0] = 1
                state.leafConnectee = 1
                while state.leafConnectee < state.leavesTotal + 1:
                    state.次MiniGap = state.leafBelow[state.次Leaf]
                    state.leafComparison[state.leafConnectee] = (state.次MiniGap - state.次Leaf + state.leavesTotal) % state.leavesTotal
                    state.次Leaf = state.次MiniGap
                    state.leafConnectee += 1
                for boxOfTuples in state.indices:
                    state.leafConnectee = 1
                    for 次Left, 次Right in boxOfTuples:
                        if state.leafComparison[次Left] != state.leafComparison[次Right]:
                            state.leafConnectee = 0
                            break
                    state.symmetricFolds += state.leafConnectee
            else:
                state.dimensionsUnconstrained = state.dimensionsTotal
                state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
                state.次Dimension = 0
                while state.次Dimension < state.dimensionsTotal:
                    state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leaf1ndex]
                    if state.leafConnectee == state.leaf1ndex:
                        state.dimensionsUnconstrained -= 1
                    else:
                        while state.leafConnectee != state.leaf1ndex:
                            state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
                            if state.countDimensionsGapped[state.leafConnectee] == 0:
                                state.gap1ndexCeiling += 1
                            state.countDimensionsGapped[state.leafConnectee] += 1
                            state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
                    state.次Dimension += 1
                state.次MiniGap = state.gap1ndex
                while state.次MiniGap < state.gap1ndexCeiling:
                    state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.次MiniGap]
                    if state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] == state.dimensionsUnconstrained:
                        state.gap1ndex += 1
                    state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] = 0
                    state.次MiniGap += 1
        while state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]:
            state.leaf1ndex -= 1
            state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
            state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
        state.gap1ndex -= 1
        state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
        state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
        state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
        state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
        state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
        state.leaf1ndex += 1
    else:
        state.symmetricFolds *= 2
    state.symmetricFolds = (state.symmetricFolds + 1) // 2
    return state

def doTheNeedful(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state = transitionOnGroupsOfFolds(state)
    state = count(state)
    return state
