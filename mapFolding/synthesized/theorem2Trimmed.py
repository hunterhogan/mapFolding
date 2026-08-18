from __future__ import annotations

from mapFolding.dataBaskets import StateMapFolding
from mapFolding.synthesized.initializeState import transitionOnGroupsOfFolds

def count(state: StateMapFolding) -> StateMapFolding:
    while state.leaf1ndex > 4:
        if state.leafBelow[0] == 1:
            if state.leaf1ndex > state.totalLeaves:
                state.groupsOfFolds += 1
            else:
                state.dimensionsUnconstrained = state.totalDimensions
                state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
                state.次Dimension = 0
                while state.次Dimension < state.totalDimensions:
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
        state.groupsOfFolds *= 2
    return state

def doTheNeedful(state: StateMapFolding) -> StateMapFolding:
    state = transitionOnGroupsOfFolds(state)
    state = count(state)
    return state
