"""Advance a new insertion search to its first counted group of foldings.

(AI generated docstring)

The Theorem 2 counting functions start from the state that this module returns.

`mapFolding.kitAST.mapFolding._doTheNeedful.makeInitializeState` generates this module.
Regenerate the module instead of editing it.

Contents
--------
Functions
    transitionOnGroupsOfFolds
        Count the groups of foldings by inserting and removing leaves.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mapFolding.dataBaskets import StateMapFolding

def transitionOnGroupsOfFolds(state: StateMapFolding) -> StateMapFolding:
    """Count the groups of foldings by inserting and removing leaves.

    (AI generated docstring)

    You can count the foldings of the map that `connectionGraph` describes. Each leaf goes into each
    gap that every constraining dimension admits. When every leaf is in the folding,
    `groupsOfFolds` increases by 1. Each group contains `totalLeaves` foldings. The generators in
    `mapFolding.kitAST` replace the `StateMapFolding` parameter of this function with its fields, so
    this description names the fields directly.
    """
    while state.groupsOfFolds == 0:
        if state.leaf1ndex <= 1 or state.leafBelow[0] == 1:
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
                if not state.dimensionsUnconstrained:
                    state.次Leaf = 0
                    while state.次Leaf < state.leaf1ndex:
                        state.gapsWhere[state.gap1ndexCeiling] = state.次Leaf
                        state.gap1ndexCeiling += 1
                        state.次Leaf += 1
                state.次MiniGap = state.gap1ndex
                while state.次MiniGap < state.gap1ndexCeiling:
                    state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.次MiniGap]
                    if state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] == state.dimensionsUnconstrained:
                        state.gap1ndex += 1
                    state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] = 0
                    state.次MiniGap += 1
        while state.leaf1ndex > 0 and state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]:
            state.leaf1ndex -= 1
            state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
            state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
        if state.leaf1ndex > 0:
            state.gap1ndex -= 1
            state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
            state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
            state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
            state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
            state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
            state.leaf1ndex += 1
    return state
