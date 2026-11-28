"""Count with the counting function specialized by Theorem 2.

(AI generated docstring)

`mapFolding.kitAST.mapFolding._count.makeTheorem2` generates this module.
Regenerate the module instead of editing it.

Contents
--------
Functions
    count
        Count the groups of foldings by inserting and removing leaves.
    doTheNeedful
        Count the groups of foldings of the map in `state`.
"""
from __future__ import annotations

from mapFolding.synthesized.initializeState import transitionOnGroupsOfFolds
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mapFolding.dataBaskets import StateMapFolding

def count(state: StateMapFolding) -> StateMapFolding:
    """Count the groups of foldings by inserting and removing leaves.

    (AI generated docstring)

    You can count the foldings of the map that `connectionGraph` describes. Each leaf goes into each
    gap that every constraining dimension admits. When every leaf is in the folding,
    `groupsOfFolds` increases by 1. Each group contains `totalLeaves` foldings. The generators in
    `mapFolding.kitAST` replace the `StateMapFolding` parameter of this function with its fields, so
    this description names the fields directly.
    """
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
    """Count the groups of foldings of the map in `state`.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        An initialized insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with `groupsOfFolds` complete.
    """
    state = transitionOnGroupsOfFolds(state)
    state = count(state)
    return state
