"""Count symmetric foldings by testing each group of foldings for symmetry.

(AI generated docstring)

`mapFolding.kitAST.mapFoldingSymmetric.makeModules.addSymmetryCheck` generates this module.
Regenerate the module instead of editing it.

Contents
--------
Functions
    activeLeafGreaterThan0
        Test whether the search still has a leaf to insert or remove.
    activeLeafGreaterThanTotalLeaves
        Test whether every leaf is in the folding.
    activeLeafIsTheFirstLeaf
        Test whether the active leaf is the first leaf.
    activeLeafIsUnconstrainedInAllDimensions
        Test whether no dimension constrains the gaps of the active leaf.
    activeLeafUnconstrainedInThisDimension
        Record that the current dimension does not constrain the active leaf.
    count
        Count the groups of foldings by inserting and removing leaves.
    doTheNeedful
        Count the groups of foldings of the map in `state`.
    filterAsymmetricFolds
        Add the symmetric foldings in the group of `state.leafBelow` to `state.symmetricFolds`.
    filterCommonGaps
        Keep a candidate gap when every constraining dimension admits the gap.
    gapAvailable
        Test whether the active leaf has a gap for insertion.
    incrementActiveGap
        Advance the cursor of kept gaps.
    incrementGap1ndexCeiling
        Advance the upper bound of candidate gaps.
    incrementIndexMiniGap
        Advance the cursor within the candidate gaps.
    initializeIndexMiniGap
        Start the cursor within the candidate gaps at the cursor of kept gaps.
    initializeVariablesToFindGaps
        Prepare the counters and cursors for finding the gaps of the active leaf.
    insertActiveLeaf
        Record every position below the active leaf as a candidate gap.
    insertActiveLeafAtGap
        Insert the active leaf at the last kept gap and select the next leaf.
    leafBelowSentinelIs1
        Test whether leaf 1 is directly below the sentinel.
    leafConnecteeIsActiveLeaf
        Test whether the connection graph connects the active leaf to itself.
    lookForGaps
        Record the selected connectee as a candidate gap in the current dimension.
    lookupLeafConnecteeInConnectionGraph
        Select the first connectee of the active leaf in the current dimension.
    loopingLeavesConnectedToActiveLeaf
        Test whether the selected connectee differs from the active leaf.
    loopingThroughTheDimensions
        Test whether the dimension cursor has dimensions remaining.
    loopingToActiveGapCeiling
        Test whether the cursor within the candidate gaps has candidates remaining.
    noGapsHere
        Test whether the active leaf has no kept gaps remaining.
    tryAnotherLeafConnectee
        Select the next connectee of the active leaf in the current dimension.
    tryNextDimension
        Advance the dimension cursor.
    undoLastLeafPlacement
        Remove the most recently inserted leaf from the folding.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mapFolding.dataBaskets import StateMapFoldingSymmetric

def filterAsymmetricFolds(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Add the symmetric foldings in the group of `state.leafBelow` to `state.symmetricFolds`.

    (AI generated docstring)

    `state.leafComparison` receives, for each position in the stack, the difference modulo
    `state.totalLeaves` between successive leaves. Each array of index pairs in `state.indices`
    describes one candidate symmetry, and a folding of the group has that symmetry when every pair of
    `state.leafComparison` entries is equal.

    Parameters
    ----------
    state : StateMapFoldingSymmetric
        Insertion search with a complete folding in `state.leafBelow`.

    Returns
    -------
    state : StateMapFoldingSymmetric
        The same state with `state.symmetricFolds` increased by the number of candidate symmetries
        that hold.
    """
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
        state.symmetricFolds += state.leafConnectee
    return state

def activeLeafGreaterThan0(state: StateMapFoldingSymmetric) -> bool:
    """Test whether the search still has a leaf to insert or remove.

    (AI generated docstring)

    The search ends when removing leaves moves `leaf1ndex` below the first leaf.

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    activeLeafGreaterThan0 : bool
        `True` when `leaf1ndex` is greater than 0.
    """
    return state.leaf1ndex > 0

def activeLeafGreaterThanTotalLeaves(state: StateMapFoldingSymmetric) -> bool:
    """Test whether every leaf is in the folding.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    activeLeafGreaterThanTotalLeaves : bool
        `True` when `leaf1ndex` is greater than `totalLeaves`.
    """
    return state.leaf1ndex > state.totalLeaves

def activeLeafIsTheFirstLeaf(state: StateMapFoldingSymmetric) -> bool:
    """Test whether the active leaf is the first leaf.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    activeLeafIsTheFirstLeaf : bool
        `True` when `leaf1ndex` is at most 1.
    """
    return state.leaf1ndex <= 1

def activeLeafIsUnconstrainedInAllDimensions(state: StateMapFoldingSymmetric) -> bool:
    """Test whether no dimension constrains the gaps of the active leaf.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    activeLeafIsUnconstrainedInAllDimensions : bool
        `True` when `dimensionsUnconstrained` is 0.
    """
    return not state.dimensionsUnconstrained

def activeLeafUnconstrainedInThisDimension(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Record that the current dimension does not constrain the active leaf.

    (AI generated docstring)

    The connection graph connects the active leaf to itself in dimension `次Dimension`, so this
    dimension cannot exclude any gap. `dimensionsUnconstrained` decreases by 1.

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with `dimensionsUnconstrained` decreased by 1.
    """
    state.dimensionsUnconstrained -= 1
    return state

def filterCommonGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Keep a candidate gap when every constraining dimension admits the gap.

    (AI generated docstring)

    The candidate gap at `次MiniGap` moves to position `gap1ndex` of `gapsWhere`. When the number of
    dimensions that admit the candidate gap equals `dimensionsUnconstrained`, `gap1ndex` advances, so
    the gap stays in the kept gaps. The count of dimensions that admit the candidate gap returns to 0.

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with the candidate gap kept or discarded.
    """
    state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.次MiniGap]
    if state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] == state.dimensionsUnconstrained:
        state = incrementActiveGap(state)
    state.countDimensionsGapped[state.gapsWhere[state.次MiniGap]] = 0
    return state

def gapAvailable(state: StateMapFoldingSymmetric) -> bool:
    """Test whether the active leaf has a gap for insertion.

    (AI generated docstring)

    After the search removes every leaf without a kept gap, a positive `leaf1ndex` means the active
    leaf has a kept gap.

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    gapAvailable : bool
        `True` when `leaf1ndex` is greater than 0.
    """
    return state.leaf1ndex > 0

def incrementActiveGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Advance the cursor of kept gaps.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with `gap1ndex` increased by 1.
    """
    state.gap1ndex += 1
    return state

def incrementGap1ndexCeiling(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Advance the upper bound of candidate gaps.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with `gap1ndexCeiling` increased by 1.
    """
    state.gap1ndexCeiling += 1
    return state

def incrementIndexMiniGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Advance the cursor within the candidate gaps.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with `次MiniGap` increased by 1.
    """
    state.次MiniGap += 1
    return state

def initializeIndexMiniGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Start the cursor within the candidate gaps at the cursor of kept gaps.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with `次MiniGap` equal to `gap1ndex`.
    """
    state.次MiniGap = state.gap1ndex
    return state

def initializeVariablesToFindGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Prepare the counters and cursors for finding the gaps of the active leaf.

    (AI generated docstring)

    Every dimension starts as a constraining dimension. The candidate gaps of the active leaf start
    after the gap range of the previous leaf, and the dimension cursor starts at the first dimension.

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with `dimensionsUnconstrained`, `gap1ndexCeiling`, and `次Dimension`
        prepared.
    """
    state.dimensionsUnconstrained = state.totalDimensions
    state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
    state.次Dimension = 0
    return state

def insertActiveLeaf(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Record every position below the active leaf as a candidate gap.

    (AI generated docstring)

    No dimension constrains the active leaf, so each leaf label from 0 to `leaf1ndex - 1` becomes a
    candidate gap.

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with the candidate gaps appended to `gapsWhere`.
    """
    state.次Leaf = 0
    while state.次Leaf < state.leaf1ndex:
        state.gapsWhere[state.gap1ndexCeiling] = state.次Leaf
        state.gap1ndexCeiling += 1
        state.次Leaf += 1
    return state

def insertActiveLeafAtGap(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Insert the active leaf at the last kept gap and select the next leaf.

    (AI generated docstring)

    The active leaf goes directly below the leaf at the gap in `leafAbove` and `leafBelow`.
    `gapRangeStart` records where the kept gaps of the active leaf end, and `leaf1ndex` advances.

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with the active leaf in the folding.
    """
    state.gap1ndex -= 1
    state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
    state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
    state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
    state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
    state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
    state.leaf1ndex += 1
    return state

def leafBelowSentinelIs1(state: StateMapFoldingSymmetric) -> bool:
    """Test whether leaf 1 is directly below the sentinel.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    leafBelowSentinelIs1 : bool
        `True` when `leafBelow[0]` is 1.
    """
    return state.leafBelow[0] == 1

def leafConnecteeIsActiveLeaf(state: StateMapFoldingSymmetric) -> bool:
    """Test whether the connection graph connects the active leaf to itself.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    leafConnecteeIsActiveLeaf : bool
        `True` when `leafConnectee` is `leaf1ndex`.
    """
    return state.leafConnectee == state.leaf1ndex

def lookForGaps(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Record the selected connectee as a candidate gap in the current dimension.

    (AI generated docstring)

    The first dimension that admits a gap appends the gap to the candidate gaps. Each dimension that
    admits the gap increases the count of dimensions in `countDimensionsGapped`.

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with the gap recorded.
    """
    state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
    if state.countDimensionsGapped[state.leafConnectee] == 0:
        state = incrementGap1ndexCeiling(state)
    state.countDimensionsGapped[state.leafConnectee] += 1
    return state

def lookupLeafConnecteeInConnectionGraph(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Select the first connectee of the active leaf in the current dimension.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with `leafConnectee` from `connectionGraph`.
    """
    state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leaf1ndex]
    return state

def loopingLeavesConnectedToActiveLeaf(state: StateMapFoldingSymmetric) -> bool:
    """Test whether the selected connectee differs from the active leaf.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    loopingLeavesConnectedToActiveLeaf : bool
        `True` when `leafConnectee` is not `leaf1ndex`.
    """
    return state.leafConnectee != state.leaf1ndex

def loopingThroughTheDimensions(state: StateMapFoldingSymmetric) -> bool:
    """Test whether the dimension cursor has dimensions remaining.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    loopingThroughTheDimensions : bool
        `True` when `次Dimension` is less than `totalDimensions`.
    """
    return state.次Dimension < state.totalDimensions

def loopingToActiveGapCeiling(state: StateMapFoldingSymmetric) -> bool:
    """Test whether the cursor within the candidate gaps has candidates remaining.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    loopingToActiveGapCeiling : bool
        `True` when `次MiniGap` is less than `gap1ndexCeiling`.
    """
    return state.次MiniGap < state.gap1ndexCeiling

def noGapsHere(state: StateMapFoldingSymmetric) -> bool:
    """Test whether the active leaf has no kept gaps remaining.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    noGapsHere : bool
        `True` when `leaf1ndex` is greater than 0 and `gap1ndex` is at the start of the gap range of the
        previous leaf.
    """
    return state.leaf1ndex > 0 and state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]

def tryAnotherLeafConnectee(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Select the next connectee of the active leaf in the current dimension.

    (AI generated docstring)

    The next connectee is the connectee of the leaf below the selected connectee.

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with `leafConnectee` from `connectionGraph`.
    """
    state.leafConnectee = state.connectionGraph[state.次Dimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
    return state

def tryNextDimension(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Advance the dimension cursor.

    (AI generated docstring)

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search with `次Dimension` increased by 1.
    """
    state.次Dimension += 1
    return state

def undoLastLeafPlacement(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Remove the most recently inserted leaf from the folding.

    (AI generated docstring)

    `leaf1ndex` returns to the previous leaf, and that leaf leaves `leafAbove` and `leafBelow`.

    Parameters
    ----------
    state : StateMapFolding
        The insertion search.

    Returns
    -------
    state : StateMapFolding
        The same insertion search without the removed leaf.
    """
    state.leaf1ndex -= 1
    state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
    state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
    return state

def count(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
    """Count the groups of foldings by inserting and removing leaves.

    (AI generated docstring)

    You can count the foldings of the map that `connectionGraph` describes. Each leaf goes into each
    gap that every constraining dimension admits. When every leaf is in the folding,
    `groupsOfFolds` increases by 1. Each group contains `totalLeaves` foldings. The generators in
    `mapFolding.kitAST` replace the `StateMapFolding` parameter of this function with its fields, so
    this description names the fields directly.
    """
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
    state = count(state)
    return state
