"""Count the number of symmetric folds in the group of folds defined by `leafBelow`.

Notes
-----
- About constructing `leafComparison`:
    - The first iteration of the loop is hardcoded to save processing time.
    - I _feel_ there must be a more efficient way to do this.
- Some implementation details are based on Numba compatibility. Incompatible:
    - `numpy.take(..., out=...)`
    - `numpy.all(..., axis=...)`
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mapFolding.dataBaskets import StateMapFoldingSymmetric

# IMPROVEMENT this algorithm is slow.
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
            # IMPROVEMENT The entire `leafComparison` array is computed, so when a `leafComparison` is
            # disqualified, all of the computations for the remaining tuples were unnecessary
            # computations. However, with the current algorithm for computing `leafComparison`, it
            # would be nearly impossible to efficiently validate tuples during the computation.
            if state.leafComparison[次Left] != state.leafComparison[次Right]:
                state.leafConnectee = 0
                break
        state.symmetricFolds += state.leafConnectee

    return state
