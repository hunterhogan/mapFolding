"""Count the number of symmetric folds in the group of folds defined by `leafBelow`.

Notes
-----
- About constructing `leafComparison`:
    - This branch of the algorithm executes IFF `leafBelow[0] == 1`.
    - Therefore, `leafComparison[0]` must be `1`.
    - Therefore, the first iteration of the loop is hardcoded to save processing time.
    - I _feel_ there must be a more efficient way to do this.
- Some implementation details are based on Numba compatibility. Incompatible:
    - `numpy.take(..., out=...)`
    - `numpy.all(..., axis=...)`
"""
from mapFolding.dataBaskets import SymmetricFoldsState
import numpy

def filterAsymmetricFolds(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state.indexLeaf = 1
    state.leafComparison[0] = 1
    state.leafConnectee = 1

    while state.leafConnectee < state.leavesTotal + 1:
        state.indexMiniGap = state.leafBelow[state.indexLeaf]
        state.leafComparison[state.leafConnectee] = (state.indexMiniGap - state.indexLeaf + state.leavesTotal) % state.leavesTotal
        state.indexLeaf = state.indexMiniGap

        state.leafConnectee += 1

    state.arrayGroupOfFolds = numpy.take(state.leafComparison, state.indicesArrayGroupOfFolds)
    compared = state.arrayGroupOfFolds[..., 0:state.leavesTotal // 2] == state.arrayGroupOfFolds[..., state.leavesTotal // 2:None]

    for indexRow in range(len(compared)):
        state.groupsOfFolds += compared[indexRow].all()

    return state

