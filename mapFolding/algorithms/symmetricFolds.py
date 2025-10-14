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

