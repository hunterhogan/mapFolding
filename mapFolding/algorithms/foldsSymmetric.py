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

from mapFolding.dataBaskets import SymmetricFoldsState

def filterAsymmetricFolds(state: SymmetricFoldsState) -> SymmetricFoldsState:
	state.次Leaf = 1
	state.leafComparison[0] = 1
	state.leafConnectee = 1

	while state.leafConnectee < state.totalLeaves + 1:
		state.次MiniGap = state.leafBelow[state.次Leaf]
		state.leafComparison[state.leafConnectee] = (state.totalLeaves + state.次MiniGap - state.次Leaf) % state.totalLeaves
		state.次Leaf = state.次MiniGap

		state.leafConnectee += 1

	for boxOfTuples in state.indices:
		state.leafConnectee = 1
		for 次Left, 次Right in boxOfTuples:
			# TODO The entire `leafComparison` array is computed, so when a `leafComparison` is
			# disqualified, all of the computations for the remaining tuples were unnecessary
			# computations. However, with the current algorithm for computing `leafComparison`, it
			# would be nearly impossible to efficiently validate tuples during the computation.
			if state.leafComparison[次Left] != state.leafComparison[次Right]:
				state.leafConnectee = 0
				break
		state.symmetricFolds += state.leafConnectee

	return state
