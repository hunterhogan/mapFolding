from collections.abc import Iterator
from functools import cache
from itertools import pairwise, permutations, repeat
from mapFolding._e import PinnedLeaves
from mapFolding._e.pinIt import excludeLeafRBeforeLeafK, makeFolding
from mapFolding.algorithms.iff import productOfDimensions, thisLeafFoldingIsValid
from mapFolding.dataBaskets import EliminationState
from math import factorial
from more_itertools import iter_index, unique

def count(state: EliminationState) -> EliminationState:
	state.groupsOfFolds += sum(map(countPinnedLeaves, state.listPinnedLeaves, repeat(state.mapShape), repeat(state.leavesTotal)))
	return state

def countPinnedLeaves(pinnedLeaves: PinnedLeaves, mapShape: tuple[int, ...], leavesTotal: int) -> int:
	return sum(map(thisLeafFoldingIsValid, map(makeFolding, repeat(pinnedLeaves), permutePermutands(pinnedLeaves, leavesTotal)), repeat(mapShape)))

@cache
def setOfLeaves(leavesTotal: int) -> set[int]:
	return set(range(leavesTotal))

def permutands(pinnedLeaves: PinnedLeaves, leavesTotal: int) -> tuple[int, ...]:
	return tuple(setOfLeaves(leavesTotal).difference(pinnedLeaves.values()))

def permutePermutands(pinnedLeaves: PinnedLeaves, leavesTotal: int) -> Iterator[tuple[int, ...]]:
	return permutations(permutands(pinnedLeaves, leavesTotal))

def theorem2b(state: EliminationState) -> EliminationState:
	"""Implement Lunnon Theorem 2(b): If some dimension pᵢ > 2 (with Theorem 4 inactive), G is divisible by 2·n.

	Conditions
	----------
	Executed only when:
	- `state.Theorem4Multiplier == 1` (Theorem 4 did not apply),
	- `max(mapShape) > 2` (at least one dimension exceeds 2),
	- `leavesTotal > 4` (non-trivial folding size).

	Mechanism
	---------
	Select the maximal dimension, compute `k = productOfDimensions(mapShape, dimension)` and `r = 2 * k`, then eliminate
	configurations where r precedes k (via `excludeLeafRBeforeLeafK`). This enforces the structural constraint underpinning
	the `2·n` divisibility and sets `state.Theorem2Multiplier = 2`.

	Side Effects
	------------
	Mutates `state.listPinnedLeaves` and `state.Theorem2Multiplier` when applicable.

	Returns
	-------
	EliminationState
		Same state instance after potential exclusion.

	See Also
	--------
	theorem4, excludeLeafRBeforeLeafK
	"""
	if state.Theorem4Multiplier == 1 and max(state.mapShape) > 2 and (state.leavesTotal > 4):
		state.Theorem2Multiplier = 2
		dimension: int = state.mapShape.index(max(state.mapShape))
		k: int = productOfDimensions(state.mapShape, dimension)
		r: int = 2 * k
		state = excludeLeafRBeforeLeafK(state, k, r)

	return state

def theorem4(state: EliminationState) -> EliminationState:
	"""Implement Lunnon Theorem 4 (divisibility by d! · p^d) via systematic leaf exclusions.

	Statement
	---------
	For a map whose shape has repeated dimension magnitudes (say a size `p` occurring `d` times), the total number of foldings
	`G(p^d)` is divisible by `d! · p^d`. This routine encodes the constructive elimination implied by that divisibility.

	Method
	------
	1. Group dimensions sharing the same magnitude (`listIndicesSameMagnitude`).
	2. When a group size > 1, set `state.Theorem4Multiplier = factorial(group_size)` (the `d!` component).
	3. For each adjacent pair of dimensions (alpha, beta) in the group, compute leaf indices
	`k, r` as `productOfDimensions(mapShape, dimensionAlpha/Beta)` and exclude configurations where r precedes k using
	`excludeLeafRBeforeLeafK`.

	Side Effects
	------------
	Mutates `state.listPinnedLeaves` and potentially `state.Theorem4Multiplier`.

	Returns
	-------
	EliminationState
		Same state instance after applying all required exclusions.

	See Also
	--------
	theorem2b : Applies the complementary 2(b) divisibility when 4 does not trigger.
	excludeLeafRBeforeLeafK : Performs the actual leaf ordering elimination.
	"""
	for listIndicesSameMagnitude in [list(iter_index(state.mapShape, magnitude)) for magnitude in unique(state.mapShape)]:
		if len(listIndicesSameMagnitude) > 1:
			state.Theorem4Multiplier = factorial(len(listIndicesSameMagnitude))
			for dimensionAlpha, dimensionBeta in pairwise(listIndicesSameMagnitude):
				k, r = (productOfDimensions(state.mapShape, dimension) for dimension in (dimensionAlpha, dimensionBeta))
				state = excludeLeafRBeforeLeafK(state, k, r)
	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:  # noqa: ARG001
	"""Count the number of valid foldings for a given number of leaves."""
	from mapFolding._e.pinning2Dn import pinByFormula  # noqa: PLC0415
# NOTE Lunnon Theorem 2(a): `foldsTotal` is divisible by `leavesTotal`; pin leaf0 in pile0, which eliminates leaf1 through leafLast at pile0
	state.listPinnedLeaves = [{0: 0}]

	state = theorem4(state)
	state = theorem2b(state)
	state = pinByFormula(state)

	return count(state)

