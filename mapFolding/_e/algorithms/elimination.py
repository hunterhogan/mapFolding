from collections.abc import Iterable
from itertools import pairwise, permutations, repeat
from mapFolding._e import indicesMapShapeDimensionLengthsAreEqual, leafOrigin, PermutationSpace, pileOrigin
from mapFolding._e.algorithms.iff import thisLeafFoldingIsValid
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pinIt import excludeLeaf_rBeforeLeaf_k, makeFolding
from math import factorial

# TODO make sure all permutationSpace have pile-ranges and update their pile-ranges

def count(state: EliminationState) -> EliminationState:
	state.groupsOfFolds += sum(map(countPermutationSpace, state.listPermutationSpace, repeat(state.mapShape), repeat(range(state.leavesTotal))))
	return state

def countPermutationSpace(permutationSpace: PermutationSpace, mapShape: tuple[int, ...], leavesToInsert: Iterable[int]) -> int:
	"""# TODO Replace `permutations` with the `noDuplicates` filter on `CartesianProduct` of the domains of each leaf.

	permutationSpace must be in order by `pile`.
	filter with `oop`.
	pileRangeOfLeaves.iter_set() returns an iterator of int corresponding to the leaves in the pile's range.
	https://gmpy2.readthedocs.io/en/latest/advmpz.html#gmpy2.xmpz.iter_set
	filter out "leaf" = state.leavesTotal
	CartesianProduct over these iterators.
	filter via noDuplicates.
	each product tuple is a `leavesToInsert` argument to makeFolding.
	return sum(
		map(thisLeafFoldingIsValid
			, map(makeFolding
				, repeat(permutationSpace)
				, CartesianProduct(*map(getIteratorOfLeaves, extractPilesWithPileRangeOfLeaves(permutationSpace).values()))
			)
			, repeat(mapShape)
		)
	)
	"""
	return sum(map(thisLeafFoldingIsValid, map(makeFolding, repeat(permutationSpace), permutations(tuple(set(leavesToInsert).difference(permutationSpace.values())))), repeat(mapShape)))


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
	Mutates `state.listPermutationSpace` and `state.Theorem2Multiplier` when applicable.

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
		k: int = state.productsOfDimensions[dimension]
		r: int = 2 * k
		state = excludeLeaf_rBeforeLeaf_k(state, k, r)
	return state

def theorem4(state: EliminationState) -> EliminationState:
	"""***Bluntly*** implement Lunnon Theorem 4 (divisibility by d! · p^d) via systematic leaf exclusions.

	This function is ignorant of the actual domains of the dimension-origin leaves, so it creates `PermutationSpace` dictionaries to
	exclude at every `pile`. The permutation space is still valid. However, for each `PermutationSpace` dictionary, for each `pile`
	*not* in the domain of a dimension-origin leaf, the function creates approximately `leavesTotal - 1` unnecessary
	`PermutationSpace` dictionaries. They are "unnecessary" because we didn't need to exclude the leaf from a pile in which it could
	never appear. The net result is that the number of unnecessary dictionaries grows exponentially with the number of repeated
	dimension magnitudes.

	For a map whose shape has repeated dimension magnitudes (say a size `p` occurring `d` times), the total number of foldings
	`G(p^d)` is divisible by `d! · p^d`. This routine encodes the constructive elimination implied by that divisibility.

	Returns
	-------
	EliminationState
		Same state instance after applying all required exclusions.

	See Also
	--------
	theorem2b : Applies the complementary 2(b) divisibility if theorem4 does not apply.
	excludeLeafRBeforeLeafK : Performs the actual leaf ordering elimination.
	"""
	for indicesSameDimensionLength in indicesMapShapeDimensionLengthsAreEqual(state.mapShape):
		state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
		for index_k, index_r in pairwise(indicesSameDimensionLength):
			state = excludeLeaf_rBeforeLeaf_k(state, state.productsOfDimensions[index_k], state.productsOfDimensions[index_r])
	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:  # noqa: ARG001
	"""Count the number of valid foldings for a given number of leaves."""
	if not state.listPermutationSpace:
		"""Lunnon Theorem 2(a): `foldsTotal` is divisible by `leavesTotal`; pin `leafOrigin` at `pileOrigin`, which eliminates other leaves at `pileOrigin`."""
		state.listPermutationSpace = [{pileOrigin: leafOrigin}]
		state = theorem4(state)
		state = theorem2b(state)

	return count(state)

