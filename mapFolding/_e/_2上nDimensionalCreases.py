from collections.abc import Iterator
from functools import cache
from gmpy2 import bit_flip, is_even, is_odd
from mapFolding._e import dimensionNearestTail, dimensionNearest首, howManyDimensionsHaveOddParity, Leaf, leafOrigin
from mapFolding._e.dataBaskets import EliminationState

def getLeavesCreaseAnte(state: EliminationState, leaf: Leaf) -> Iterator[Leaf]:
	"""1) `leaf` has at most `dimensionsTotal - 1` many creases.

	2) The list is ordered by increasing dimension number, which corresponds to an increasing absolute magnitude of _change_ in `leaf` number.

	3) The list of creases *might* be a list of Gray codes.
	"""
	return iter(_getCreases(state, leaf, increase=False))

def getLeavesCreasePost(state: EliminationState, leaf: Leaf) -> Iterator[Leaf]:
	"""1) `leaf` has at most `dimensionsTotal - 1` many creases.

	2) The list is ordered by increasing dimension number, which corresponds to an increasing absolute magnitude of _change_ in `leaf` number.

	3) The list of creases *might* be a list of Gray codes.
	"""
	return iter(_getCreases(state, leaf, increase=True))

def _getCreases(state: EliminationState, leaf: Leaf, *, increase: bool = True) -> tuple[Leaf, ...]:
	return _makeCreases(leaf, state.dimensionsTotal)[increase]
@cache
def _makeCreases(leaf: Leaf, dimensionsTotal: int) -> tuple[tuple[Leaf, ...], tuple[Leaf, ...]]:
	listLeavesCrease: list[Leaf] = [int(bit_flip(leaf, dimension)) for dimension in range(dimensionsTotal)]

	if leaf == leafOrigin: # A special case I've been unable to figure out how to incorporate in the formula.
		listLeavesCreasePost: list[Leaf] = [1]
		listLeavesCreaseAnte: list[Leaf] = []
	else:
		slicingIndices: int = is_odd(howManyDimensionsHaveOddParity(leaf))

		slicerAnte: slice = slice(slicingIndices, dimensionNearest首(leaf) * bit_flip(slicingIndices, 0) or None)
		slicerPost: slice = slice(bit_flip(slicingIndices, 0), dimensionNearest首(leaf) * slicingIndices or None)

		if is_even(leaf):
			if slicerAnte.start == 1:
				slicerAnte = slice(slicerAnte.start + dimensionNearestTail(leaf), slicerAnte.stop)
			if slicerPost.start == 1:
				slicerPost = slice(slicerPost.start + dimensionNearestTail(leaf), slicerPost.stop)
		listLeavesCreaseAnte: list[Leaf] = listLeavesCrease[slicerAnte]
		listLeavesCreasePost: list[Leaf] = listLeavesCrease[slicerPost]

		if leaf == 1: # A special case I've been unable to figure out how to incorporate in the formula.
			listLeavesCreaseAnte = [0]
	return (tuple(listLeavesCreaseAnte), tuple(listLeavesCreasePost))

