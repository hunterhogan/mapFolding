from __future__ import annotations

from functools import cache
from gmpy2 import bit_flip, is_even as isEven吗, is_odd as isOdd吗
from mapFolding._e import leafOrigin
from mapFolding._e._2上nDimensional import 工dimensionTail, 工dimension首零, 工totalDimensionsOdd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterator
	from mapFolding._e.dataBaskets import StateElimination
	from mapFolding._e.theTypes import Leaf

def getLeavesCreaseAnte(state: StateElimination, leaf: Leaf) -> Iterator[Leaf]:
	"""1) `leaf` has at most `totalDimensions - 1` many creases.

	2) The list is ordered by increasing dimension number, which corresponds to an increasing absolute magnitude of _change_ in `leaf` number.

	3) The list of creases *might* be a list of Gray codes.
	"""
	return iter(_getCreases(state, leaf, increase=False))

def getLeavesCreasePost(state: StateElimination, leaf: Leaf) -> Iterator[Leaf]:
	"""1) `leaf` has at most `totalDimensions - 1` many creases.

	2) The list is ordered by increasing dimension number, which corresponds to an increasing absolute magnitude of _change_ in `leaf` number.

	3) The list of creases *might* be a list of Gray codes.
	"""
	return iter(_getCreases(state, leaf, increase=True))

def _getCreases(state: StateElimination, leaf: Leaf, *, increase: bool = True) -> tuple[Leaf, ...]:
	return _makeCreases(leaf, state.totalDimensions)[increase]
@cache
def _makeCreases(leaf: Leaf, totalDimensions: int) -> tuple[tuple[Leaf, ...], tuple[Leaf, ...]]:
	boxOfLeavesCrease: list[Leaf] = [int(bit_flip(leaf, dimension)) for dimension in range(totalDimensions)]

	if leaf == leafOrigin:  # A special case I've been unable to figure out how to incorporate in the formula.
		boxOfLeavesCreasePost: list[Leaf] = [1]
		boxOfLeavesCreaseAnte: list[Leaf] = []
	else:
		slicingIndices: int = isOdd吗(工totalDimensionsOdd(leaf))

		slicerAnte: slice = slice(slicingIndices, 工dimension首零(leaf) * bit_flip(slicingIndices, 0) or None)
		slicerPost: slice = slice(bit_flip(slicingIndices, 0), 工dimension首零(leaf) * slicingIndices or None)

		if isEven吗(leaf):
			if slicerAnte.start == 1:
				slicerAnte = slice(slicerAnte.start + 工dimensionTail(leaf), slicerAnte.stop)
			if slicerPost.start == 1:
				slicerPost = slice(slicerPost.start + 工dimensionTail(leaf), slicerPost.stop)
		boxOfLeavesCreaseAnte: list[Leaf] = boxOfLeavesCrease[slicerAnte]
		boxOfLeavesCreasePost: list[Leaf] = boxOfLeavesCrease[slicerPost]

		if leaf == 1:  # A special case I've been unable to figure out how to incorporate in the formula.
			boxOfLeavesCreaseAnte = [0]
	return (tuple(boxOfLeavesCreaseAnte), tuple(boxOfLeavesCreasePost))
