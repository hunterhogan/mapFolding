"""Enumerate directional crease neighbors for a leaf.

(AI generated docstring)

You can inspect crease-connected leaves compatible with an earlier or later position in the pile
order. You obtain these leaves from the map dimensions and the integer label of the queried leaf.

Contents
--------
Functions
	getLeavesCreaseAnte
		Iterate over crease-connected leaves that can appear before `leaf`.
	getLeavesCreasePost
		Iterate over crease-connected leaves that can appear after `leaf`.
"""
from __future__ import annotations

from functools import cache
from gmpy2 import bit_flip, is_even as isEven吗, is_odd as isOdd吗
from mapFolding import leafOrigin
from mapFolding.measure import 工dimensionTail, 工dimension首零, 工totalDimensionsOdd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterator
	from mapFolding.dataBaskets import StateElimination
	from mapFolding.theTypes import Leaf

def getLeavesCreaseAnte(state: StateElimination, leaf: Leaf) -> Iterator[Leaf]:
	"""Iterate over crease-connected leaves that can appear before `leaf`.

	(AI generated docstring)

	You can retrieve an iterator over neighboring leaves compatible with placing a crease-connected
	neighbor before `leaf` in the pile order. You obtain these leaves from the map dimensions without
	inspecting pinned piles.

	Parameters
	----------
	state : StateElimination
		Elimination state that provides the map shape and dimension count.
	leaf : Leaf
		Leaf index whose crease neighbors are requested.

	Returns
	-------
	leavesCrease : Iterator[Leaf]
		Iterator over leaf indices in increasing order of the flipped dimension
		index. An origin leaf has no predecessor choice.

	See Also
	--------
	`mapFolding._semiotics`
		Inspect the coordinate symbols and their dimension-index convention.

	Dimension Order
	---------------
	The binary positional-numeral system assigns one
	dimension to each bit. Possible leaves differ from `leaf` by one bit. Leaf labels
	need not increase with the flipped dimension index.

	Examples
	--------
	From `easyRun/pinning.py`, with `state` for a map whose axis lengths are `2`.

		```python
		print(*getLeavesCreaseAnte(state, 53))
		```

	"""
	return iter(_getCreases(state, leaf, increase=False))

def getLeavesCreasePost(state: StateElimination, leaf: Leaf) -> Iterator[Leaf]:
	"""Iterate over crease-connected leaves that can appear after `leaf`.

	(AI generated docstring)

	You can retrieve an iterator over neighboring leaves compatible with placing a crease-connected
	neighbor after `leaf` in the pile order. You obtain these leaves from the map dimensions without
	inspecting pinned piles.

	Parameters
	----------
	state : StateElimination
		Elimination state that provides the map shape and dimension count.
	leaf : Leaf
		Leaf index whose crease neighbors are requested.

	Returns
	-------
	leavesCrease : Iterator[Leaf]
		Iterator over leaf indices in increasing order of the flipped dimension
		index. An origin leaf has only leaf `1` as a successor choice.

	See Also
	--------
	`mapFolding._semiotics`
		Inspect the coordinate symbols and their dimension-index convention.

	Dimension Order
	---------------
	The binary positional-numeral system assigns one
	dimension to each bit. Possible leaves differ from `leaf` by one bit. Leaf labels
	need not increase with the flipped dimension index.

	Examples
	--------
	From `easyRun/pinning.py`, with `state` for a map whose axis lengths are `2`.

		```python
		print(*getLeavesCreasePost(state, 22))
		```

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
