from __future__ import annotations

from gmpy2 import bit_test as isBit1吗
from mapFolding._e._2上nDimensional import 零
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.theTypes import DimensionIndex, Leaf, LeafSpace

def moreThanLeaf零吗(leaf: LeafSpace) -> bool:
	"""Test to ensure `leaf` is greater than `leafOrigin` (0) and `leaf零` (1).

	You can use `moreThanLeaf零吗` in an `if` statement, or you can pass `moreThanLeaf零吗` as a
	predicate to a filtering utility described in the module docstring.

	Parameters
	----------
	leaf : Leaf
		`leaf` index.

	Returns
	-------
	leafIsNotOriginOrZero : bool
		`True` if `零 < leaf`.

	References
	----------
	[1] mapFolding._e.零
	"""
	return 零 < leaf

def oddLeaf2上nDimensional吗(dimension: DimensionIndex, leaf: Leaf) -> bool:
	"""Test whether the bit at `dimension` position is set in `leaf`.

	You can use `oddLeaf2上nDimensional吗` in an `if` statement, or you can pass `oddLeaf2上
	nDimensional吗` as a predicate to a filtering utility described in the module docstring.

	Parameters
	----------
	dimension : DimensionIndex
		The bit position to test.
	leaf : Leaf
		The leaf index to test.

	Returns
	-------
	isOddLeaf : bool
		`True` if the bit at position `dimension` is set in `leaf`.

	References
	----------
	[1] gmpy2 - `bit_test` function
		https://gmpy2.readthedocs.io/en/latest/gmpy2.html#gmpy2.bit_test
	"""
	return isBit1吗(leaf, dimension)
