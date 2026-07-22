from __future__ import annotations

from gmpy2 import bit_test as isBit1吗
from humpy_cytoolz import curry as syntacticCurry
from mapFolding._e._2上nDimensional import 零
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.theTypes import DimensionIndex, Leaf, LeafSpace

def notLeafOriginOrLeaf零(leaf: LeafSpace) -> bool:
	"""Test to ensure `leaf` is not `leafOrigin` (0) or `leaf零` (1).

	You can use `notLeafOriginOrLeaf零` in an `if` statement, or you can pass `notLeafOriginOrLeaf零` as a predicate to a
	filtering utility described in the module docstring.

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

@syntacticCurry
def oddLeaf2上nDimensional吗(dimension: DimensionIndex, leaf: Leaf) -> bool:
	return isBit1吗(leaf, dimension)
