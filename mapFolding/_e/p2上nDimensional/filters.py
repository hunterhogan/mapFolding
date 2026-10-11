"""Test leaf indices and individual dimension coordinates.

(AI generated docstring)

You can select leaf-space values above `1` or test a coordinate bit of a leaf index. You receive
Boolean results without modifying the inputs.

Contents
--------
Functions
	moreThanLeaf零吗
		Test whether `leaf` is greater than the first non-origin leaf.
	oddLeaf2上nDimensional吗
		Test whether the coordinate bit at `dimension` is set in `leaf`.
"""
from __future__ import annotations

from gmpy2 import bit_test as isBit1吗
from mapFolding._e.p2上nDimensional import 零
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.theTypes import DimensionIndex, LeafSpace
	from mapFolding.theTypes import Leaf, Pile

def moreThanLeaf零吗(leaf: LeafSpace) -> bool:
	"""Test whether `leaf` is greater than the first non-origin leaf.

	You can exclude the origin leaf and the leaf numbered `1` from a sequence of leaf indices by
	comparing the numeric value directly with `零` [1]. You do not distinguish a leaf index from an
	encoded domain in this comparison.

	Parameters
	----------
	leaf : LeafSpace
		Numeric leaf-space value to compare with `1`.

	Returns
	-------
	leafIsNotOriginOrZero : bool
		`True` when the numeric value of `leaf` is greater than `1`.

	Examples
	--------
	From `pinByDomain.py`, with the coordinate values already computed.

		```python
		filter(moreThanLeaf零吗, mapShape首ProductsSumsInSubSubHyperplane)
		```

	References
	----------
	[1] `mapFolding._e.p2上nDimensional.semiotics.零`
	"""
	return 零 < leaf

def oddLeaf2上nDimensional吗(dimension: DimensionIndex, leaf: Leaf) -> bool:
	"""Test whether the coordinate bit at `dimension` is set in `leaf`.

	You can inspect whether `leaf` has coordinate `1` on the axis indexed by
	`dimension`. The bit test uses `gmpy2.bit_test` [1].

	Parameters
	----------
	dimension : DimensionIndex
		Nonnegative dimension index, counted from the least significant bit.
	leaf : Leaf
		Leaf index whose coordinate is tested.

	Returns
	-------
	isOddLeaf : bool
		`True` when the bit at `dimension` is set in `leaf`.

	References
	----------
	[1] gmpy2 integer operations.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return isBit1吗(leaf, dimension)
