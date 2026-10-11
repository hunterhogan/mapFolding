"""Access shared bitset operations and map-shape arithmetic without state imports.

You can construct and inspect leaf sets and compute dimension products and sums. Pass values directly
to use these operations in modules that define elimination state without introducing circular imports.

Contents
--------
ChoicesLeaf functions
	choicesLeafAND
		Intersect two leaf bitsets.
	choicesLeafLeafNone
		Normalize a bitset to a leaf, an empty result, or a bitset.
	lengthChoicesLeaf
		Count leaves in a sentinel-bearing bitset.
	makeAntiChoicesLeaf
		Build a leaf bitset by excluding the specified leaf indices.
	makeChoicesLeaf
		Build a bitset from the specified leaf indices.
Be DRY functions
	getMapShapeProducts
		Compute cumulative products of the dimension lengths in a map shape.
	getMapShapeProductsSums
		Compute cumulative sums of a map shape's dimension products.
	getMapShape首ProductsSums
		Compute cumulative dimension-product sums from the head of a map.
Flow control
	mapShapeLengthsAreEqual
		Group dimension indices that share a repeated length greater than one.
"""
from __future__ import annotations

from functools import partial, reduce
from gmpy2 import bit_clear, bit_mask, bit_set
from humpy_cytoolz import unique
from hunterMakesPy import inclusive, raiseIfNone, zeroIndexed
from itertools import accumulate
from mapFolding._e.filters import choicesLeaf吗
from more_itertools import iter_index
from operator import add, mul
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from mapFolding._e.theTypes import ChoicesLeaf
	from mapFolding.theTypes import Leaf

#======== `ChoicesLeaf` functions ================================================

def lengthChoicesLeaf(choicesLeaf: ChoicesLeaf) -> int:
	"""Count leaves in a sentinel-bearing bitset.

	You can count the leaf indices represented by `choicesLeaf`, excluding the sentinel from the set-bit
	count provided by `gmpy2.mpz.bit_count` [1].

	Parameters
	----------
	choicesLeaf : ChoicesLeaf
		Leaf bitset with one sentinel bit above all leaf bits.

	Returns
	-------
	leavesCount : int
		Number of possible leaves, excluding the sentinel.

	Examples
	--------
	The subset reducer compares leaf count with the number of matching piles.

		```python
		if _e.lengthChoicesLeaf(choicesLeaf) == len(boxOfPiles):
			permutationSpace = reduceLeafSpace(
				permutationSpace, DOTitems(dissoc(pilesUndetermined, *boxOfPiles)),
				_e.makeAntiChoicesLeaf(state.totalLeaves, _e.getIteratorOfLeaves(choicesLeaf))
			)
		```

	References
	----------
	[1] gmpy2 integer operations and the mpz type.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return choicesLeaf.bit_count() - 1

def makeAntiChoicesLeaf(totalLeaves: int, leaves: Iterable[Leaf]) -> ChoicesLeaf:
	"""Build a leaf bitset by excluding the specified leaf indices.

	You can obtain a bitset containing every leaf index below `totalLeaves` except the indices in
	`leaves`. You retain the sentinel while clearing excluded bits with `gmpy2.bit_clear` [1].

	Parameters
	----------
	totalLeaves : int
		Total number of leaves and index of the sentinel bit.
	leaves : Iterable[Leaf]
		Leaf indices to exclude. Each index must be below `totalLeaves`.

	Returns
	-------
	antiChoicesLeaf : ChoicesLeaf
		Bitset containing the remaining leaves and the sentinel.

	Examples
	--------
	State normalization excludes leaves already pinned in the permutation space.

		```python
		antiChoicesLeaf = _e.makeAntiChoicesLeaf(len(self), leavesPinned.values())
		```

	References
	----------
	[1] gmpy2 integer operations and the mpz type.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return reduce(bit_clear, leaves, bit_mask(totalLeaves + inclusive))

def makeChoicesLeaf(totalLeaves: int, leaves: Iterable[Leaf]) -> ChoicesLeaf:
	"""Build a bitset from the specified leaf indices.

	You can encode `leaves` as a bitset with a sentinel at `totalLeaves`. You set the leaf bits with
	`gmpy2.bit_set` [1].

	Parameters
	----------
	totalLeaves : int
		Total number of leaves and index of the sentinel bit.
	leaves : Iterable[Leaf]
		Leaf indices to include. Each index must be below `totalLeaves`.

	Returns
	-------
	choicesLeaf : ChoicesLeaf
		Bitset containing the requested leaves and the sentinel.

	Examples
	--------
	The pile-options implementation encodes the leaves that pass its filters.

		```python
		return makeChoicesLeaf(totalLeaves, choicesLeaf)
		```

	References
	----------
	[1] gmpy2 integer operations and the mpz type.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	return reduce(bit_set, leaves, bit_set(0, totalLeaves))

# SEMIOTICS
def choicesLeafLeafNone(choicesLeaf: ChoicesLeaf, /) -> ChoicesLeaf | Leaf | None:
	"""Normalize a bitset to a leaf, an empty result, or a bitset.

	You can distinguish an empty leaf set from a forced leaf assignment. You receive a bitset when
	several leaves remain, or the unchanged input when its value is not an instance of the bitset type.

	Bitset normalization
	--------------------
	You can inspect the set-bit count and first set bit with `gmpy2.mpz.bit_count` and
	`gmpy2.mpz.bit_scan1` [1]. Two set bits identify one leaf plus the sentinel. One set bit identifies
	the sentinel alone.

	Parameters
	----------
	choicesLeaf : ChoicesLeaf
		Bitset with a sentinel above all leaf bits.

	Returns
	-------
	leafSpaceOrNone : ChoicesLeaf | Leaf | None
		The only remaining leaf, `None` for a sentinel-only bitset, or the unchanged value.

	References
	----------
	[1] gmpy2 integer operations and the mpz type.
		https://gmpy2.readthedocs.io/en/latest/mpz.html
	"""
	whoAmI: ChoicesLeaf | Leaf | None = choicesLeaf
	if choicesLeaf吗(choicesLeaf):
		if choicesLeaf.bit_count() == 2:
			whoAmI = raiseIfNone(choicesLeaf.bit_scan1())
		elif choicesLeaf.bit_count() == 1:
			whoAmI = None
	return whoAmI

def choicesLeafAND(choicesLeafDISPOSABLE: ChoicesLeaf, choicesLeaf: ChoicesLeaf) -> ChoicesLeaf:
	"""Intersect two leaf bitsets.

	You can keep leaf indices present in both `choicesLeafDISPOSABLE` and `choicesLeaf`. You receive a
	new bitset with both arguments unchanged.

	Parameters
	----------
	choicesLeafDISPOSABLE : ChoicesLeaf
		Leaf mask containing allowed leaf bits and the sentinel.
	choicesLeaf : ChoicesLeaf
		Leaf set to restrict.

	Returns
	-------
	maskedChoicesLeaf : ChoicesLeaf
		Intersection retaining the sentinel when both inputs contain the sentinel.
	"""
	return choicesLeaf & choicesLeafDISPOSABLE

#======== Be DRY functions ================================================

def getMapShapeProducts(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	"""Compute cumulative products of the dimension lengths in a map shape.

	You can convert `mapShape` into the strides used to identify leaves across dimensions. You receive a
	tuple that includes the initial empty product and the final product of every dimension.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimension lengths in their original order.

	Returns
	-------
	mapShapeProducts : tuple[int, ...]
		Cumulative products beginning with `1`, with length `len(mapShape) + 1`.

	Examples
	--------
	Elimination-state initialization stores the products.

		```python
		self.mapShapeProducts = _e.getMapShapeProducts(self.mapShape)
		```
	"""
	return tuple(accumulate(mapShape, mul, initial=1))

def getMapShapeProductsSums(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	"""Compute cumulative sums of a map shape's dimension products.

	You can sum the cumulative products from `getMapShapeProducts` [1]. You receive a tuple that includes
	the initial empty sum and the sum including the total leaf count.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimension lengths in their original order.

	Returns
	-------
	mapShapeProductsSums : tuple[int, ...]
		Cumulative sums beginning with `0`, with length `len(mapShape) + 2`.

	References
	----------
	[1] `getMapShapeProducts`
	"""
	return tuple(accumulate(getMapShapeProducts(mapShape), add, initial=0))

def getMapShape首ProductsSums(mapShapeProducts: tuple[int, ...], totalDimensions: int | None = None, dimensionFrom首: int | None = None) -> tuple[int, ...]:
	"""Compute cumulative dimension-product sums from the head of a map.

	You can use `mapShapeProducts` to compute bounds measured from the largest dimension stride toward
	the smallest dimension stride. You receive a tuple beginning with the empty sum.

	Product selection
	-----------------
	The slice endpoint is `dimensionFrom首 - (totalDimensions + 1)`. With the default dimension count, the
	slice retains `mapShapeProducts[:dimensionFrom首]`. The total leaf-count product is excluded at the
	default endpoint.

	Parameters
	----------
	mapShapeProducts : tuple[int, ...]
		Cumulative dimension products, including the initial empty product and total leaf count.
	totalDimensions : int | None = None
		Dimension count. A false-valued value infers `len(mapShapeProducts) - 1`.
	dimensionFrom首 : int | None = None
		Number of leading dimension products to retain. `None` uses `totalDimensions`.

	Returns
	-------
	mapShape首ProductsSums : tuple[int, ...]
		Cumulative sums of the retained products in reverse order, beginning with `0`.

	Examples
	--------
	Elimination-state initialization requests all dimension strides.

		```python
		self.mapShape首ProductsSums = _e.getMapShape首ProductsSums(
			self.mapShapeProducts, self.totalDimensions, self.totalDimensions
		)
		```
	"""
	totalDimensions = totalDimensions or len(mapShapeProducts) - 1

	if dimensionFrom首 is None:
		dimensionFrom首 = totalDimensions

	mapShapeProductsTruncator: int = dimensionFrom首 - (totalDimensions + zeroIndexed)

	mapShapeProductsFrom首: tuple[int, ...] = mapShapeProducts[0:mapShapeProductsTruncator][::-1]

	mapShape首ProductsSums: tuple[int, ...] = tuple(accumulate(mapShapeProductsFrom首, add, initial=0))

	return mapShape首ProductsSums

#======== Flow control ================================================

def mapShapeLengthsAreEqual(mapShape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
	"""Group dimension indices that share a repeated length greater than one.

	You can identify dimension groups that can share an ordering restriction. You receive groups in the
	order of the first occurrence of each length in `mapShape`, excluding dimension lengths of one.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimension lengths in their original order.

	Returns
	-------
	indicesSameDimensionLength : Iterator[tuple[int, ...]]
		Iterator of ascending index tuples, each containing at least two indices.

	Examples
	--------
	The elimination algorithm applies symmetry restrictions to repeated dimension lengths.

		```python
		for indicesSameDimensionLength in mapShapeLengthsAreEqual(state.mapShape):
			state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
		```
	"""
	return filter(lambda indices: 1 < len(indices), map(tuple, map(partial(iter_index, mapShape), unique(filter((1).__lt__, mapShape)))))
