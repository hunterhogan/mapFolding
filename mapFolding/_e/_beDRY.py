"""You can use this module to share elimination-algorithm utilities that avoid `EliminationState` imports.

This module is a workbench utility layer for `mapFolding._e` algorithms. The module primarily contains utilities that are intended
to work beyond the $2^n$-dimensional special case.

You should avoid putting functions in this module that only work on $2^n$-dimensional maps. You cannot import `EliminationState`
into this module without causing circular import problems. This constraint exists as of 2026-01-26.

Contents
--------
Disaggregation and deconstruction functions
	DOTitems
		You can iterate over `(key, value)` pairs in a `Mapping`.
	DOTkeys
		You can iterate over keys in a `Mapping`.
	DOTvalues
		You can iterate over values in a `Mapping`.
	getIteratorOfLeaves
		You can iterate over each `Leaf` bit that is set in a `ChoicesLeaf`.

`ChoicesLeaf` functions
	getAntiChoicesLeaf
		You can build a complement `ChoicesLeaf` by clearing each `leaf` bit.
	makeChoicesLeaf
		You can build a `ChoicesLeaf` by setting each `leaf` bit.
	choicesLeafLeafNone
		You can normalize a `ChoicesLeaf` into a `Leaf` or `None` when the range is degenerate.
	choicesLeafAND
		You can AND a `ChoicesLeaf` with a disposable mask.

Be DRY functions
	getMapShapeProducts
		You can compute prefix products of `mapShape` dimension lengths.
	getMapShapeProductsSums
		You can compute prefix sums of `getMapShapeProducts(mapShape)`.
	getMapShape首ProductsSums
		You can compute prefix sums of reversed dimension products for head-first coordinate arithmetic.
	reverseLookup
		You can find a key in a `dict` by matching a value.

Flow control
	indicesMapShapeDimensionLengthsAreEqual
		You can group dimension indices by repeated dimension lengths.
	mapShapeIs2上nDimensions
		You can test whether `mapShape` is a $2^n$-dimensional map, optionally with a minimum dimension count.

References
----------
[1] mapFolding._e.dataBaskets.EliminationState
	Internal package reference

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
	from mapFolding._e.theTypes import ChoicesLeaf, Leaf

#======== `ChoicesLeaf` functions ================================================

def lengthChoicesLeaf(choicesLeaf: ChoicesLeaf) -> int:
	"""Count the number of `Leaf` indices encoded in a `ChoicesLeaf` bitset.

	You can use this function to determine the cardinality of the domain represented by
	`choicesLeaf`. The function counts the number of set bits in `choicesLeaf` minus one
	(the sentinel bit) [1]. The result represents how many distinct `Leaf` indices are
	present in `choicesLeaf`.

	Parameters
	----------
	choicesLeaf : ChoicesLeaf
		Bitset encoding a set of `Leaf` indices.

	Returns
	-------
	leavesCount : int
		The number of `Leaf` indices with set bits in `choicesLeaf`, excluding the sentinel bit.

	Examples
	--------
	The function is used to identify groups of piles sharing the same domain cardinality.

		itemfilter(lambda groupBy: (lengthChoicesLeaf(groupBy[choicesLeafKey])) == len(groupBy[piles]), groupByChoicesLeaf)

	References
	----------
	[1] gmpy2.mpz.bit_count - gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/mpz.html#gmpy2.mpz.bit_count

	"""
	return choicesLeaf.bit_count() - 1

def makeAntiChoicesLeaf(leavesTotal: int, leaves: Iterable[Leaf]) -> ChoicesLeaf:
	"""You can build a complement `ChoicesLeaf` by clearing each `Leaf` bit in `leaves`.

	The returned `ChoicesLeaf` contains a bit for every `Leaf` in `range(leavesTotal)` except each `Leaf` in `leaves`.
	The returned `ChoicesLeaf` also preserves the sentinel bit that indicates the value is a `ChoicesLeaf`.

	Parameters
	----------
	leavesTotal : int
		Total number of leaves in the map.
	leaves : Iterable[Leaf]
		Iterable of `Leaf` indices to exclude from the returned `ChoicesLeaf`.

	Returns
	-------
	antiChoicesLeaf : ChoicesLeaf
		`ChoicesLeaf` bitset containing each allowed `Leaf` plus the `ChoicesLeaf` sentinel bit.

	Examples
	--------
	The function is used to start from the full domain.

		antiChoicesLeaf: ChoicesLeaf = getAntiChoicesLeaf(state.leavesTotal, frozenset())

	The function is used to exclude every `Leaf` not in a crease relation.

		antiChoicesLeaf = getAntiChoicesLeaf(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease))

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding.inclusive
	"""
	return reduce(bit_clear, leaves, bit_mask(leavesTotal + inclusive))

def makeChoicesLeaf(leavesTotal: int, leaves: Iterable[Leaf]) -> ChoicesLeaf:
	"""You can build a `ChoicesLeaf` by setting each `Leaf` bit in `leaves`.

	The returned `ChoicesLeaf` contains the sentinel bit that indicates the value is a `ChoicesLeaf`. The returned
	`ChoicesLeaf` also contains a bit for each `Leaf` in `leaves`.

	Parameters
	----------
	leavesTotal : int
		Total number of leaves in the map.
	leaves : Iterable[Leaf]
		Iterable of `Leaf` indices to include in the returned `ChoicesLeaf`.

	Returns
	-------
	choicesLeaf : ChoicesLeaf
		`ChoicesLeaf` bitset containing each `Leaf` in `leaves` plus the `ChoicesLeaf` sentinel bit.

	Examples
	--------
	The function is used to create a domain bitset before normalizing with `choicesLeafLeafNone`.

		permutationSpace2上nDomainDefaults: PermutationSpace = {pile: raiseIfNone(choicesLeafLeafNone(makeChoicesLeaf(state.leavesTotal, choicesLeaf)))
											for pile, choicesLeaf in getDictionaryChoicesLeaf(state).items()}

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding._e._beDRY.choicesLeafLeafNone
	"""
	return reduce(bit_set, leaves, bit_set(0, leavesTotal))

# SEMIOTICS
def choicesLeafLeafNone(choicesLeaf: ChoicesLeaf, /) -> ChoicesLeaf | Leaf | None:
	"""You can normalize a `ChoicesLeaf` into a `Leaf`, `ChoicesLeaf`, or `None` when the range is degenerate.

	When `choicesLeaf` is a `ChoicesLeaf`, `choicesLeaf` contains one sentinel bit that indicates the value is a `ChoicesLeaf`.
	This function interprets the total set-bit count as a compact encoding of domain cardinality.

	- When `choicesLeaf.bit_count() == 1`, `choicesLeaf` is an empty domain. The only set bit is the sentinel bit, so the function returns `None`.
	- When `choicesLeaf.bit_count() == 2`, `choicesLeaf` contains exactly one `Leaf` plus the sentinel bit. The function converts the range to a
		`Leaf` by returning `raiseIfNone(choicesLeaf.bit_scan1())`.
	- Otherwise, the function returns `choicesLeaf` unchanged.

	Parameters
	----------
	choicesLeaf : ChoicesLeaf
		`ChoicesLeaf` to inspect.

	Returns
	-------
	leafSpaceOrNone : Leaf | ChoicesLeaf | None
		A `Leaf` when `choicesLeaf` encodes exactly one leaf, `None` when `choicesLeaf` encodes an empty domain, or `choicesLeaf` otherwise.

	Examples
	--------
	The function is used to normalize a masked domain.

		if (ImaLeafSpaceNotAWalrusSubscript := choicesLeafLeafNone(choicesLeafAND(antiChoicesLeaf, choicesLeaf))) is None:
			return {}

	The function is used to normalize per-pile domains into pinned leaves when possible.

		permutationSpace2上nDomainDefaults: PermutationSpace = {pile: raiseIfNone(choicesLeafLeafNone(makeChoicesLeaf(state.leavesTotal, choicesLeaf)))
											for pile, choicesLeaf in getDictionaryChoicesLeaf(state).items()}

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding._e.filters.thisIsAChoicesLeaf

	[3] hunterMakesPy - Context7
		https://context7.com/hunterhogan/huntermakespy

	"""
	whoAmI: ChoicesLeaf | Leaf | None = choicesLeaf
	if choicesLeaf吗(choicesLeaf):
		if choicesLeaf.bit_count() == 2:
			whoAmI = raiseIfNone(choicesLeaf.bit_scan1())
		elif choicesLeaf.bit_count() == 1:
			whoAmI = None
	return whoAmI

def choicesLeafAND(choicesLeafDISPOSABLE: ChoicesLeaf, choicesLeaf: ChoicesLeaf) -> ChoicesLeaf:
	"""Compute the bitwise AND of two `ChoicesLeaf`.

	You can use this function to mask `choicesLeaf` with `choicesLeafDISPOSABLE` [1]. The
	function performs bitwise AND and returns the intersection of the two leaf sets.

	Parameters
	----------
	choicesLeafDISPOSABLE : ChoicesLeaf
		Bitset mask applied to `choicesLeaf`.
	choicesLeaf : ChoicesLeaf
		Bitset to be masked by `choicesLeafDISPOSABLE`.

	Returns
	-------
	maskedChoicesLeaf : ChoicesLeaf
		Bitwise AND of `choicesLeaf` and `choicesLeafDISPOSABLE`.
	"""
	return choicesLeaf & choicesLeafDISPOSABLE

#======== Be DRY functions ================================================

def getMapShapeProducts(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	"""You can compute prefix products of each dimension length in `mapShape`.

	The returned tuple starts with the product of zero dimensions, which is `1`. Each subsequent element multiplies the next
	dimension length in `mapShape`.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.

	Returns
	-------
	mapShapeProducts : tuple[int, ...]
		Tuple of prefix products with `mapShapeProducts[0] == 1`.

	Examples
	--------
	The function is used during `EliminationState` initialization.

		self.mapShapeProducts = getMapShapeProducts(self.mapShape)

	References
	----------
	[1] itertools.accumulate
		https://docs.python.org/3/library/itertools.html#itertools.accumulate
	[2] operator.mul
		https://docs.python.org/3/library/operator.html#operator.mul
	[3] mapFolding._e.dataBaskets.EliminationState
	"""
	return tuple(accumulate(mapShape, mul, initial=1))

def getMapShapeProductsSums(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	"""You can compute prefix sums of `getMapShapeProducts(mapShape)`.

	The returned tuple starts with the sum of zero products, which is `0`. Each subsequent element adds the next product from
	`getMapShapeProducts(mapShape)`.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.

	Returns
	-------
	mapShapeProductsSums : tuple[int, ...]
		Tuple of prefix sums with `mapShapeProductsSums[0] == 0`.

	Examples
	--------
	The function is used during `EliminationState` initialization.

		self.mapShapeProductsSums = getMapShapeProductsSums(self.mapShape)

	References
	----------
	[1] itertools.accumulate
		https://docs.python.org/3/library/itertools.html#itertools.accumulate
	[2] operator.add
		https://docs.python.org/3/library/operator.html#operator.add
	[3] mapFolding._e._beDRY.getMapShapeProducts

	[4] mapFolding._e.dataBaskets.EliminationState
	"""
	return tuple(accumulate(getMapShapeProducts(mapShape), add, initial=0))

def getMapShape首ProductsSums(mapShapeProducts: tuple[int, ...], dimensionsTotal: int | None = None, dimensionFrom首: int | None = None) -> tuple[int, ...]:
	"""Compute prefix sums of reversed dimension products for head-first coordinate arithmetic.

	You can use this function to obtain a tuple of cumulative sums computed from reversed
	dimension products. This tuple is useful when you are using integers as proxies for
	Cartesian coordinates in multidimensional space [1] and you need to compute offsets
	from the "anti-origin" (the maximum coordinate) rather than from the origin.

	The function reverses the first `dimensionFrom首` dimension products from `mapShapeProducts`,
	then computes prefix sums [2] of the reversed products. This provides a complementary
	perspective to `getMapShapeProductsSums` [3] by ordering dimension products in
	descending order before summation.

	Parameters
	----------
	mapShapeProducts : tuple[int, ...]
		Prefix products of dimension lengths, typically from `getMapShapeProducts` [4].
	dimensionsTotal : int | None = None
		Total number of dimensions in the map. When `None`, inferred as
		`len(mapShapeProducts) - 1`.
	dimensionFrom首 : int | None = None
		Dimension index defining which products to include in the sum computation. When `None`,
		defaults to `dimensionsTotal`. This parameter controls how many dimension products are
		reversed and summed.

	Returns
	-------
	mapShape首ProductsSums : tuple[int, ...]
		Tuple of prefix sums computed from reversed dimension products. Element `[i]` contains
		the sum of the first `i` elements of the reversed product sequence.

	Examples
	--------
	The function is used during state initialization to compute head-first sums.

		self.mapShape首ProductsSums = getMapShape首ProductsSums(self.mapShapeProducts, self.dimensionsTotal, self.dimensionsTotal)

	The function is used to compute offset bounds in sub-hyperplane computations.

		mapShape首ProductsSumsInSubHyperplane: tuple[int, ...] = getMapShape首ProductsSums(state.mapShapeProducts, state.dimensionsTotal, state.dimensionsTotal - 1)

	References
	----------
	[1] Integer encoding of multidimensional coordinates
		Internal implementation detail
	[2] itertools.accumulate
		https://docs.python.org/3/library/itertools.html#itertools.accumulate
	[3] mapFolding._e._beDRY.getMapShapeProductsSums

	[4] mapFolding._e._beDRY.getMapShapeProducts

	"""
	dimensionsTotal = dimensionsTotal or len(mapShapeProducts) - 1

	if dimensionFrom首 is None:
		dimensionFrom首 = dimensionsTotal

	mapShapeProductsTruncator: int = dimensionFrom首 - (dimensionsTotal + zeroIndexed)

	mapShapeProductsFrom首: tuple[int, ...] = mapShapeProducts[0:mapShapeProductsTruncator][::-1]

	mapShape首ProductsSums: tuple[int, ...] = tuple(accumulate(mapShapeProductsFrom首, add, initial=0))

	return mapShape首ProductsSums

#======== Flow control ================================================

def mapShapeLengthsAreEqual(mapShape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
	"""You can group dimension indices in `mapShape` by repeated dimension lengths.

	The returned `Iterator` yields one `tuple` per distinct dimension length in `mapShape` where the dimension length occurs more
	than once. Each yielded `tuple` contains each index where `mapShape[index]` equals the repeated dimension length.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.

	Returns
	-------
	iteratorIndicesSameDimensionLength : Iterator[tuple[int, ...]]
		Iterator of index tuples. Each tuple has length at least 2.

	Examples
	--------
	The function is used to iterate repeated dimension magnitudes during elimination.

		for indicesSameDimensionLength in mapShapeLengthsAreEqual(state.mapShape):
			state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
			for 次k, 次r in pairwise(indicesSameDimensionLength):
				state = excludeLeaf_rBeforeLeaf_k(state, state.mapShapeProducts[次k], state.mapShapeProducts[次r])

	References
	----------
	[1] cytoolz.itertoolz.unique
		https://toolz.readthedocs.io/en/latest/api.html#toolz.itertoolz.unique
	[2] more_itertools.iter_index
		https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.iter_index
	[3] mapFolding._e.algorithms.elimination.theorem4
	"""
	return filter(lambda indices: 1 < len(indices), map(tuple, map(partial(iter_index, mapShape), unique(filter((1).__lt__, mapShape)))))
