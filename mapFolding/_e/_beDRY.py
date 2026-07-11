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
		You can iterate over each `Leaf` bit that is set in a `LeafOptions`.

`LeafOptions` functions
	getAntiLeafOptions
		You can build a complement `LeafOptions` by clearing each `leaf` bit.
	makeLeafOptions
		You can build a `LeafOptions` by setting each `leaf` bit.
	JeanValjean
		You can normalize a `LeafOptions` into a `Leaf` or `None` when the range is degenerate.
	leafOptionsAND
		You can AND a `LeafOptions` with a disposable mask in a curry-friendly parameter order.

Be DRY functions
	getProductsOfDimensions
		You can compute prefix products of `mapShape` dimension lengths.
	getSumsOfProductsOfDimensions
		You can compute prefix sums of `getProductsOfDimensions(mapShape)`.
	getSumsOfProductsOfDimensionsNearest首
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
from humpy_cytoolz import curry as syntacticCurry, unique
from hunterMakesPy import inclusive, raiseIfNone, zeroIndexed
from itertools import accumulate
from mapFolding._e.filters import isLeafOptions吗
from more_itertools import iter_index
from operator import add, mul
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from mapFolding._e.theTypes import Leaf, LeafOptions, LeafSpace

#======== `LeafOptions` functions ================================================

def makeLeafAntiOptions(leavesTotal: int, leaves: Iterable[Leaf]) -> LeafOptions:
	"""You can build a complement `LeafOptions` by clearing each `Leaf` bit in `leaves`.

	The returned `LeafOptions` contains a bit for every `Leaf` in `range(leavesTotal)` except each `Leaf` in `leaves`.
	The returned `LeafOptions` also preserves the sentinel bit that indicates the value is a `LeafOptions`.

	Parameters
	----------
	leavesTotal : int
		Total number of leaves in the map.
	leaves : Iterable[Leaf]
		Iterable of `Leaf` indices to exclude from the returned `LeafOptions`.

	Returns
	-------
	antiLeafOptions : LeafOptions
		`LeafOptions` bitset containing each allowed `Leaf` plus the `LeafOptions` sentinel bit.

	Examples
	--------
	The function is used to start from the full domain.

		antiLeafOptions: LeafOptions = getAntiLeafOptions(state.leavesTotal, frozenset())

	The function is used to exclude every `Leaf` not in a crease relation.

		antiLeafOptions = getAntiLeafOptions(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease))

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding.inclusive
	"""
	return reduce(bit_clear, leaves, bit_mask(leavesTotal + inclusive))

def makeLeafOptions(leavesTotal: int, leaves: Iterable[Leaf]) -> LeafOptions:
	"""You can build a `LeafOptions` by setting each `Leaf` bit in `leaves`.

	The returned `LeafOptions` contains the sentinel bit that indicates the value is a `LeafOptions`. The returned
	`LeafOptions` also contains a bit for each `Leaf` in `leaves`.

	Parameters
	----------
	leavesTotal : int
		Total number of leaves in the map.
	leaves : Iterable[Leaf]
		Iterable of `Leaf` indices to include in the returned `LeafOptions`.

	Returns
	-------
	leafOptions : LeafOptions
		`LeafOptions` bitset containing each `Leaf` in `leaves` plus the `LeafOptions` sentinel bit.

	Examples
	--------
	The function is used to create a domain bitset before normalizing with `JeanValjean`.

		permutationSpace2上nDomainDefaults: PermutationSpace = {pile: raiseIfNone(JeanValjean(makeLeafOptions(state.leavesTotal, leafOptions)))
											for pile, leafOptions in getDictionaryLeafOptions(state).items()}

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding._e._beDRY.JeanValjean
	"""
	return reduce(bit_set, leaves, bit_set(0, leavesTotal))

def howManyLeavesInLeafOptions(leafOptions: LeafOptions) -> int:
	"""Count the number of `Leaf` indices encoded in a `LeafOptions` bitset.

	You can use this function to determine the cardinality of the domain represented by
	`leafOptions`. The function counts the number of set bits in `leafOptions` minus one
	(the sentinel bit) [1]. The result represents how many distinct `Leaf` indices are
	present in `leafOptions`.

	Parameters
	----------
	leafOptions : LeafOptions
		Bitset encoding a set of `Leaf` indices.

	Returns
	-------
	leavesCount : int
		The number of `Leaf` indices with set bits in `leafOptions`, excluding the sentinel bit.

	Examples
	--------
	The function is used to identify groups of piles sharing the same domain cardinality.

		itemfilter(lambda groupBy: (howManyLeavesInLeafOptions(groupBy[leafOptionsKey])) == len(groupBy[piles]), groupByLeafOptions)

	References
	----------
	[1] gmpy2.mpz.bit_count - gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/mpz.html#gmpy2.mpz.bit_count

	"""
	return leafOptions.bit_count() - 1

# SEMIOTICS `JeanValjean`.
def JeanValjean(p24601: LeafOptions, /) -> LeafSpace | None:
	"""You can normalize a `LeafOptions` into a `Leaf` or `None` when the range is degenerate.

	When `p24601` is a `LeafOptions`, `p24601` contains one sentinel bit that indicates the value is a `LeafOptions`.
	This function interprets the total set-bit count as a compact encoding of domain cardinality.

	- When `p24601.bit_count() == 1`, `p24601` is an empty domain. The only set bit is the sentinel bit, so the function returns `None`.
	- When `p24601.bit_count() == 2`, `p24601` contains exactly one `Leaf` plus the sentinel bit. The function converts the range to a
		`Leaf` by returning `raiseIfNone(p24601.bit_scan1())`.
	- Otherwise, the function returns `p24601` unchanged.

	Parameters
	----------
	p24601 : LeafOptions
		Candidate `LeafOptions` value.

	Returns
	-------
	leafSpaceOrNone : LeafSpace | None
		A `Leaf` when `p24601` encodes exactly one leaf, `None` when `p24601` encodes an empty domain, or `p24601` otherwise.

	Examples
	--------
	The function is used to normalize a masked domain.

		if (ImaLeafSpaceNotAWalrusSubscript := JeanValjean(leafOptionsAND(antiLeafOptions, leafOptions))) is None:
			return {}

	The function is used to normalize per-pile domains into pinned leaves when possible.

		permutationSpace2上nDomainDefaults: PermutationSpace = {pile: raiseIfNone(JeanValjean(makeLeafOptions(state.leavesTotal, leafOptions)))
											for pile, leafOptions in getDictionaryLeafOptions(state).items()}

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding._e.filters.thisIsALeafOptions

	[3] hunterMakesPy - Context7
		https://context7.com/hunterhogan/huntermakespy

	"""
	whoAmI: LeafSpace | None = p24601
	if isLeafOptions吗(p24601):
		if p24601.bit_count() == 1:
			whoAmI = None
		elif p24601.bit_count() == 2:
			whoAmI = raiseIfNone(p24601.bit_scan1())
	return whoAmI

@syntacticCurry
def leafOptionsAND(leafOptionsDISPOSABLE: LeafOptions, leafOptions: LeafOptions) -> LeafOptions:
	"""Compute the bitwise AND of two `LeafOptions` with curry-friendly parameter order.

	You can use this function to mask `leafOptions` with `leafOptionsDISPOSABLE` [1]. The
	function performs bitwise AND and returns the intersection of the two leaf sets. The
	parameter order is reversed from typical AND operations to support currying [2]: you can
	partially apply `leafOptionsDISPOSABLE` to create a reusable masking function.

	Parameters
	----------
	leafOptionsDISPOSABLE : LeafOptions
		Bitset mask applied to `leafOptions`. This parameter is the first parameter to enable
		currying: you can fix `leafOptionsDISPOSABLE` and reuse the resulting function.
	leafOptions : LeafOptions
		Bitset to be masked by `leafOptionsDISPOSABLE`.

	Returns
	-------
	maskedLeafOptions : LeafOptions
		Bitwise AND of `leafOptions` and `leafOptionsDISPOSABLE`.

	Examples
	--------
	The function is used with `JeanValjean` to normalize masked domains.

		if (ImaLeafSpaceNotAWalrusSubscript := JeanValjean(leafOptionsAND(antiLeafOptions, leafOptions))) is None:
			return {}

	The function is used to mask pile domains during constraint propagation.

		leafSpace: LeafSpace | None = JeanValjean(leafOptionsAND(leafAntiOptions, leafOptions))

	Important
	---------
	The parameter order is reversed from typical bitwise AND operations. The first parameter
	`leafOptionsDISPOSABLE` is the mask, and the second parameter `leafOptions` is the value
	being masked. This order facilitates currying.

	References
	----------
	[1] gmpy2.mpz bitwise operations - gmpy2 documentation
		https://gmpy2.readthedocs.io/en/latest/mpz.html#mpz-methods
	[2] cytoolz.functoolz.curry
		https://toolz.readthedocs.io/en/latest/api.html#toolz.functoolz.curry

	"""
	return leafOptions & leafOptionsDISPOSABLE

#======== Be DRY functions ================================================

def getProductsOfDimensions(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	"""You can compute prefix products of each dimension length in `mapShape`.

	The returned tuple starts with the product of zero dimensions, which is `1`. Each subsequent element multiplies the next
	dimension length in `mapShape`.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.

	Returns
	-------
	productsOfDimensions : tuple[int, ...]
		Tuple of prefix products with `productsOfDimensions[0] == 1`.

	Examples
	--------
	The function is used during `EliminationState` initialization.

		self.productsOfDimensions = getProductsOfDimensions(self.mapShape)

	References
	----------
	[1] itertools.accumulate
		https://docs.python.org/3/library/itertools.html#itertools.accumulate
	[2] operator.mul
		https://docs.python.org/3/library/operator.html#operator.mul
	[3] mapFolding._e.dataBaskets.EliminationState
	"""
	return tuple(accumulate(mapShape, mul, initial=1))

def getSumsOfProductsOfDimensions(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	"""You can compute prefix sums of `getProductsOfDimensions(mapShape)`.

	The returned tuple starts with the sum of zero products, which is `0`. Each subsequent element adds the next product from
	`getProductsOfDimensions(mapShape)`.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.

	Returns
	-------
	sumsOfProductsOfDimensions : tuple[int, ...]
		Tuple of prefix sums with `sumsOfProductsOfDimensions[0] == 0`.

	Examples
	--------
	The function is used during `EliminationState` initialization.

		self.sumsOfProductsOfDimensions = getSumsOfProductsOfDimensions(self.mapShape)

	References
	----------
	[1] itertools.accumulate
		https://docs.python.org/3/library/itertools.html#itertools.accumulate
	[2] operator.add
		https://docs.python.org/3/library/operator.html#operator.add
	[3] mapFolding._e._beDRY.getProductsOfDimensions

	[4] mapFolding._e.dataBaskets.EliminationState
	"""
	return tuple(accumulate(getProductsOfDimensions(mapShape), add, initial=0))

def getSumsOfProductsOfDimensionsNearest首(productsOfDimensions: tuple[int, ...], dimensionsTotal: int | None = None, dimensionFrom首: int | None = None) -> tuple[int, ...]:
	"""Compute prefix sums of reversed dimension products for head-first coordinate arithmetic.

	You can use this function to obtain a tuple of cumulative sums computed from reversed
	dimension products. This tuple is useful when you are using integers as proxies for
	Cartesian coordinates in multidimensional space [1] and you need to compute offsets
	from the "anti-origin" (the maximum coordinate) rather than from the origin.

	The function reverses the first `dimensionFrom首` dimension products from `productsOfDimensions`,
	then computes prefix sums [2] of the reversed products. This provides a complementary
	perspective to `getSumsOfProductsOfDimensions` [3] by ordering dimension products in
	descending order before summation.

	Parameters
	----------
	productsOfDimensions : tuple[int, ...]
		Prefix products of dimension lengths, typically from `getProductsOfDimensions` [4].
	dimensionsTotal : int | None = None
		Total number of dimensions in the map. When `None`, inferred as
		`len(productsOfDimensions) - 1`.
	dimensionFrom首 : int | None = None
		Dimension index defining which products to include in the sum computation. When `None`,
		defaults to `dimensionsTotal`. This parameter controls how many dimension products are
		reversed and summed.

	Returns
	-------
	sumsOfProductsOfDimensionsNearest首 : tuple[int, ...]
		Tuple of prefix sums computed from reversed dimension products. Element `[i]` contains
		the sum of the first `i` elements of the reversed product sequence.

	Examples
	--------
	The function is used during state initialization to compute head-first sums.

		self.sumsOfProductsOfDimensionsNearest首 = getSumsOfProductsOfDimensionsNearest首(self.productsOfDimensions, self.dimensionsTotal, self.dimensionsTotal)

	The function is used to compute offset bounds in sub-hyperplane computations.

		sumsOfProductsOfDimensionsNearest首InSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, state.dimensionsTotal, state.dimensionsTotal - 1)

	References
	----------
	[1] Integer encoding of multidimensional coordinates
		Internal implementation detail
	[2] itertools.accumulate
		https://docs.python.org/3/library/itertools.html#itertools.accumulate
	[3] mapFolding._e._beDRY.getSumsOfProductsOfDimensions

	[4] mapFolding._e._beDRY.getProductsOfDimensions

	"""
	dimensionsTotal = dimensionsTotal or len(productsOfDimensions) - 1

	if dimensionFrom首 is None:
		dimensionFrom首 = dimensionsTotal

	productsOfDimensionsTruncator: int = dimensionFrom首 - (dimensionsTotal + zeroIndexed)

	productsOfDimensionsFrom首: tuple[int, ...] = productsOfDimensions[0:productsOfDimensionsTruncator][::-1]

	sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = tuple(accumulate(productsOfDimensionsFrom首, add, initial=0))

	return sumsOfProductsOfDimensionsNearest首

#======== Flow control ================================================

def indicesMapShapeDimensionLengthsAreEqual(mapShape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
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

		for indicesSameDimensionLength in indicesMapShapeDimensionLengthsAreEqual(state.mapShape):
			state.Theorem4Multiplier *= factorial(len(indicesSameDimensionLength))
			for index_k, index_r in pairwise(indicesSameDimensionLength):
				state = excludeLeaf_rBeforeLeaf_k(state, state.productsOfDimensions[index_k], state.productsOfDimensions[index_r])

	References
	----------
	[1] cytoolz.itertoolz.unique
		https://toolz.readthedocs.io/en/latest/api.html#toolz.itertoolz.unique
	[2] more_itertools.iter_index
		https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.iter_index
	[3] mapFolding._e.algorithms.elimination.theorem4
	"""
	return filter(lambda indices: 1 < len(indices), map(tuple, map(partial(iter_index, mapShape), unique(mapShape))))
