"""You can use this module to share elimination-algorithm utilities that avoid `EliminationState` imports.

This module is a workbench utility layer for `mapFolding._e` algorithms. The module primarily contains utilities that are intended
to work beyond the $2^n$-dimensional special case.

You should avoid putting functions in this module that only work on $2^n$-dimensional maps. You cannot import `EliminationState`
into this module without causing circular import problems. This constraint exists as of 2026-01-26.

Contents
--------
Group-by functions
	bifurcatePermutationSpace
		You can split a `PermutationSpace` into pinned leaves and pile-range domains.

Disaggregation and deconstruction functions
	DOTitems
		You can iterate over `(key, value)` pairs in a `Mapping`.
	DOTkeys
		You can iterate over keys in a `Mapping`.
	DOTvalues
		You can iterate over values in a `Mapping`.
	getIteratorOfLeaves
		You can iterate over each `Leaf` bit that is set in a `PileRangeOfLeaves`.

`PileRangeOfLeaves` functions
	DOTgetPileIfLeaf
		You can read `permutationSpace[pile]` only when `permutationSpace[pile]` is a `Leaf`.
	DOTgetPileIfPileRangeOfLeaves
		You can read `permutationSpace[pile]` only when `permutationSpace[pile]` is a `PileRangeOfLeaves`.
	getAntiPileRangeOfLeaves
		You can build a complement `PileRangeOfLeaves` by clearing each `leaf` bit.
	getPileRangeOfLeaves
		You can build a `PileRangeOfLeaves` by setting each `leaf` bit.
	JeanValjean
		You can normalize a `PileRangeOfLeaves` into a `Leaf` or `None` when the range is degenerate.
	pileRangeOfLeavesAND
		You can AND a `PileRangeOfLeaves` with a disposable mask in a curry-friendly parameter order.

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
from collections.abc import Iterable, Iterator, Mapping
from cytoolz.dicttoolz import dissoc as dissociatePiles
from cytoolz.functoolz import curry as syntacticCurry
from cytoolz.itertoolz import unique
from functools import partial, reduce
from gmpy2 import bit_clear, bit_mask, bit_set, xmpz
from hunterMakesPy import raiseIfNone
from itertools import accumulate
from mapFolding import inclusive, zeroIndexed
from mapFolding._e import (
	Leaf, LeafOrPileRangeOfLeaves, PermutationSpace, Pile, PileRangeOfLeaves, PilesWithPileRangeOfLeaves, PinnedLeaves)
from mapFolding._e.filters import extractPinnedLeaves, thisIsALeaf, thisIsAPileRangeOfLeaves
from more_itertools import iter_index
from operator import add, mul
from typing import Any

#======== Group-by functions ================================================

def bifurcatePermutationSpace(permutationSpace: PermutationSpace) -> tuple[PinnedLeaves, PilesWithPileRangeOfLeaves]:
	"""Separate `permutationSpace` into two dictionaries: one of `pile: leaf` and one of `pile: pileRangeOfLeaves`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary of `pile: leaf` and `pile: pileRangeOfLeaves`.

	Returns
	-------
	leavesPinned, pilesWithPileRangeOfLeaves : tuple[PinnedLeaves, PilesWithPileRangeOfLeaves]
	"""
	leavesPinned: PinnedLeaves = extractPinnedLeaves(permutationSpace)
	return (leavesPinned, dissociatePiles(permutationSpace, *DOTkeys(leavesPinned))) # pyright: ignore[reportReturnType]  # ty:ignore[invalid-return-type]

#======== Disaggregation and deconstruction functions ================================================

def DOTitems[文件, 文义](dictionary: Mapping[文件, 文义], /) -> Iterator[tuple[文件, 文义]]:
	"""Analogous to `dict.items()`: create an `Iterator` with the "items" from `dictionary`.

	Parameters
	----------
	dictionary : Mapping[文件, 文义]
		Source mapping.

	Returns
	-------
	aRiverOfItems : Iterator[tuple[文件, 文义]]
		`Iterator` of items from `dictionary`.
	"""
	return iter(dictionary.items())

def DOTkeys[个](dictionary: Mapping[个, Any], /) -> Iterator[个]:
	"""Analogous to `dict.keys()`: create an `Iterator` with the "keys" from `dictionary`.

	Parameters
	----------
	dictionary : Mapping[个, Any]
		Source mapping.

	Returns
	-------
	aRiverOfKeys : Iterator[个]
		`Iterator` of keys from `dictionary`.
	"""
	return iter(dictionary.keys())

def DOTvalues[个](dictionary: Mapping[Any, 个], /) -> Iterator[个]:
	"""Analogous to `dict.values()`: create an `Iterator` with the "values" from `dictionary`.

	Parameters
	----------
	dictionary : Mapping[Any, 个]
		Source mapping.

	Returns
	-------
	aRiverOfValues : Iterator[个]
		`Iterator` of values from `dictionary`.
	"""
	return iter(dictionary.values())

def getIteratorOfLeaves(pileRangeOfLeaves: PileRangeOfLeaves) -> Iterator[Leaf]:
	"""Convert `pileRangeOfLeaves` to an `Iterator` of `type` `int` `leaf`.

	Parameters
	----------
	pileRangeOfLeaves : PileRangeOfLeaves
		An integer with one bit for each `leaf` in `leavesTotal`, plus an extra bit that means "I'm a `pileRangeOfLeaves` not a `leaf`.

	Returns
	-------
	iteratorOfLeaves : Iterator[int]
		An `Iterator` with one `int` for each `leaf` in `pileRangeOfLeaves`.

	See Also
	--------
	https://gmpy2.readthedocs.io/en/latest/advmpz.html#gmpy2.xmpz.iter_set
	"""
	iteratorOfLeaves: xmpz = xmpz(pileRangeOfLeaves)
	iteratorOfLeaves[-1] = 0
	return iteratorOfLeaves.iter_set()

#======== `PileRangeOfLeaves` functions ================================================

# TODO Should `PermutationSpace` be a subclass of `dict` so I can add methods? NOTE I REFUSE TO BE AN OBJECT-ORIENTED
# PROGRAMMER!!! But, I'll use some OOP if it makes sense. I think collections has some my-first-dict-subclass functions.
def DOTgetPileIfLeaf(permutationSpace: PermutationSpace, pile: Pile, default: Leaf | None = None) -> Leaf | None:
	"""Like `permutationSpace.get(pile)`, but only return a `leaf` or `default`."""
	ImaLeaf: LeafOrPileRangeOfLeaves | None = permutationSpace.get(pile)
	if thisIsALeaf(ImaLeaf):
		return ImaLeaf
	return default

def DOTgetPileIfPileRangeOfLeaves(permutationSpace: PermutationSpace, pile: Pile, default: PileRangeOfLeaves | None = None) -> PileRangeOfLeaves | None:
	"""You can read `permutationSpace[pile]` only when `permutationSpace[pile]` is a `PileRangeOfLeaves`.

	This function is a typed analogue of `dict.get`. The function returns a `PileRangeOfLeaves` when `permutationSpace[pile]` is a
	`PileRangeOfLeaves`, and the function returns `default` when `permutationSpace[pile]` is a `Leaf` or `None`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary that maps each `Pile` to either a pinned `Leaf` or a `PileRangeOfLeaves` domain.
	pile : Pile
		`Pile` key to look up.
	default : PileRangeOfLeaves | None = None
		Value to return when `permutationSpace[pile]` is not a `PileRangeOfLeaves`.

	Returns
	-------
	pileRangeOfLeavesOrNone : PileRangeOfLeaves | None
		`PileRangeOfLeaves` value from `permutationSpace[pile]`, or `default`.

	Examples
	--------
	The function is used to retrieve a domain bitset with a fallback mask.

		pileRangeOfLeaves: PileRangeOfLeaves = raiseIfNone(DOTgetPileIfPileRangeOfLeaves(permutationSpace, domain[index], default=bit_mask(len(permutationSpace))))

	References
	----------
	[1] mapFolding._e.filters.thisIsAPileRangeOfLeaves
		Internal package reference
	[2] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[3] hunterMakesPy - Context7
		https://context7.com/hunterhogan/huntermakespy

	"""
	ImaPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None = permutationSpace.get(pile)
	if thisIsAPileRangeOfLeaves(ImaPileRangeOfLeaves):
		return ImaPileRangeOfLeaves
	return default

# TODO semiotics `getAntiPileRangeOfLeaves`, Improve.
def getAntiPileRangeOfLeaves(leavesTotal: int, leaves: Iterable[Leaf]) -> PileRangeOfLeaves:
	"""You can build a complement `PileRangeOfLeaves` by clearing each `Leaf` bit in `leaves`.

	The returned `PileRangeOfLeaves` contains a bit for every `Leaf` in `range(leavesTotal)` except each `Leaf` in `leaves`.
	The returned `PileRangeOfLeaves` also preserves the sentinel bit that indicates the value is a `PileRangeOfLeaves`.

	Parameters
	----------
	leavesTotal : int
		Total number of leaves in the map.
	leaves : Iterable[Leaf]
		Iterable of `Leaf` indices to exclude from the returned `PileRangeOfLeaves`.

	Returns
	-------
	antiPileRangeOfLeaves : PileRangeOfLeaves
		`PileRangeOfLeaves` bitset containing each allowed `Leaf` plus the `PileRangeOfLeaves` sentinel bit.

	Examples
	--------
	The function is used to start from the full domain.

		antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, frozenset())

	The function is used to exclude every `Leaf` not in a crease relation.

		antiPileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease))

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding.inclusive
		Internal package reference

	"""
	return reduce(bit_clear, leaves, bit_mask(leavesTotal + inclusive))

def getPileRangeOfLeaves(leavesTotal: int, leaves: Iterable[Leaf]) -> PileRangeOfLeaves:
	"""You can build a `PileRangeOfLeaves` by setting each `Leaf` bit in `leaves`.

	The returned `PileRangeOfLeaves` contains the sentinel bit that indicates the value is a `PileRangeOfLeaves`. The returned
	`PileRangeOfLeaves` also contains a bit for each `Leaf` in `leaves`.

	Parameters
	----------
	leavesTotal : int
		Total number of leaves in the map.
	leaves : Iterable[Leaf]
		Iterable of `Leaf` indices to include in the returned `PileRangeOfLeaves`.

	Returns
	-------
	pileRangeOfLeaves : PileRangeOfLeaves
		`PileRangeOfLeaves` bitset containing each `Leaf` in `leaves` plus the `PileRangeOfLeaves` sentinel bit.

	Examples
	--------
	The function is used to create a domain bitset before normalizing with `JeanValjean`.

		permutationSpace2上nDomainDefaults: PermutationSpace = {pile: raiseIfNone(JeanValjean(getPileRangeOfLeaves(state.leavesTotal, pileRangeOfLeaves)))
											for pile, pileRangeOfLeaves in getDictionaryPileRanges(state).items()}

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding._e._beDRY.JeanValjean
		Internal package reference

	"""
	return reduce(bit_set, leaves, bit_set(0, leavesTotal))

def JeanValjean(p24601: PileRangeOfLeaves, /) -> LeafOrPileRangeOfLeaves | None:
	"""You can normalize a `PileRangeOfLeaves` into a `Leaf` or `None` when the range is degenerate.

	When `p24601` is a `PileRangeOfLeaves`, `p24601` contains one sentinel bit that indicates the value is a `PileRangeOfLeaves`.
	This function interprets the total set-bit count as a compact encoding of domain cardinality.

	- When `p24601.bit_count() == 1`, `p24601` is an empty domain. The only set bit is the sentinel bit, so the function returns `None`.
	- When `p24601.bit_count() == 2`, `p24601` contains exactly one `Leaf` plus the sentinel bit. The function converts the range to a
		`Leaf` by returning `raiseIfNone(p24601.bit_scan1())`.
	- Otherwise, the function returns `p24601` unchanged.

	Parameters
	----------
	p24601 : PileRangeOfLeaves
		Candidate `PileRangeOfLeaves` value.

	Returns
	-------
	leafOrPileRangeOfLeavesOrNone : LeafOrPileRangeOfLeaves | None
		A `Leaf` when `p24601` encodes exactly one leaf, `None` when `p24601` encodes an empty domain, or `p24601` otherwise.

	Examples
	--------
	The function is used to normalize a masked domain.

		if (ImaLeafOrPileRangeOfLeavesNotAWalrusSubscript := JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, pileRangeOfLeaves))) is None:
			return {}

	The function is used to normalize per-pile domains into pinned leaves when possible.

		permutationSpace2上nDomainDefaults: PermutationSpace = {pile: raiseIfNone(JeanValjean(getPileRangeOfLeaves(state.leavesTotal, pileRangeOfLeaves)))
											for pile, pileRangeOfLeaves in getDictionaryPileRanges(state).items()}

	References
	----------
	[1] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	[2] mapFolding._e.filters.thisIsAPileRangeOfLeaves
		Internal package reference
	[3] hunterMakesPy - Context7
		https://context7.com/hunterhogan/huntermakespy

	"""
	whoAmI: LeafOrPileRangeOfLeaves | None = p24601
	if thisIsAPileRangeOfLeaves(p24601):
		if p24601.bit_count() == 1:
			whoAmI = None
		elif p24601.bit_count() == 2:
			whoAmI = raiseIfNone(p24601.bit_scan1())
	return whoAmI

@syntacticCurry
def pileRangeOfLeavesAND(pileRangeOfLeavesDISPOSABLE: PileRangeOfLeaves, pileRangeOfLeaves: PileRangeOfLeaves) -> PileRangeOfLeaves:
	"""Modify `pileRangeOfLeaves` by bitwise AND with `pileRangeOfLeavesDISPOSABLE`.

	Important
	---------
	The order of the parameters is likely the opposite of what you expect. This is to facilitate currying.
	"""
	return pileRangeOfLeaves & pileRangeOfLeavesDISPOSABLE

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
		Internal package reference

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
		Internal package reference
	[4] mapFolding._e.dataBaskets.EliminationState
		Internal package reference

	"""
	return tuple(accumulate(getProductsOfDimensions(mapShape), add, initial=0))

def getSumsOfProductsOfDimensionsNearest首(productsOfDimensions: tuple[int, ...], dimensionsTotal: int | None = None, dimensionFrom首: int | None = None) -> tuple[int, ...]:
	"""Get a useful list of numbers.

	This list of numbers is useful because I am using integers as a proxy for Cartesian coordinates in multidimensional space--and
	because I am trying to abstract the coordinates whether I am enumerating from the origin (0, 0, ..., 0) as represented by the
	integer 0 or from the "anti-origin", which is represented by an integer: but that integer varies based on the mapShape.

	By using products of dimensions and sums of products of dimensions, I can use the integer-as-coordinate by referencing its
	relative location in the products and sums of products of dimensions.

	`sumsOfProductsOfDimensionsNearest首` is yet another perspective on these abstractions. Instead of ordering the products in
	ascending order, I order them descending. Then I sum the descending-ordered products.

	(At least I think that is what I am doing. 2025 December 29.)

	`dimensionFrom首` almost certainly needs a better identifier. The purpose of the parameter is to define the list of products
	of which I want the sums.

	"""
	if dimensionsTotal is None:
		dimensionsTotal = len(productsOfDimensions) - 1

	if dimensionFrom首 is None:
		dimensionFrom首 = dimensionsTotal

	productsOfDimensionsTruncator: int = dimensionFrom首 - (dimensionsTotal + zeroIndexed)

	productsOfDimensionsFrom首: tuple[int, ...] = productsOfDimensions[0:productsOfDimensionsTruncator][::-1]

	sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = tuple(
						sum(productsOfDimensionsFrom首[0:aProduct], start=0)
							for aProduct in range(len(productsOfDimensionsFrom首) + inclusive)
	)
	return sumsOfProductsOfDimensionsNearest首

def reverseLookup[文件, 文义](dictionary: dict[文件, 文义], keyValue: 文义) -> 文件 | None:
	"""Return the key in `dictionary` that corresponds to `keyValue`.

	- I assume all `dictionary.values()` are distinct. If multiple keys contain `keyValue`, the returned key is not predictable.
	- I return `None` if no key maps to `keyValue`, but it is not an efficient way to check for membership.
	- If you *know* a value will be returned, consider combining with `hunterMakesPy.raiseIfNone`.
	"""
	for key, value in dictionary.items():
		if value == keyValue:
			return key
	return None

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
		Internal package reference

	"""
	return filter(lambda indices: 1 < len(indices), map(tuple, map(partial(iter_index, mapShape), unique(mapShape))))

# NOTE This 2^n-dimensional function is in this module to avoid `import` problems.
def mapShapeIs2上nDimensions(mapShape: tuple[int, ...], *, youMustBeDimensionsTallToPinThis: int = 3) -> bool:
	"""You can test whether `mapShape` is a $2^n$-dimensional map with a configurable minimum dimension count.

	This predicate is used as a flow guard for algorithms and pinning rules that only apply to the `mapShape == (2,) * n` special
	case. The predicate returns `True` only when `len(mapShape) >= youMustBeDimensionsTallToPinThis` and each `dimensionLength` in
	`mapShape` equals `2`.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Map shape as a tuple of dimension lengths.
	youMustBeDimensionsTallToPinThis : int = 3
		Minimum number of dimensions required before treating a $2^n$-dimensional special case as eligible.

	Returns
	-------
	is2上nDimensions : bool
		`True` when `mapShape` is a $2^n$-dimensional map with the required minimum dimension count.

	Examples
	--------
	The predicate is used to gate pinning logic.

		if not mapShapeIs2上nDimensions(state.mapShape):
			return state

	The predicate is used to gate deeper special cases.

		if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=5):
			return state

	References
	----------
	[1] mapFolding._e.pin2上nDimensions.pinPilesAtEnds
		Internal package reference
	[2] mapFolding._e.dataDynamic.addPileRangesOfLeaves
		Internal package reference

	"""
	return (youMustBeDimensionsTallToPinThis <= len(mapShape)) and all(dimensionLength == 2 for dimensionLength in mapShape)
