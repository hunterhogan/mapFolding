"""Avoid putting functions in here that only work on 2^n-dimensional maps."""
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from cytoolz.functoolz import curry as syntacticCurry
from cytoolz.itertoolz import unique
from functools import partial, reduce
from gmpy2 import bit_clear, bit_mask, bit_set, mpz, xmpz
from hunterMakesPy import Ordinals, raiseIfNone
from itertools import accumulate
from mapFolding import inclusive, zeroIndexed
from mapFolding._e import (
	Leaf, LeafOrPileRangeOfLeaves, PermutationSpace, Pile, PileRangeOfLeaves, PilesWithPileRangeOfLeaves, PinnedLeaves, 零)
from more_itertools import all_unique as allUnique吗, always_reversible, consecutive_groups, extract, iter_index
from operator import add, mul
from typing import Any, overload, TypeGuard

#======== Boolean filters ================================================

@syntacticCurry
def between[小于: Ordinals](floor: 小于, ceiling: 小于, comparand: 小于) -> bool:
	"""Inclusive `floor <= comparand <= ceiling`."""
	return floor <= comparand <= ceiling

def consecutive(flatContainer: Iterable[int]) -> bool:
	"""The integers in the `flatContainer` are consecutive, either ascending or descending."""
	return ((len(list(next(consecutive_groups(flatContainer)))) == len(list(flatContainer)))
	or (len(list(next(consecutive_groups(always_reversible(flatContainer))))) == len(list(flatContainer))))

def hasDuplicates(flatContainer: Iterable[Any]) -> bool:
	return not allUnique吗(flatContainer)

@syntacticCurry
def leafIsInPileRange(leaf: Leaf, pileRangeOfLeaves: PileRangeOfLeaves) -> bool:
	return pileRangeOfLeaves.bit_test(leaf)

@syntacticCurry
def leafIsNotPinned(permutationSpace: PermutationSpace, leaf: Leaf) -> bool:
	"""Return True if `leaf` is not presently pinned in `permutationSpace`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` index.

	Returns
	-------
	leafIsNotPinned : bool
		True if the mapping does not include `leaf`.
	"""
	return leaf not in permutationSpace.values()

@overload
def leafIsPinned(permutationSpace: PermutationSpace, leaf: Leaf) -> bool:...
@overload
def leafIsPinned(permutationSpace: PinnedLeaves, leaf: Leaf) -> bool:...
@syntacticCurry
def leafIsPinned(permutationSpace: PermutationSpace | PinnedLeaves, leaf: Leaf) -> bool:
	"""Return True if `leaf` is pinned in `permutationSpace`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` index.

	Returns
	-------
	leafIsPinned : bool
		True if the mapping includes `leaf`.
	"""
	return leaf in permutationSpace.values()

@syntacticCurry
def mappingHasKey[文件: Hashable](lookup: Mapping[文件, Any], key: 文件) -> bool:
	"""Return `True` if `key` is in `lookup`."""
	return key in lookup

def notLeafOriginOrLeaf零(leaf: Leaf) -> bool:
	return 零 < leaf

@syntacticCurry
def notPileLast(pileLast: Pile, pile: Pile) -> bool:
	"""Return True if `pile` is not the last pile.

	Parameters
	----------
	pileLast : int
		Index of the last pile.
	pile : int
		`pile` index.

	Returns
	-------
	notPileLast : bool
		True if `pile` is not equal to `pileLast`.
	"""
	return pileLast != pile

@syntacticCurry
def pileIsNotOpen(permutationSpace: PermutationSpace, pile: Pile) -> bool:
	"""Return True if `pile` is not presently pinned in `permutationSpace`.

	Do you want to know if the pile is open or do you really want to know the Python `type` of the value at that key?

	Parameters
	----------
	permutationSpace : PermutationSpace
		Partial folding mapping from pile -> leaf.
	pile : int
		`pile` index.

	Returns
	-------
	pileIsOpen : bool
		True if either `pile` is not a key in `permutationSpace` or `permutationSpace[pile]` is a `PileRangeOfLeaves`.

	See Also
	--------
	thisIsALeaf, thisIsAPileRangeOfLeaves
	"""
	return thisIsALeaf(permutationSpace.get(pile))

@syntacticCurry
def pileIsOpen(permutationSpace: PermutationSpace, pile: Pile) -> bool:
	"""Return True if `pile` is not presently pinned in `permutationSpace`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Partial folding mapping from pile -> leaf.
	pile : int
		`pile` index.

	Returns
	-------
	pileIsOpen : bool
		True if either `pile` is not a key in `permutationSpace` or `permutationSpace[pile]` is a `PileRangeOfLeaves`.
	"""
	return not thisIsALeaf(permutationSpace.get(pile))

@syntacticCurry
def thisHasThat[个](this: Iterable[个], that: 个) -> bool:
	return that in this

def thisIsALeaf(leafOrPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None) -> TypeGuard[Leaf]:
	"""Return True if `leafOrPileRangeOfLeaves` is a `leaf`.

	Parameters
	----------
	leafOrPileRangeOfLeaves : LeafOrPileRangeOfLeaves | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	intIsProbablyALeaf : TypeGuard[int]
		Technically, we only know the type is `int`.
	"""
	return (leafOrPileRangeOfLeaves is not None) and isinstance(leafOrPileRangeOfLeaves, int)

def thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None) -> TypeGuard[PileRangeOfLeaves]:
	"""Return True if `leafOrPileRangeOfLeaves` is a pile's range of leaves.

	Parameters
	----------
	leafOrPileRangeOfLeaves : LeafOrPileRangeOfLeaves | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	youHaveAPileRange : TypeGuard[PileRangeOfLeaves]
		Congrats, you have a pile range!
	"""
	return (leafOrPileRangeOfLeaves is not None) and isinstance(leafOrPileRangeOfLeaves, mpz)

#======== `LeafOrPileRangeOfLeaves` stuff ================================================
# https://gmpy2.readthedocs.io/en/latest/mpz.html

# TODO Should `PermutationSpace` be a subclass of `dict` so I can add methods? NOTE I REFUSE TO BE AN OBJECT-ORIENTED
# PROGRAMMER!!! But, I'll use some OOP if it makes sense. I think collections has some my-first-dict-subclass functions.
def DOTgetPileIfLeaf(permutationSpace: PermutationSpace, pile: Pile, default: Leaf | None = None) -> Leaf | None:
	"""Like `permutationSpace.get(pile)`, but only return a `leaf` or `default`."""
	ImaLeaf: LeafOrPileRangeOfLeaves | None = permutationSpace.get(pile)
	if thisIsALeaf(ImaLeaf):
		return ImaLeaf
	return default

def DOTgetPileIfPileRangeOfLeaves(permutationSpace: PermutationSpace, pile: Pile, default: PileRangeOfLeaves | None = None) -> PileRangeOfLeaves | None:
	ImaPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None = permutationSpace.get(pile)
	if thisIsAPileRangeOfLeaves(ImaPileRangeOfLeaves):
		return ImaPileRangeOfLeaves
	return default

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
	https://gmpy2.readthedocs.io/en/latest/advmpz.html
	"""
	iteratorOfLeaves = xmpz(pileRangeOfLeaves)
	iteratorOfLeaves[-1] = 0
	return iteratorOfLeaves.iter_set()

# TODO Improve semiotics of identifier `getAntiPileRangeOfLeaves`.
def getAntiPileRangeOfLeaves(leavesTotal: int, leaves: Iterable[Leaf]) -> PileRangeOfLeaves:
	return reduce(bit_clear, leaves, bit_mask(leavesTotal + inclusive))

def getPileRangeOfLeaves(leavesTotal: int, leaves: Iterable[Leaf]) -> PileRangeOfLeaves:
	return reduce(bit_set, leaves, bit_set(0, leavesTotal))

def extractPinnedLeaves(permutationSpace: PermutationSpace) -> PinnedLeaves:
	"""Create a dictionary *sorted* by `pile` of only `pile: leaf` without `pile: pileRangeOfLeaves`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary of `pile: leaf` and `pile: pileRangeOfLeaves`.

	Returns
	-------
	dictionaryOfPileLeaf : dict[int, int]
		Dictionary of `pile` with pinned `leaf`, if a `leaf` is pinned at `pile`.
	"""
	return {pile: leafOrPileRangeOfLeaves for pile, leafOrPileRangeOfLeaves in sorted(permutationSpace.items()) if thisIsALeaf(leafOrPileRangeOfLeaves)}

def extractPilesWithPileRangeOfLeaves(permutationSpace: PermutationSpace) -> PilesWithPileRangeOfLeaves:
	"""Return a dictionary of all pile-ranges of leaves in `permutationSpace`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary of `pile: leaf` and `pile: pileRangeOfLeaves`.

	Returns
	-------
	pilesWithPileRangeOfLeaves : dict[int, PileRangeOfLeaves]
		Dictionary of `pile: pileRangeOfLeaves`, if a `pileRangeOfLeaves` is defined at `pile`.
	"""
	return {pile: leafOrPileRangeOfLeaves for pile, leafOrPileRangeOfLeaves in permutationSpace.items() if thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves)}

# TODO efficient implementation of Z0Z_bifurcatePermutationSpace
def Z0Z_bifurcatePermutationSpace(permutationSpace: PermutationSpace) -> tuple[PinnedLeaves, PilesWithPileRangeOfLeaves]:
	"""Separate `permutationSpace` into two dictionaries: one of `pile: leaf` and one of `pile: pileRangeOfLeaves`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary of `pile: leaf` and `pile: pileRangeOfLeaves`.

	Returns
	-------
	dictionaryOfPileLeaf, pilesWithPileRangeOfLeaves : tuple[dict[int, int], dict[int, PileRangeOfLeaves]]
	"""
	permutationSpace = {}
	del permutationSpace
	yoMama: tuple[PinnedLeaves, PilesWithPileRangeOfLeaves] = ({}, {})
	return yoMama

@syntacticCurry
def pileRangeOfLeavesAND(pileRangeOfLeavesDISPOSABLE: PileRangeOfLeaves, pileRangeOfLeaves: PileRangeOfLeaves) -> PileRangeOfLeaves:
	"""Modify `pileRangeOfLeaves` by bitwise AND with `pileRangeOfLeavesDISPOSABLE`.

	Important
	---------
	The order of the parameters is likely the opposite of what you expect. This is to facilitate currying.
	"""
	return pileRangeOfLeaves & pileRangeOfLeavesDISPOSABLE

# TODO docstring that includes notes from the comments.
def JeanValjean(p24601: PileRangeOfLeaves, /) -> LeafOrPileRangeOfLeaves | None:
	whoAmI: LeafOrPileRangeOfLeaves | None = p24601
	if thisIsAPileRangeOfLeaves(p24601):
		if p24601.bit_count() == 1:
			# The pile-range of leaves is null; the only "set bit" is the bit that means "I am a pileRangeOfLeaves."
			whoAmI = None
		elif p24601.bit_count() == 2:
			# Only one `leaf` is in the pile-range of leaves, so convert it to a `type` `int`.
			whoAmI = raiseIfNone(p24601.bit_scan1())
	return whoAmI

#======== Workbench functions ===============================================

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

def exclude[个](flatContainer: Sequence[个], indices: Iterable[int]) -> Iterator[个]:
	"""Yield items from `flatContainer` whose positions are not in `indices`."""
	lengthIterable: int = len(flatContainer)
	def normalizeIndex(index: int) -> int:
		if index < 0:
			index = (index + lengthIterable) % lengthIterable
		return index
	indicesInclude: list[int] = sorted(set(range(lengthIterable)).difference(map(normalizeIndex, indices)))
	return extract(flatContainer, indicesInclude)

def reverseLookup[文件, 文义](dictionary: dict[文件, 文义], keyValue: 文义) -> 文件 | None:
	"""Return the key in `dictionary` that corresponds to `keyValue`.

	Prototype.

	- I assume all `dictionary.values()` are distinct. If multiple keys contain `keyValue`, the returned key is not predictable.
	- I removed `sorted()` for speed.
	- I return `None` if no key maps to `keyValue`, but it is not an efficient way to check for membership.
	"""
	for key, value in dictionary.items():
		if value == keyValue:
			return key
	return None

# TODO docstring
def getProductsOfDimensions(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	return tuple(accumulate(mapShape, mul, initial=1))

# TODO docstring
def getSumsOfProductsOfDimensions(mapShape: tuple[int, ...]) -> tuple[int, ...]:
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

#======== Flow control ================================================

# TODO docstring
def mapShapeIs2上nDimensions(mapShape: tuple[int, ...], *, youMustBeDimensionsTallToPinThis: int = 3) -> bool:
	return (youMustBeDimensionsTallToPinThis <= len(mapShape)) and all(dimensionLength == 2 for dimensionLength in mapShape)

# TODO docstring
def indicesMapShapeDimensionLengthsAreEqual(mapShape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
	return filter(lambda indices: 1 < len(indices), map(tuple, map(partial(iter_index, mapShape), unique(mapShape))))
