from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache, reduce
from gmpy2 import bit_clear, bit_mask, bit_set, mpz, xmpz
from hunterMakesPy import intInnit, raiseIfNone
from itertools import accumulate
from mapFolding import inclusive, zeroIndexed
from mapFolding._e import LeafOrPileRangeOfLeaves, PermutationSpace, PileRangeOfLeaves, 零
from more_itertools import all_unique, always_reversible, consecutive_groups, extract
from operator import add, getitem, mul
from typing import Any, Protocol, Self, TypeGuard

class _Ordinals(Protocol):
	"""Any Python `object` `type` that may be ordered before or after a comparable `object` `type` using a less-than-or-equal-to comparison."""

	def __le__(self, not_self_selfButSelfSelf_youKnow: Self, /) -> bool:
		"""Comparison by "***l***ess than or ***e***qual to"."""
		...

# ======= Boolean filters ================================================

@syntacticCurry
def between[小于: _Ordinals](floor: 小于, ceiling: 小于, comparand: 小于) -> bool:
	"""Inclusive `floor <= comparand <= ceiling`."""
	return floor <= comparand <= ceiling

def consecutive(flatContainer: Iterable[int]) -> bool:
	"""The integers in the `flatContainer` are consecutive, either ascending or descending."""
	return ((len(list(next(consecutive_groups(flatContainer)))) == len(list(flatContainer)))
	or (len(list(next(consecutive_groups(always_reversible(flatContainer))))) == len(list(flatContainer))))

def hasDuplicates(flatContainer: Iterable[Any]) -> bool:
	return not all_unique(flatContainer)

@syntacticCurry
def leafIsInPileRange(leaf: int, pileRangeOfLeaves: PileRangeOfLeaves) -> bool:
	return pileRangeOfLeaves.bit_test(leaf)

@syntacticCurry
def leafIsNotPinned(leavesPinned: PermutationSpace, leaf: int) -> bool:
	"""Return True if `leaf` is not presently pinned in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` index.

	Returns
	-------
	leafIsNotPinned : bool
		True if the mapping does not include `leaf`.
	"""
	return leaf not in leavesPinned.values()

@syntacticCurry
def leafIsPinned(leavesPinned: PermutationSpace, leaf: int) -> bool:
	"""Return True if `leaf` is pinned in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` index.

	Returns
	-------
	leafIsPinned : bool
		True if the mapping includes `leaf`.
	"""
	return leaf in leavesPinned.values()

@syntacticCurry
def mappingHasKey[文件: Hashable](lookup: Mapping[文件, Any], key: 文件) -> bool:
	"""Return `True` if `key` is in `lookup`."""
	return key in lookup

def notLeafOriginOrLeaf零(leaf: int) -> bool:
	return 零 < leaf

@syntacticCurry
def notPileLast(pileLast: int, pile: int) -> bool:
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
def pileIsNotOpen(leavesPinned: PermutationSpace, pile: int) -> bool:
	"""Return True if `pile` is not presently pinned in `leavesPinned`.

	Do you want to know if the pile is open or do you really want to know the Python `type` of the value at that key?

	Parameters
	----------
	leavesPinned : PermutationSpace
		Partial folding mapping from pile -> leaf.
	pile : int
		`pile` index.

	Returns
	-------
	pileIsOpen : bool
		True if either `pile` is not a key in `leavesPinned` or `leavesPinned[pile]` is a pile-range (`mpz`).

	See Also
	--------
	thisIsALeaf, thisIsAPileRangeOfLeaves
	"""
	return thisIsALeaf(leavesPinned.get(pile))

# TODO Consider if it is possible to return TypeGuard[mpz] (or TypeGuard[None]?) here.
@syntacticCurry
def pileIsOpen(leavesPinned: PermutationSpace, pile: int) -> bool:
	"""Return True if `pile` is not presently pinned in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Partial folding mapping from pile -> leaf.
	pile : int
		`pile` index.

	Returns
	-------
	pileIsOpen : bool
		True if either `pile` is not a key in `leavesPinned` or `leavesPinned[pile]` is a pile-range (`mpz`).
	"""
	return not thisIsALeaf(leavesPinned.get(pile))

def thisIsALeaf(leafOrPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None) -> TypeGuard[int]:
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

def thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None) -> TypeGuard[mpz]:
	"""Return True if `leafOrPileRangeOfLeaves` is a pile's range of leaves.

	Parameters
	----------
	leafOrPileRangeOfLeaves : LeafOrPileRangeOfLeaves | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	youHaveAPileRange : TypeGuard[mpz]
		Congrats, you have a pile range!
	"""
	return (leafOrPileRangeOfLeaves is not None) and isinstance(leafOrPileRangeOfLeaves, mpz)

# ======= `LeafOrPileRangeOfLeaves` stuff ================================================
# https://gmpy2.readthedocs.io/en/latest/advmpz.html

# TODO I have a vague memory of using overload/TypeGuard in a similar function: I think it was in the stub file for `networkx`. Check that out.
def DOTgetPileIfLeaf(leavesPinned: PermutationSpace, pile: int, default: int | None = None) -> int | None:
# TODO I have a bad feeling that I am going to make `PermutationSpace` a subclass of `dict` and add methods.
# NOTE I REFUSE TO BE AN OBJECT-ORIENTED PROGRAMMER!!! But, I'll use some OOP if it makes sense.
# I think collections has some my-first-dict-subclass functions.
	"""Like `leavesPinned.get(pile)`, but only return a `leaf` or `default`."""
	ImaLeaf = leavesPinned.get(pile)
	if thisIsALeaf(ImaLeaf):
		return ImaLeaf
	return default

def DOTgetPileIfPileRangeOfLeaves(leavesPinned: PermutationSpace, pile: int, default: PileRangeOfLeaves | None = None) -> PileRangeOfLeaves | None:
	ImaPileRangeOfLeaves = leavesPinned.get(pile)
	if thisIsAPileRangeOfLeaves(ImaPileRangeOfLeaves):
		return ImaPileRangeOfLeaves
	return default

def getIteratorOfLeaves(pileRangeOfLeaves: mpz) -> Iterator[int]:
	"""Convert `pileRangeOfLeaves` to an `Iterator` of `type` `int` `leaf`.

	Parameters
	----------
	pileRangeOfLeaves : mpz
		An integer with one bit for each `leaf` in `leavesTotal`, plus an extra bit that means "I'm a `pileRangeOfLeaves` not a `leaf`.

	Returns
	-------
	iteratorOfLeaves : Iterator[int]
		An `Iterator` with one `int` for each `leaf` in `pileRangeOfLeaves`.
	"""
	iteratorOfLeaves = xmpz(pileRangeOfLeaves)
	iteratorOfLeaves[-1] = 0
	return iteratorOfLeaves.iter_set()

def getAntiPileRangeOfLeaves(leavesTotal: int, leaves: Iterable[int]) -> mpz:
	return reduce(bit_clear, leaves, bit_mask(leavesTotal + inclusive))  # ty:ignore[invalid-return-type]

def getPileRangeOfLeaves(leavesTotal: int, leaves: Iterable[int]) -> mpz:
	return reduce(bit_set, leaves, bit_set(0, leavesTotal))  # ty:ignore[invalid-return-type]

def oopsAllLeaves(leavesPinned: PermutationSpace) -> dict[int, int]:
	"""Create a dictionary *sorted* by `pile` of only `pile: leaf` without `pile: pileRangeOfLeaves`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Dictionary of `pile: leaf` and `pile: pileRangeOfLeaves`.

	Returns
	-------
	dictionaryOfPileLeaf : dict[int, int]
		Dictionary of `pile` with pinned `leaf`, if a `leaf` is pinned at `pile`.
	"""
	return {pile: leafOrPileRangeOfLeaves for pile, leafOrPileRangeOfLeaves in sorted(leavesPinned.items()) if thisIsALeaf(leafOrPileRangeOfLeaves)}

def oopsAllPileRangesOfLeaves(leavesPinned: PermutationSpace) -> dict[int, PileRangeOfLeaves]:
	"""Return a dictionary of all pile-ranges of leaves in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Dictionary of `pile: leaf` and `pile: pileRangeOfLeaves`.

	Returns
	-------
	dictionaryOfPilePileRangeOfLeaves : dict[int, PileRangeOfLeaves]
		Dictionary of `pile: pileRangeOfLeaves`, if a `pileRangeOfLeaves` is defined at `pile`.
	"""
	return {pile: leafOrPileRangeOfLeaves for pile, leafOrPileRangeOfLeaves in leavesPinned.items() if thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves)}

@syntacticCurry
def pileRangeOfLeavesAND(pileRangeOfLeavesDISPOSABLE: mpz, pileRangeOfLeaves: mpz) -> mpz:
	"""Modify `pileRangeOfLeaves` by bitwise AND with `pileRangeOfLeavesDISPOSABLE`.

	Important
	---------
	The order of the parameters is likely the opposite of what you expect. This is to facilitate currying.
	"""
	return pileRangeOfLeaves & pileRangeOfLeavesDISPOSABLE

def Z0Z_JeanValjean(p24601: mpz) -> int | mpz | None:
	whoAmI: int | mpz | None = p24601
	if thisIsAPileRangeOfLeaves(p24601):
		if p24601.bit_count() == 1:
			# The pile-range of leaves is null; the only "set bit" is the bit that means "I am a pileRangeOfLeaves."
			whoAmI = None
		elif p24601.bit_count() == 2:
			# Only one `leaf` is in the pile-range of leaves, so convert it to a `type` `int`.
			whoAmI = raiseIfNone(p24601.bit_scan1())
	return whoAmI

# ======= Workbench functions ===============================================

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

@syntacticCurry
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

@cache
def Z0Z_invert(dimensionsTotal: int, integerNonnegative: int) -> int:
	anInteger: int = getitem(intInnit([integerNonnegative], 'integerNonnegative', type[int]), 0)  # ty:ignore[invalid-argument-type]
	if anInteger < 0:
		message: str = f"I received `{integerNonnegative = }`, but I need a value greater than or equal to 0."
		raise ValueError(message)
	return int(anInteger ^ bit_mask(dimensionsTotal))

def getProductsOfDimensions(mapShape: tuple[int, ...]) -> tuple[int, ...]:
	return tuple(accumulate(mapShape, mul, initial=1))

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

# ======= Flow control ================================================

def mapShapeIs2上nDimensions(mapShape: tuple[int, ...], *, youMustBeDimensionsTallToPinThis: int = 3) -> bool:
	return (youMustBeDimensionsTallToPinThis <= len(mapShape)) and all(dimensionLength == 2 for dimensionLength in mapShape)


