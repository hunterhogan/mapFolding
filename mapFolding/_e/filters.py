"""You can use this module to express boolean antecedents and apply antecedents as filters.

This module groups small boolean antecedents (predicates) that are convenient to reuse.
You can use each antecedent in a Python `if` statement, or you can pass an antecedent
as a predicate argument to a filtering utility such as `filter` [1],
`cytoolz.dicttoolz.keyfilter` [2], `cytoolz.dicttoolz.itemfilter` [2],
`cytoolz.dicttoolz.valfilter` [2], `more_itertools.filter_map` [3], or
`more_itertools.filterfalse` [3].

This module also provides a small number of filtering functions that are already
specialized to the map-folding data structures used by `_e` algorithms.

Contents
--------
Boolean antecedents
	between
		You can test whether `floor <= comparand <= ceiling`.
	consecutive
		You can test whether the integers in `flatContainer` are consecutive.
	hasDuplicates
		You can test whether `flatContainer` contains duplicate values.
	leafIsInPileRange
		You can test whether a `leaf` is present in `pileRangeOfLeaves`.
	leafIsNotPinned
		You can test whether a `leaf` is absent from `permutationSpace.values()`.
	leafIsPinned
		You can test whether a `leaf` is present in `permutationSpace.values()`.
	mappingHasKey
		You can test whether `key` is present in `lookup`.
	notLeafOriginOrLeaf零
		You can test whether `leaf` is greater than `零`.
	notPileLast
		You can test whether `pile` is not equal to `pileLast`.
	pileIsNotOpen
		You can test whether `permutationSpace[pile]` is a `Leaf`.
	pileIsOpen
		You can test whether `permutationSpace[pile]` is not a `Leaf`.
	thisHasThat
		You can test whether `that` is present in `this`.
	thisIsALeaf
		You can narrow `leafOrPileRangeOfLeaves` to a `Leaf`.
	thisIsAPileRangeOfLeaves
		You can narrow `leafOrPileRangeOfLeaves` to a `PileRangeOfLeaves`.

Filter functions
	exclude
		You can yield items from `flatContainer` whose positions are not in `indices`.
	extractPinnedLeaves
		You can extract only `pile: leaf` mappings from a `PermutationSpace`.
	extractPilesWithPileRangeOfLeaves
		You can extract only `pile: pileRangeOfLeaves` mappings from a `PermutationSpace`.

References
----------
[1] Built-in Functions - `filter` (Python documentation)
	https://docs.python.org/3/library/functions.html#filter
[2] cytoolz - dicttoolz
	https://toolz.readthedocs.io/en/latest/api.html#module-toolz.dicttoolz
[3] more-itertools - API Reference
	https://more-itertools.readthedocs.io/en/stable/api.html

"""
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from cytoolz.dicttoolz import valfilter as leafFilter
from cytoolz.functoolz import curry as syntacticCurry
from gmpy2 import mpz
from hunterMakesPy import Ordinals
from mapFolding._e import (
	Leaf, LeafOrPileRangeOfLeaves, PermutationSpace, Pile, PileRangeOfLeaves, PilesWithPileRangeOfLeaves, PinnedLeaves, 零)
from more_itertools import all_unique as allUnique吗, always_reversible, consecutive_groups, extract
from typing import Any, overload
from typing_extensions import TypeIs

#======== Boolean antecedents ================================================

@syntacticCurry
def between[小于: Ordinals](floor: 小于, ceiling: 小于, comparand: 小于) -> bool:
	"""Inclusive `floor <= comparand <= ceiling`."""
	return floor <= comparand <= ceiling

# TODO `consecutive` and `intInnit`?
# TODO `consecutive`: `raise` if not `int`?
# def consecutive[个: Iterable[int]](flatContainer: 个) -> bool:
def consecutive[个: Iterable[int]](flatContainer: 个) -> TypeIs[个]:
	"""The integers in the `flatContainer` are consecutive, either ascending or descending."""
	return (all(isinstance(item, int) for item in flatContainer)
	and ((len(list(next(consecutive_groups(flatContainer)))) == len(list(flatContainer)))
		or (len(list(next(consecutive_groups(always_reversible(flatContainer))))) == len(list(flatContainer)))))

def hasDuplicates(flatContainer: Iterable[Any]) -> bool:
	"""You can test whether `flatContainer` contains duplicate values.

	You can use `hasDuplicates` in an `if` statement, or you can pass `hasDuplicates` as a
	predicate to a filtering utility described in the module docstring.

	Parameters
	----------
	flatContainer : Iterable[Any]
		Iterable of values to test for duplicate values.

	Returns
	-------
	flatContainerHasDuplicates : bool
		`True` if `flatContainer` contains at least one duplicate value.

	References
	----------
	[1] more-itertools - `all_unique`
		https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.all_unique

	"""
	return not allUnique吗(flatContainer)

@syntacticCurry
def leafIsInPileRange(leaf: Leaf, pileRangeOfLeaves: PileRangeOfLeaves) -> bool:
	"""You can test whether `leaf` is present in `pileRangeOfLeaves`.

	You can use `leafIsInPileRange` in an `if` statement, or you can pass `leafIsInPileRange`
	as a predicate to a filtering utility described in the module docstring.

	Parameters
	----------
	leaf : Leaf
		`leaf` index.
	pileRangeOfLeaves : PileRangeOfLeaves
		Bitset of `leaf` membership for a pile.

	Returns
	-------
	leafIsPresent : bool
		`True` if `pileRangeOfLeaves` contains `leaf`.

	References
	----------
	[1] gmpy2 - `mpz` type and methods
		https://gmpy2.readthedocs.io/en/latest/mpz.html

	"""
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
def leafIsPinnedAtPile(permutationSpace: PermutationSpace, leaf: Leaf, pile: Pile) -> bool:
	"""Return `True` if `leaf` is presently pinned at `pile` in `permutationSpace`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` whose presence at `pile` is being checked.
	pile : int
		`pile` index.

	Returns
	-------
	leafIsPinnedAtPile : bool
		True if the mapping includes `pile: leaf`.
	"""
	return leaf == permutationSpace.get(pile)

@syntacticCurry
def mappingHasKey[文件: Hashable](lookup: Mapping[文件, Any], key: 文件) -> bool:
	"""Return `True` if `key` is in `lookup`."""
	return key in lookup

def notLeafOriginOrLeaf零(leaf: Leaf) -> bool:
	"""You can test whether `leaf` is greater than `零`.

	You can use `notLeafOriginOrLeaf零` in an `if` statement, or you can pass
	`notLeafOriginOrLeaf零` as a predicate to a filtering utility described in the module
	docstring.

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
		Internal package reference

	"""
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
def pileIsNotOpen(permutationSpace: PermutationSpace, pile: Pile) -> TypeIs[Leaf]:
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
	return thisIsALeaf(permutationSpace[pile])

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
	"""You can test whether `that` is present in `this`.

	You can use `thisHasThat` in an `if` statement, or you can pass `thisHasThat` as a
	predicate to a filtering utility described in the module docstring.

	Parameters
	----------
	this : Iterable[个]
		Iterable to search.
	that : 个
		Value to find.

	Returns
	-------
	thatIsPresent : bool
		`True` if `that in this`.

	References
	----------
	[1] `operator.contains` (Python documentation)
		https://docs.python.org/3/library/operator.html#operator.contains

	"""
	return that in this

def thisIsALeaf(leafOrPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None) -> TypeIs[Leaf]:
	"""Return True if `leafOrPileRangeOfLeaves` is a `leaf`.

	Parameters
	----------
	leafOrPileRangeOfLeaves : LeafOrPileRangeOfLeaves | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	intIsProbablyALeaf : TypeIs[int]
		Technically, we only know the type is `int`.
	"""
	return (leafOrPileRangeOfLeaves is not None) and isinstance(leafOrPileRangeOfLeaves, int)

def thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None) -> TypeIs[PileRangeOfLeaves]:
	"""Return True if `leafOrPileRangeOfLeaves` is a pile's range of leaves.

	Parameters
	----------
	leafOrPileRangeOfLeaves : LeafOrPileRangeOfLeaves | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	youHaveAPileRange : TypeIs[PileRangeOfLeaves]
		Congrats, you have a pile range!
	"""
	return (leafOrPileRangeOfLeaves is not None) and isinstance(leafOrPileRangeOfLeaves, mpz)

#======== Filter functions ================================================

def exclude[个](flatContainer: Sequence[个], indices: Iterable[int]) -> Iterator[个]:
	"""Yield items from `flatContainer` whose positions are not in `indices`."""
	lengthIterable: int = len(flatContainer)
	def normalizeIndex(index: int) -> int:
		if index < 0:
			index = (index + lengthIterable) % lengthIterable
		return index
	indicesInclude: list[int] = sorted(set(range(lengthIterable)).difference(map(normalizeIndex, indices)))
	return extract(flatContainer, indicesInclude)

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
	return dict(sorted(leafFilter(thisIsALeaf, permutationSpace).items()))

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
	return leafFilter(thisIsAPileRangeOfLeaves, permutationSpace)

