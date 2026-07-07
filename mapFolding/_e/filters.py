# ruff: noqa: DOC201
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
		You can test whether a `leaf` is present in `leafOptions`.
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
		You can narrow `leafSpace` to a `Leaf`.
	thisIsALeafOptions
		You can narrow `leafSpace` to a `LeafOptions`.

Filter functions
	exclude
		You can yield items from `flatContainer` whose positions are not in `indices`.
	extractPinnedLeaves
		You can extract only `pile: leaf` mappings from a `PermutationSpace`.
	extractUndeterminedPiles
		You can extract only `pile: leafOptions` mappings from a `PermutationSpace`.

References
----------
[1] Built-in Functions - `filter` (Python documentation)
	https://docs.python.org/3/library/functions.html#filter
[2] cytoolz - dicttoolz
	https://toolz.readthedocs.io/en/latest/api.html#module-toolz.dicttoolz
[3] more-itertools - API Reference
	https://more-itertools.readthedocs.io/en/stable/api.html

"""
from __future__ import annotations

from collections.abc import Sequence
from gmpy2 import mpz, sign
from humpy_cytoolz import curry as syntacticCurry, valfilter as filterLeaf
from hunterMakesPy import inclusive
from hunterMakesPy.parseParameters import intInnit
from mapFolding._e import DOTitems, 零
from more_itertools import always_reversible, consecutive_groups, extract
from typing import overload, TYPE_CHECKING
from operator import eq

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from hunterMakesPy import Ordinals
	from mapFolding._e.theTypes import Leaf, LeafOptions, LeafSpace, PermutationSpace, Pile, PinnedLeaves, UndeterminedPiles
	from typing import TypeIs

#======== Boolean antecedents ================================================

@syntacticCurry
def between吗[小于: Ordinals](floor: 小于, ceiling: 小于, comparand: 小于) -> bool:
	"""Inclusive `floor <= comparand <= ceiling`."""
	return floor <= comparand <= ceiling

# NOTE `个` typevar exists to help ty with static type checking. See https://github.com/astral-sh/ty/issues/2799.
def consecutive吗[个: Sequence[int]](flatContainer: 个) -> bool:
	"""Are the integers in `flatContainer` consecutive, either ascending or descending?"""
	ImaListOfInt: list[int] = intInnit(flatContainer, 'flatContainer', Sequence[int])
	qty = ImaListOfInt[-1] - ImaListOfInt[0]
	mustBeTrue = qty == len(ImaListOfInt)
	direction = sign(qty)
	rr = range(ImaListOfInt[0], ImaListOfInt[0] + qty, direction)
	jj = all(map(eq, ImaListOfInt, rr))
	ll = list(rr)
	bb = ll == ImaListOfInt
	return ((len(list(next(consecutive_groups(ImaListOfInt)))) == len(ImaListOfInt))
		or (len(list(next(consecutive_groups(always_reversible(ImaListOfInt))))) == len(ImaListOfInt)))

@syntacticCurry
def leafIsInPileRange(leaf: Leaf, leafOptions: LeafOptions) -> bool:
	"""Test whether `leaf` is present in `leafOptions`.

	You can use `leafIsInPileRange` in an `if` statement, or you can pass `leafIsInPileRange`
	as a predicate to a filtering utility described in the module docstring.

	Parameters
	----------
	leaf : Leaf
		`leaf` index.
	leafOptions : LeafOptions
		Bitset of `leaf` membership for a pile.

	Returns
	-------
	leafIsPresent : bool
		`True` if `leafOptions` contains `leaf`.

	References
	----------
	[1] gmpy2 - `mpz` type and methods
		https://gmpy2.readthedocs.io/en/latest/mpz.html

	"""
	return leafOptions.bit_test(leaf)

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
def leafIsPinned(permutationSpace: PermutationSpace, leaf: Leaf) -> bool: ...
@overload
def leafIsPinned(permutationSpace: PinnedLeaves, leaf: Leaf) -> bool: ...
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
		True if either `pile` is not a key in `permutationSpace` or `permutationSpace[pile]` is a `LeafOptions`.

	See Also
	--------
	thisIsALeaf, thisIsALeafOptions
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
		True if either `pile` is not a key in `permutationSpace` or `permutationSpace[pile]` is a `LeafOptions`.
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

@syntacticCurry
def thisNotHaveThat[个](this: Iterable[个], that: 个) -> bool:
	return not thisHasThat(this, that)

def thisIsALeaf(leafSpace: LeafSpace | None) -> TypeIs[Leaf]:
	"""Return True if `leafSpace` is a `leaf`.

	Parameters
	----------
	leafSpace : LeafSpace | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	intIsProbablyALeaf : TypeIs[int]
		Technically, we only know the type is `int`.
	"""
	return (leafSpace is not None) and isinstance(leafSpace, int)

def thisIsLeafOptions(leafSpace: LeafSpace | None) -> TypeIs[LeafOptions]:
	"""Return True if `leafSpace` is a pile's range of leaves.

	Parameters
	----------
	leafSpace : LeafSpace | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	youHaveAPileRange : TypeIs[LeafOptions]
		Congrats, you have a pile range!
	"""
	return (leafSpace is not None) and isinstance(leafSpace, mpz)

#======== Filtering functions ================================================

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
	"""Create a dictionary *sorted* by `pile` of only `pile: leaf` without `pile: leafOptions`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary of `pile: leaf` and `pile: leafOptions`.

	Returns
	-------
	dictionaryOfPileLeaf : dict[int, int]
		Dictionary of `pile` with pinned `leaf`, if a `leaf` is pinned at `pile`.
	"""
	return dict(sorted(DOTitems(filterLeaf(thisIsALeaf, permutationSpace))))

def extractUndeterminedPiles(permutationSpace: PermutationSpace) -> UndeterminedPiles:
	"""Return a dictionary of all pile-ranges of leaves in `permutationSpace`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary of `pile: leaf` and `pile: leafOptions`.

	Returns
	-------
	pilesUndetermined : dict[int, LeafOptions]
		Dictionary of `pile: leafOptions`, if a `leafOptions` is defined at `pile`.
	"""
	return filterLeaf(thisIsLeafOptions, permutationSpace)
