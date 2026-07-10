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
	leafIsInPileRange
		You can test whether a `leaf` is present in `leafOptions`.
	leafIsNotPinned
		You can test whether a `leaf` is absent from `permutationSpace.values()`.
	leafPinned吗
		You can test whether a `leaf` is present in `permutationSpace.values()`.
	notLeafOriginOrLeaf零
		You can test whether `leaf` is greater than `零`.
	notPileLast
		You can test whether `pile` is not equal to `pileLast`.
	pileIsNotOpen
		You can test whether `permutationSpace[pile]` is a `Leaf`.
	pileIsOpen
		You can test whether `permutationSpace[pile]` is not a `Leaf`.
	thisIsALeaf
		You can narrow `leafSpace` to a `Leaf`.
	thisIsALeafOptions
		You can narrow `leafSpace` to a `LeafOptions`.

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

from humpy_cytoolz import curry as syntacticCurry, valfilter as filterLeaf
from mapFolding._e import 零
from mapFolding._e.theTypes import Leaf, LeafOptions
from typing import overload, TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import PermutationSpace
	from mapFolding._e.theTypes import LeafSpace, Pile, PinnedLeaves
	from typing import TypeIs

#======== Boolean antecedents ================================================

@syntacticCurry
def leafInLeafOptions吗(leaf: Leaf, leafOptions: LeafOptions) -> bool:
	"""Test whether `leaf` is present in `leafOptions`.

	You can use `leafInLeafOptions吗` in an `if` statement, or you can pass `leafInLeafOptions吗`
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
def leafNotPinned吗(permutationSpace: PermutationSpace, leaf: Leaf) -> bool:
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
def leafPinned吗(permutationSpace: PermutationSpace, leaf: Leaf) -> bool: ...
@overload
def leafPinned吗(permutationSpace: PinnedLeaves, leaf: Leaf) -> bool: ...
@syntacticCurry
def leafPinned吗(permutationSpace: PermutationSpace | PinnedLeaves, leaf: Leaf) -> bool:
	"""Return True if `leaf` is pinned in `permutationSpace`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` index.

	Returns
	-------
	leafPinned吗 : bool
		True if the mapping includes `leaf`.
	"""
	return leaf in permutationSpace.values()

@syntacticCurry
def leafPinnedAtPile吗(permutationSpace: PermutationSpace, leaf: Leaf, pile: Pile) -> bool:
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
def pileNotOpen吗(permutationSpace: PermutationSpace, pile: Pile) -> bool:
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
	return isLeaf吗(permutationSpace[pile])

@syntacticCurry
def pileOpen吗(permutationSpace: PermutationSpace, pile: Pile) -> bool:
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
	return not isLeaf吗(permutationSpace[pile])

def isLeaf吗(leafSpace: LeafSpace | None) -> TypeIs[Leaf]:
	"""Return True if `leafSpace` is a `leaf`.

	Parameters
	----------
	leafSpace : LeafSpace | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	intIsProbablyALeaf : TypeIs[Leaf]
		Technically, we only know the type is `Leaf`.
	"""
	return isinstance(leafSpace, Leaf)

def isLeafOptions吗(leafSpace: LeafSpace | None) -> TypeIs[LeafOptions]:
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
	return isinstance(leafSpace, LeafOptions)
