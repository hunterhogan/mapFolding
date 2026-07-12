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

from humpy_cytoolz import curry as syntacticCurry
from mapFolding._e.theTypes import Leaf, LeafOptions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
def leafPinned吗(leavesPinned: PinnedLeaves, leaf: Leaf) -> bool:
	"""Return `True` if `leaf` is pinned in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PinnedLeaves
		Pinned `Leaf` by `Pile`.
	leaf : Leaf
		`Leaf` index.

	Returns
	-------
	leafIsPinned : bool
		`True` if `leavesPinned` includes `leaf`.
	"""
	return leaf in leavesPinned.values()

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

def isLeaf吗(leafSpace: LeafSpace | None) -> TypeIs[Leaf]:
	"""Identify and narrow a `LeafSpace` value to `Leaf`.

	Use this function when control flow already has a `LeafSpace` value and needs to distinguish
	the Python types in `LeafSpace`. A `True` result narrows `leafSpace` to `Leaf`, but does not
	validate that the integer is a valid `Leaf` index. Use `PermutationSpace.pilePinned吗` when
	the logic instead asks about the assignment state of a `Pile`.

	Parameters
	----------
	leafSpace : LeafSpace | None
		`Leaf`, `LeafOptions`, or `None` to inspect.

	Returns
	-------
	leafSpaceIsLeaf : TypeIs[Leaf]
		`True` if `leafSpace` is an instance of `Leaf` and the positive branch can treat
		`leafSpace` as `Leaf`.

	See Also
	--------
	`isLeafOptions吗`
		Narrow a `LeafSpace` value to `LeafOptions`.
	`mapFolding._e.dataBaskets.PermutationSpace.pilePinned吗`
		Determine whether a `Pile` already has a pinned `Leaf`.
	`mapFolding._e.dataBaskets.PermutationSpace.pileUndetermined吗`
		Determine whether a `Pile` still requires a `Leaf` assignment.
	"""
	return isinstance(leafSpace, Leaf)

def isLeafOptions吗(leafSpace: LeafSpace | None) -> TypeIs[LeafOptions]:
	"""Identify and narrow a `LeafSpace` value to `LeafOptions`.

	Use this function when control flow already has a `LeafSpace` value and needs to distinguish
	the Python types in `LeafSpace`. A `True` result narrows `leafSpace` to the bitset type
	`LeafOptions`. Use `PermutationSpace.pileUndetermined吗` when the logic instead asks whether
	a `Pile` still requires a `Leaf` assignment.

	Parameters
	----------
	leafSpace : LeafSpace | None
		`Leaf`, `LeafOptions`, or `None` to inspect.

	Returns
	-------
	leafSpaceIsLeafOptions : TypeIs[LeafOptions]
		`True` if `leafSpace` is an instance of `LeafOptions` and the positive branch can treat
		`leafSpace` as `LeafOptions`.

	See Also
	--------
	`isLeaf吗`
		Narrow a `LeafSpace` value to `Leaf`.
	`mapFolding._e.dataBaskets.PermutationSpace.pileUndetermined吗`
		Determine whether a `Pile` still requires a `Leaf` assignment.
	`mapFolding._e.dataBaskets.PermutationSpace.pilePinned吗`
		Determine whether a `Pile` already has a pinned `Leaf`.
	"""
	return isinstance(leafSpace, LeafOptions)
