"""Generalized pinning functions in the "Elimination" algorithm for any `mapShape`.

Functions for 2^n-dimensional maps must go in other modules.
"""
from __future__ import annotations

from functools import partial
from gmpy2 import bit_clear
from humpy_cytoolz import groupby as toolz_groupby
from hunterMakesPy import raiseIfNone
from mapFolding import _e
from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
from mapFolding._e.filters import isLeafOptions吗, isLeaf吗, leafInLeafOptions吗
from mapFolding._e.theTypes import LeafOptions, LeafSpace
from more_itertools import filter_map
from operator import methodcaller
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding._e.theTypes import Leaf, Pile

#======== Group by =======================

def segregateLeafPinnedAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile) -> tuple[list[PermutationSpace], list[PermutationSpace]]:
	"""Partition `listPermutationSpace` into (notPinned, isPinned) groups for `leaf` pinned at `pile`.

	Parameters
	----------
	listPermutationSpace : Iterable[PermutationSpace]
		Collection of partial folding dictionaries.
	leaf : int
		`leaf` to test.
	pile : int
		`pile` index.

	Returns
	-------
	segregatedLists : tuple[list[PermutationSpace], list[PermutationSpace]]
		First element: dictionaries where `leaf` is NOT pinned at `pile`.
		Second element: dictionaries where `leaf` IS pinned at `pile`.
	"""
	isPinned: Callable[[PermutationSpace], bool] = partial(PermutationSpace.leafPinnedAtPile吗, leaf=leaf, pile=pile)
	grouped: dict[bool, list[PermutationSpace]] = toolz_groupby(isPinned, listPermutationSpace)
	return (grouped.get(False, []), grouped.get(True, []))

#======== Bulk modifications =======================

def excludeLeaf_rBeforeLeaf_k(state: EliminationState, leaf_k: Leaf, leaf_r: Leaf, domain_k: Iterable[Pile] | None = None, domain_r: Iterable[Pile] | None = None) -> EliminationState:
	"""Exclude `leaf_r` from appearing before `leaf_k` in every `pile` in the domain of `leaf_k`.

	Parameters
	----------
	state : EliminationState
		Data basket, state of the local context, and state of the global context.
	leaf_k : int
		`leaf` that must be in a `pile` preceding the `pile` of `leaf_r`.
	leaf_r : int
		`leaf` that must be in a `pile` succeeding the `pile` of `leaf_k`.
	domain_k : Iterable[int] | None = None
		The domain of each `pile` at which `leaf_k` can be pinned. If `None`, every `pile` is in the domain.
	domain_r : Iterable[int] | None = None
		The domain of each `pile` at which `leaf_r` can be pinned. If `None`, every `pile` is in the domain.

	Returns
	-------
	EliminationState
		Same state instance, mutated with updated `listPermutationSpace`.

	See Also
	--------
	_excludeLeafRBeforeLeafK, theorem4, theorem2b
	"""
	if domain_k is None:
		domain_k = _e.getLeafDomain(state, leaf_k)
	for pile_k in sorted(domain_k, reverse=True):
		state = excludeLeaf_rBeforeLeaf_kAtPile_k(state, leaf_k, leaf_r, pile_k, domainOf_leaf_r=domain_r)
	return state

def excludeLeaf_rBeforeLeaf_kAtPile_k(
	state: EliminationState
	, leaf_k: Leaf
	, leaf_r: Leaf
	, pile_k: Pile
	, domainOf_leaf_r: Iterable[Pile] | None = None
) -> EliminationState:
	listPermutationSpace: Iterable[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []

	listPermutationSpaceUnchanged: list[PermutationSpace] = []
	listExcludeLeaf_r: Iterable[PermutationSpace] = []

	for permutationSpace in listPermutationSpace:
		if permutationSpace.leafPinnedAtPile吗(leaf_k, pile_k):
			listExcludeLeaf_r.append(permutationSpace)

		elif leafInLeafOptions吗(leaf_k, permutationSpace.getLeafOptions(pile_k, LeafOptions(0))):
			permutationSpaceCopy: PermutationSpace = permutationSpace.copy()
			permutationSpaceCopy[pile_k] = bit_clear(permutationSpaceCopy[pile_k], leaf_k)
			state.listPermutationSpace.append(permutationSpaceCopy)

			listExcludeLeaf_r.append(permutationSpace.atPilePinLeaf(pile_k, leaf_k))

		else:
			listPermutationSpaceUnchanged.append(permutationSpace)

	listPermutationSpace = listExcludeLeaf_r
	del listExcludeLeaf_r

	# TODO Choose between `if domainOf_leaf_r is None:` and
	# `domainOf_leaf_r = domainOf_leaf_r or getLeafDomain(self, leaf_r)`.

	# DEVELOPMENT
	# Replace an empty `Iterable` to prevent an error state, or as a convenient default.
	# Or passing an empty `Iterable` enables a no-op.
	if domainOf_leaf_r is None:
		domainOf_leaf_r = _e.getLeafDomain(state, leaf_r)

	for pile_r in filter(pile_k.__gt__, sorted(domainOf_leaf_r, reverse=True)):
		listPermutationSpace = atPileExcludeLeaf_inListPermutationSpace(listPermutationSpace, pile_r, leaf_r)

	state.listPermutationSpace.extend(listPermutationSpace)

	state.removeCreaseViolations().reduceAllPermutationSpace()

	state.listPermutationSpace.extend(listPermutationSpaceUnchanged)

	return state

def atPileExcludeLeaf_inListPermutationSpace(listPermutationSpace: Iterable[PermutationSpace], pile: Pile, leaf: Leaf) -> list[PermutationSpace]:
	"""Return a new list of `PermutationSpace` without `leaf` at `pile`.

	Parameters
	----------
	listPermutationSpace : Iterable[PermutationSpace]
		Collection of partial pinning dictionaries to transform.
	leaf : int
		`leaf` to exclude from `pile`.
	pile : int
		`pile` at which `leaf` must not be found.

	Returns
	-------
	listPermutationSpace : list[PermutationSpace]
		Expanded / filtered list respecting the exclusion constraint.

	See Also
	--------
	requireLeafPinnedAtPile
		Complementary operation that forces a `leaf` at a `pile`.
	"""
	listPermutationSpace, _pinnedAtPile = segregateLeafPinnedAtPile(listPermutationSpace, leaf, pile)
	groupByPilePinned: dict[bool, list[PermutationSpace]] = toolz_groupby(methodcaller('pilePinned吗', pile), listPermutationSpace)

	listPermutationSpace = groupByPilePinned.get(True, [])

	for permutationSpace in groupByPilePinned.get(False, []):
		permutationSpace[pile] = bit_clear(permutationSpace[pile], leaf)
		listPermutationSpace.append(permutationSpace)
	return listPermutationSpace

def Z0Z_atPileExcludeLeaf_inListPermutationSpace(listPermutationSpace: Iterable[PermutationSpace], pile: Pile, leaf: Leaf) -> list[PermutationSpace]:
	"""Return a new list of `PermutationSpace` without `leaf` at `pile`.

	Parameters
	----------
	listPermutationSpace : Iterable[PermutationSpace]
		Collection of partial pinning dictionaries to transform.
	leaf : int
		`leaf` to exclude from `pile`.
	pile : int
		`pile` at which `leaf` must not be found.

	Returns
	-------
	listPermutationSpace : list[PermutationSpace]
		Expanded / filtered list respecting the exclusion constraint.

	See Also
	--------
	requireLeafPinnedAtPile
		Complementary operation that forces a `leaf` at a `pile`.
	"""
	excluder: Callable[[PermutationSpace], PermutationSpace | None] = partial(atPileExcludeLeaf, pile=pile, leaf=leaf)
	return list(filter_map(excluder, listPermutationSpace))

#======== One `PermutationSpace` ===============================

def atPileExcludeLeaf(permutationSpace: PermutationSpace, pile: Pile, leaf: Leaf) -> PermutationSpace | None:
	returnMe: PermutationSpace | None = permutationSpace.copy()
	rangeOfPile: LeafSpace | None = returnMe[pile]
	if isLeafOptions吗(rangeOfPile):
		# If the range size of `pile` is 0 or 1, convert to None or `Leaf`.
		rangeOfPile = _e.leafOptionsLeafNone(rangeOfPile)
	if rangeOfPile == leaf or rangeOfPile is None:
		returnMe = None
	elif isLeaf吗(rangeOfPile):
		returnMe[pile] = rangeOfPile
	# The range size of `pile` is more than 1.
	else:
		rangeOfPile = rangeOfPile.bit_clear(leaf)
		# If the range size is now 1, convert to `Leaf`.
		rangeOfPile = _e.leafOptionsLeafNone(rangeOfPile)
		returnMe[pile] = raiseIfNone(rangeOfPile)
	return returnMe
