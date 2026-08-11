"""Generalized pinning functions in the "Elimination" algorithm for any `mapShape`.

Functions for 2^n-dimensional maps must go in other modules.
"""
from __future__ import annotations

from functools import partial
from gmpy2 import bit_clear
from humpy_cytoolz import groupby as toolz_groupby
from mapFolding import _e
from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
from mapFolding._e.filters import leafInChoicesLeaf吗, 是valid
from mapFolding._e.theTypes import ChoicesLeaf
from operator import methodcaller
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding._e.theTypes import Leaf, Pile

#======== Group by =======================

def segregateLeafPinnedAtPile(boxOfPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile) -> tuple[list[PermutationSpace], list[PermutationSpace]]:
	"""Partition `boxOfPermutationSpace` into (notPinned, isPinned) groups for `leaf` pinned at `pile`.

	Parameters
	----------
	boxOfPermutationSpace : Iterable[PermutationSpace]
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
	grouped: dict[bool, list[PermutationSpace]] = toolz_groupby(isPinned, boxOfPermutationSpace)
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
		Same state instance, mutated with updated `boxOfPermutationSpace`.

	See Also
	--------
	_excludeLeafRBeforeLeafK, theorem4, theorem2b
	"""
	if domain_k is None:
		domain_k = _e.getDomainLeaf(state, leaf_k)
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
	boxOfPermutationSpace: Iterable[PermutationSpace] = state.boxOfPermutationSpace
	state.boxOfPermutationSpace = []

	boxOfPermutationSpaceUnchanged: list[PermutationSpace] = []
	boxOfExcludeLeaf_r: Iterable[PermutationSpace] = []

	for permutationSpace in boxOfPermutationSpace:
		if permutationSpace.leafPinnedAtPile吗(leaf_k, pile_k):
			boxOfExcludeLeaf_r.append(permutationSpace)

		elif leafInChoicesLeaf吗(leaf_k, permutationSpace.getChoicesLeaf(pile_k, ChoicesLeaf(0))):
			permutationSpaceCopy: PermutationSpace = permutationSpace.copy()
			permutationSpaceCopy[pile_k] = bit_clear(permutationSpaceCopy[pile_k], leaf_k)
			state.boxOfPermutationSpace.append(permutationSpaceCopy)

			boxOfExcludeLeaf_r.append(permutationSpace.atPilePinLeaf(pile_k, leaf_k))

		else:
			boxOfPermutationSpaceUnchanged.append(permutationSpace)

	boxOfPermutationSpace = boxOfExcludeLeaf_r
	del boxOfExcludeLeaf_r

	# TODO Choose between `if domainOf_leaf_r is None:` and
	# `domainOf_leaf_r = domainOf_leaf_r or getDomainLeaf(self, leaf_r)`.

	# DEVELOPMENT
	# Replace an empty `Iterable` to prevent an error state, or as a convenient default.
	# Or passing an empty `Iterable` enables a no-op.
	if domainOf_leaf_r is None:
		domainOf_leaf_r = _e.getDomainLeaf(state, leaf_r)

	for pile_r in filter(pile_k.__gt__, sorted(domainOf_leaf_r, reverse=True)):
		boxOfPermutationSpace = 是valid(atPileExcludeLeaf_inBulk(boxOfPermutationSpace, pile_r, leaf_r))

	state.boxOfPermutationSpace.extend(boxOfPermutationSpace)

	state.removeCreaseViolations().reduceAllPermutationSpace()

	state.boxOfPermutationSpace.extend(boxOfPermutationSpaceUnchanged)

	return state

def atPileExcludeLeaf_inBulk(boxOfPermutationSpace: Iterable[PermutationSpace], pile: Pile, leaf: Leaf) -> list[PermutationSpace]:
	"""Return a new list of `PermutationSpace` without `leaf` at `pile`.

	Parameters
	----------
	boxOfPermutationSpace : Iterable[PermutationSpace]
		Collection of partial pinning dictionaries to transform.
	leaf : int
		`leaf` to exclude from `pile`.
	pile : int
		`pile` at which `leaf` must not be found.

	Returns
	-------
	boxOfPermutationSpace : list[PermutationSpace]
		Expanded / filtered list respecting the exclusion constraint.

	See Also
	--------
	requireLeafPinnedAtPile
		Complementary operation that forces a `leaf` at a `pile`.
	"""
	boxOfPermutationSpace, _pinnedAtPile = segregateLeafPinnedAtPile(boxOfPermutationSpace, leaf, pile)
	groupByPilePinned: dict[bool, list[PermutationSpace]] = toolz_groupby(methodcaller('pilePinned吗', pile), boxOfPermutationSpace)

	boxOfPermutationSpace = groupByPilePinned.get(True, [])

	for permutationSpace in groupByPilePinned.get(False, []):
		boxOfPermutationSpace.append(permutationSpace.atPileExcludeLeaf(pile, leaf))
	return boxOfPermutationSpace
