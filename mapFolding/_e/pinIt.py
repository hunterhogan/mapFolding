"""Generalized pinning functions in the "Elimination" algorithm for any `mapShape`.

Functions for 2^n-dimensional maps must go in other modules.
"""
from __future__ import annotations

from mapFolding import _e
from mapFolding._e.filters import leafInChoicesLeaf吗, 是valid
from mapFolding._e.theTypes import ChoicesLeaf
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable
	from mapFolding._e.dataBaskets import PermutationSpace, StateElimination
	from mapFolding.theTypes import Leaf, Pile

def excludeLeaf_rBeforeLeaf_k(state: StateElimination, leaf_k: Leaf, leaf_r: Leaf, domain_k: Iterable[Pile] | None = None, domain_r: Iterable[Pile] | None = None) -> StateElimination:
	"""Restrict permutation spaces so a selected leaf precedes another leaf.

	(AI generated docstring)

	You can require `leaf_k` to occur before `leaf_r` across the supplied pile domains. You replace the
	state's active permutation-space list with branches satisfying the restriction. Omit a domain to use
	`mapFolding._e.leafDomains.getDomainLeaf` [1].

	Domain coverage
	---------------
	The ordering restriction covers the supplied domains. Restricting a domain limits which pile positions
	are examined; an empty iterable applies no exclusions for that domain.

	Parameters
	----------
	state : StateElimination
		Elimination state whose active permutation spaces will be partitioned and reduced.
	leaf_k : Leaf
		Leaf that must occur earlier in the pile order.
	leaf_r : Leaf
		Leaf that must occur later in the pile order.
	domain_k : Iterable[Pile] | None = None
		Pile domain to examine for `leaf_k`. `None` computes the shape-based domain.
	domain_r : Iterable[Pile] | None = None
		Pile domain to examine for `leaf_r`. `None` computes the shape-based domain. A reusable iterable is
		required because the domain is examined for each pile in `domain_k`.

	Returns
	-------
	state : StateElimination
		The same state with updated permutation-space branches.

	References
	----------
	[1] `mapFolding._e.leafDomains.getDomainLeaf`
	"""
	if domain_k is None:
		domain_k = _e.getDomainLeaf(state, leaf_k)
	for pile_k in sorted(domain_k, reverse=True):
		state = excludeLeaf_rBeforeLeaf_kAtPile_k(state, leaf_k, leaf_r, pile_k, domainOf_leaf_r=domain_r)
	return state

def excludeLeaf_rBeforeLeaf_kAtPile_k(
	state: StateElimination
	, leaf_k: Leaf
	, leaf_r: Leaf
	, pile_k: Pile
	, domainOf_leaf_r: Iterable[Pile] | None = None
) -> StateElimination:
	"""Exclude an earlier leaf when another leaf is pinned at a selected pile.

	(AI generated docstring)

	You can isolate branches that pin `leaf_k` at `pile_k` and exclude `leaf_r` from earlier piles in
	those branches. You retain branches in which `leaf_k` does not occupy `pile_k`.

	Parameters
	----------
	state : StateElimination
		Elimination state whose active branches will be partitioned, pruned, and reduced.
	leaf_k : Leaf
		Leaf to pin at `pile_k` in the restricted branches.
	leaf_r : Leaf
		Leaf to exclude from piles earlier than `pile_k`.
	pile_k : Pile
		Pile at which the ordering restriction is conditioned.
	domainOf_leaf_r : Iterable[Pile] | None = None
		Pile positions to examine for `leaf_r`. `None` computes the shape-based domain [1]; an empty iterable
		applies no leaf exclusions.

	Returns
	-------
	state : StateElimination
		The same state containing restricted branches and retained alternatives.

	References
	----------
	[1] `mapFolding._e.leafDomains.getDomainLeaf`
	"""
	boxOfPermutationSpace: list[PermutationSpace] = state.boxOfPermutationSpace
	state.boxOfPermutationSpace = []

	boxOfPermutationSpaceUnchanged: list[PermutationSpace] = []
	boxOfExcludeLeaf_r: Iterable[PermutationSpace] = []

	for permutationSpace in boxOfPermutationSpace:
		if permutationSpace.leafPinnedAtPile吗(leaf_k, pile_k):
			boxOfExcludeLeaf_r.append(permutationSpace)

		elif leafInChoicesLeaf吗(leaf_k, permutationSpace.getChoicesLeaf(pile_k, ChoicesLeaf(0))):
			boxOfExcludeLeaf_r.append(permutationSpace.atPilePinLeaf(pile_k, leaf_k))
			state.boxOfPermutationSpace.append(permutationSpace.atPileExcludeLeaf(pile_k, leaf_k))

		else:
			boxOfPermutationSpaceUnchanged.append(permutationSpace)

	boxOfPermutationSpace = boxOfExcludeLeaf_r
	del boxOfExcludeLeaf_r

	# `if domainOf_leaf_r is None:` results in: 1) `None` is replaced with a default value and 2) a
	# no-op if the `Iterable` doesn't have elements. Alternatively, I could use

	# `domainOf_leaf_r = domainOf_leaf_r or _e.getDomainLeaf(state, leaf_r)`. That would convert the
	# no-op case to a default value. I "picked" this option due to inertia: I never got around to
	# analyzing which behavior would be "better."
	if domainOf_leaf_r is None:
		domainOf_leaf_r = _e.getDomainLeaf(state, leaf_r)

	for pile_r in filter(pile_k.__gt__, sorted(domainOf_leaf_r, reverse=True)):
		boxOfPermutationSpace = 是valid(atPileExcludeLeaf(boxOfPermutationSpace, pile_r, leaf_r))

	state.boxOfPermutationSpace.extend(boxOfPermutationSpace)

	state.removeCreaseViolations().reduceAllPermutationSpace()

	state.boxOfPermutationSpace.extend(boxOfPermutationSpaceUnchanged)

	return state

def atPileExcludeLeaf(boxOfPermutationSpace: list[PermutationSpace], pile: Pile, leaf: Leaf) -> list[PermutationSpace]:
	"""Exclude a leaf at one pile in every supplied permutation space.

	(AI generated docstring)

	You can exclude a leaf from each permutation space in `boxOfPermutationSpace` and receive the
	original list with its spaces modified in place. You can invalidate a branch when the excluded leaf
	is pinned at `pile` or the exclusion empties the leaf set. Remove invalid branches separately.

	Parameters
	----------
	boxOfPermutationSpace : list[PermutationSpace]
		List of permutation spaces to modify.
	pile : Pile
		Pile from which to exclude `leaf`.
	leaf : Leaf
		Leaf index to remove from the pile's leaf set.

	Returns
	-------
	boxOfPermutationSpace : list[PermutationSpace]
		The same list containing the modified permutation spaces.
	"""
	for permutationSpace in boxOfPermutationSpace:
		permutationSpace.atPileExcludeLeaf(pile, leaf)
	return boxOfPermutationSpace
