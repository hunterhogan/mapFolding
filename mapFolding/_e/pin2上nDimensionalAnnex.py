"""Reduce permutation spaces through iterative constraint propagation.

You can use this module to shrink the search space for map-folding computations by applying
multiple constraint-propagation strategies in a unified reduction loop. The module implements
a single large constraint-satisfaction algorithm expressed as a collection of specialized
reduction functions that reinforce each other. Each function encodes one constraint type
(crease adjacency, conditional predecessors, crossed creases, naked subsets, etc.), and the
module orchestrates iterative application of these constraints until no further reduction
occurs.

Architecture
------------
The module is organized as one conceptual algorithm split across multiple functions for
readability and maintainability:

1. `reduceAllPermutationSpace` is the orchestrator that applies each
	reduction function in sequence until the permutation space stabilizes.

2. The `_reducePermutationSpace_*` functions are specialized constraint encoders that each
	implement one type of constraint. These functions are curried to accept `state` first,
	then `permutationSpace`, enabling use with `filter_map` [1].

3. `_reduceLeafSpace` is the shared subroutine that handles the mechanical work of updating
	`LeafOptions` at specified piles and propagating newly pinned leaves. All constraint
	encoders call `_reduceLeafSpace` to perform the actual updates.

The functions are not independent algorithms; the functions are interdependent components of
a constraint-propagation system. Each function assumes other functions will run afterward to
propagate the consequences of newly pinned leaves or reduced domains.

Functions
---------
Public
	reduceAllPermutationSpace
		Reduce permutation space by iteratively applying constraint propagation.

Private (Constraint Encoders)
	_reducePermutationSpace_byCrease
		I use this to enforce crease adjacency constraints.
	_reducePermutationSpace_ConditionalPredecessors
		I use this to enforce conditional predecessor constraints.
	_reducePermutationSpace_CrossedCreases
		I use this to detect and eliminate crossed creases.
	_reducePermutationSpace_HeadsBeforeTails
		I use this to enforce head-before-tail ordering constraints.
	_reducePermutationSpace_LeafIsPinned
		I use this to propagate leaf pinning constraints.
	_reducePermutationSpace_nakedSubset
		I use this to detect and exploit naked subset constraints.
	_reducePermutationSpace_noConsecutiveDimensions
		I use this to enforce non-consecutive dimension constraints.
	_reducePermutationSpace_leafDomainOf1
		I use this to detect and pin leaves with domain size one.

Private (Shared Subroutine)
	_reduceLeafSpace
		I use this to update permutation space by removing forbidden leaves from piles.

Private (Utilities)
	ImaOddLeaf2上nDimensional
		I use this to check parity for 2^n-dimensional maps using bit operations.

References
----------
[1] more_itertools.filter_map
	https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.filter_map

"""
from __future__ import annotations

from humpy_cytoolz import keyfilter as filterPile, valfilter as filterLeaf
from hunterMakesPy import inclusive
from mapFolding._e import (
	dimensionNearestTail, dimensionNearest首, getDictionaryConditionalLeafPredecessors, getLeavesCreaseAnte, getLeavesCreasePost, leafOrigin,
	makeLeafAntiOptions, mapShapeIs2上nDimensions)
from mapFolding._e.dataBaskets import PermutationSpace
from mapFolding._e.filters import isLeafOptions吗, isLeaf吗, notLeafOriginOrLeaf零, notPileLast
from mapFolding._e.pinIt import (
	reduceLeafSpace, reducePermutationSpace_CrossedCreases, reducePermutationSpace_leafDomainOf1, reducePermutationSpace_LeafIsPinned,
	reducePermutationSpace_nakedSubset)
from mapFolding.genericNeedsNewHome import between吗, DOTitems
from more_itertools import pairwise, triplewise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator, Sequence
	from mapFolding._e.dataBaskets import EliminationState
	from mapFolding._e.theTypes import Leaf, LeafOptions, Pile

# ======== Reducing `LeafOptions` ===============================

def _byCrease2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to enforce crease adjacency constraints.

	I use this constraint encoder to enforce that when a leaf is pinned at a pile and the
	adjacent pile has undetermined `LeafOptions`, the adjacent pile can only contain leaves
	that are crease neighbors of the pinned leaf. I identify pinned-leaf-adjacent-to-undetermined
	configurations and restrict the undetermined pile to crease neighbors using `_reduceLeafSpace`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	"""
	# TODO (High value improvement) To generalize, I need to
	# - know how to compute crease neighbors for arbitrary map shapes, and
	# - know if this algorithm is valid for arbitrary map shapes.
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		sum首: int = sum(map(dimensionNearest首, permutationSpace.values()))

		for (pile_k, leafSpace_k), (pile_r, leafSpace_r) in pairwise(permutationSpace.items()):
			if isLeaf吗(leafSpace_k) and isLeafOptions吗(leafSpace_r):
				pilesToUpdate: tuple[tuple[Pile, LeafOptions]] = ((pile_r, leafSpace_r),)
				leavesCrease: Iterator[Leaf] = getLeavesCreasePost(state, leafSpace_k)  # NOTE 2上nDimensional
			elif isLeafOptions吗(leafSpace_k) and isLeaf吗(leafSpace_r):
				pilesToUpdate = ((pile_k, leafSpace_k),)
				leavesCrease = getLeavesCreaseAnte(state, leafSpace_r)  # NOTE 2上nDimensional
			else:
				continue

			if not (permutationSpace := reduceLeafSpace(state, permutationSpace, pilesToUpdate
					, makeLeafAntiOptions(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease))
			)):
				return None

		if sum(map(dimensionNearest首, permutationSpace.values())) < sum首:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def _conditionalPredecessors2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to enforce conditional predecessor constraints.

	I use this constraint encoder to enforce that when a `Leaf` is pinned at a `Pile` and the `Leaf`
	has conditional `Leaf` predecessors at that `Pile`, then those `Leaf` predecessors cannot appear
	after that `Pile`.

	My formulas for computing conditional `Leaf` predecessors are inefficient, so I precompute them
	for 2ⁿ-dimensional maps with `n ≥ 6` and store them in `dictionaryConditionalLeafPredecessors`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.
	"""
	# -------------- Guard -------------------------------------------
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
		return permutationSpace

	# -------------- Initialize ------------------------------------
	leafAtPilePredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = getDictionaryConditionalLeafPredecessors(state)
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		# -------------- Initialize again ------------------------------------
		permutationSpaceHasNewLeaf = False
		sum首: int = sum(map(dimensionNearest首, permutationSpace.values()))

		for pile, leaf in DOTitems(filterPile(notPileLast(state.pileLast)
								, filterLeaf(notLeafOriginOrLeaf零
								, filterLeaf(leafAtPilePredecessors.__contains__
								, permutationSpace.extractPinnedLeaves()))
		)):
			if (pile in leafAtPilePredecessors[leaf]) and not (permutationSpace := reduceLeafSpace(state, permutationSpace
				, DOTitems(PermutationSpace(filterPile(between吗(pile + inclusive, state.pileLast - inclusive), permutationSpace)).extractUndeterminedPiles())
				, makeLeafAntiOptions(state.leavesTotal, leafAtPilePredecessors[leaf][pile])
			)):
				return None

		if sum(map(dimensionNearest首, permutationSpace.values())) < sum首:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

# TODO I don't think this applies to all map shapes.
def _headsBeforeTails2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to enforce head-before-tail ordering constraints.

	I use this constraint encoder to enforce that leaves with large coordinates in a dimension (tail)
	can only appear after all leaves with small coordinates in that dimension (head) have appeared.
	When a leaf with nonzero nearest head dimension is pinned, I remove all leaves with larger
	coordinates in that dimension from preceding piles. When a leaf with nonzero nearest tail
	dimension is pinned, I remove all leaves with smaller coordinates in that dimension from
	subsequent piles.

	Algorithm Details
	-----------------
	For each pinned leaf:

	1. Compute `dimensionNearest首(leaf)` [1] to identify the dimension with the smallest coordinate
		magnitude from the head.
	2. If nonzero, remove all leaves with larger coordinates in that dimension from piles before
		`pile`.
	3. Compute `dimensionNearestTail(leaf)` [2] to identify the dimension with the smallest coordinate
		magnitude from the tail.
	4. If nonzero, remove all leaves with smaller coordinates in that dimension from piles after
		`pile`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	References
	----------
	[1] mapFolding._e.dimensionNearest首

	[2] mapFolding._e.dimensionNearestTail
	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		sum首: int = sum(map(dimensionNearest首, permutationSpace.values()))

		# TODO `notLeafOriginOrLeaf零` and `pile1stOpen` are specific to 2ⁿ-dimensional maps. Adjust these if moved to `pinIt`.
		pile1stOpen: int = 2
		for pile, leaf in DOTitems(filterPile(notPileLast(state.pileLast), filterLeaf(notLeafOriginOrLeaf零, permutationSpace.extractPinnedLeaves()))):
			dimensionHead: int = dimensionNearest首(leaf)
			if 0 < dimensionHead and not (permutationSpace := reduceLeafSpace(state, permutationSpace
				, DOTitems(PermutationSpace(filterPile(between吗(pile1stOpen, pile - inclusive), permutationSpace)).extractUndeterminedPiles())
				, makeLeafAntiOptions(state.leavesTotal, range(state.productsOfDimensions[dimensionHead], state.leavesTotal, state.productsOfDimensions[dimensionHead]))
			)):
				return None

			dimensionTail: int = dimensionNearestTail(leaf)
			if 0 < dimensionTail and not (permutationSpace := reduceLeafSpace(state, permutationSpace
				, DOTitems(PermutationSpace(filterPile(between吗(pile + inclusive, state.pileLast - inclusive), permutationSpace)).extractUndeterminedPiles())
				, makeLeafAntiOptions(state.leavesTotal, range(leafOrigin, state.sumsOfProductsOfDimensions[dimensionTail]))
			)):
				return None

		if sum(map(dimensionNearest首, permutationSpace.values())) < sum首:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

# TODO I don't think this applies to all map shapes.
def _noConsecutiveDimensions2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to enforce non-consecutive dimension constraints.

	I use this constraint encoder to detect arithmetic progressions in pinned leaves and forbid
	the next term in the progression from appearing at the adjacent pile. When three consecutive
	piles contain leaves forming an arithmetic progression (leaf_k, leaf, leaf_r where
	`leaf - leaf_k == leaf_r - leaf`), the next term in the progression cannot appear at the
	next pile because map foldings cannot have four consecutive leaves in arithmetic progression.

	The function examines all triples of consecutive piles and identifies configurations where:
	1. Two adjacent piles have pinned leaves and the third has `LeafOptions`, or
	2. The middle pile has `LeafOptions` and the outer two have pinned leaves.

	For each pattern, I compute the forbidden leaf (the next term in the arithmetic progression)
	and remove that leaf from the undetermined pile using `_reduceLeafSpace`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		sum首: int = sum(map(dimensionNearest首, permutationSpace.values()))

		for (pile_k, leafSpace_k), (pile, leafSpace), (pile_r, leafSpace_r) in triplewise(sorted(DOTitems(permutationSpace))):
			if isLeaf吗(leafSpace_k) and isLeaf吗(leafSpace) and isLeafOptions吗(leafSpace_r):
				pilesToUpdate: tuple[tuple[Pile, LeafOptions]] = ((pile_r, leafSpace_r),)
				leafForbidden: Leaf = leafSpace + (leafSpace - leafSpace_k)
			elif isLeaf吗(leafSpace_k) and isLeafOptions吗(leafSpace) and isLeaf吗(leafSpace_r):
				pilesToUpdate = ((pile, leafSpace),)
				leafForbidden = (leafSpace_k + leafSpace_r) // 2
			elif isLeafOptions吗(leafSpace_k) and isLeaf吗(leafSpace) and isLeaf吗(leafSpace_r):
				pilesToUpdate = ((pile_k, leafSpace_k),)
				leafForbidden = leafSpace - (leafSpace_r - leafSpace)
			else:
				continue

			if 0 <= leafForbidden < state.leavesTotal and not (permutationSpace :=
				reduceLeafSpace(state, permutationSpace, pilesToUpdate, makeLeafAntiOptions(state.leavesTotal, [leafForbidden]))
			):
				return None

		if sum(map(dimensionNearest首, permutationSpace.values())) < sum首:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

# TODO The order of the functions can cause tests to fail. I don't think that ought to happen.
listFunctionsReduction2上nDimensional: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = (
	reducePermutationSpace_LeafIsPinned,
	_byCrease2上nDimensional,
	reducePermutationSpace_leafDomainOf1,
	reducePermutationSpace_nakedSubset,
	_headsBeforeTails2上nDimensional,
	_conditionalPredecessors2上nDimensional,
	reducePermutationSpace_CrossedCreases,
	_noConsecutiveDimensions2上nDimensional,
)
