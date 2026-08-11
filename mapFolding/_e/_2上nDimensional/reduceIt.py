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
	`ChoicesLeaf` at specified piles and propagating newly pinned leaves. All constraint
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

from functools import partial
from gmpy2 import bit_flip
from humpy_cytoolz import get, groupby as toolz_groupby, keyfilter as filterPile, valfilter as filterLeaf
from hunterMakesPy import errorL33T, inclusive, raiseIfNone
from itertools import combinations
from mapFolding._e import leafOrigin, makeAntiChoicesLeaf
from mapFolding._e._2上nDimensional import (
	dimensionNearestTail, dimensionNearest首, getLeafPredecessors, getLeavesCreaseAnte, getLeavesCreasePost, moreThanLeaf零吗)
from mapFolding._e._2上nDimensional.filters import oddLeaf2上nDimensional吗
from mapFolding._e.algorithms.iff import creaseViolation吗
from mapFolding._e.filters import choicesLeaf吗, leafPinned吗, leaf吗, notPileLast, pileChoicesLeaf吗
from mapFolding._e.reduceIt import (
	reduceLeafSpace, reducePermutationSpace_leafDomainOf0or1, reducePermutationSpace_LeafIsPinned, reducePermutationSpace_nakedSubset)
from mapFolding._e.theTypes import Leaf, Pile
from mapFolding.beDRY import mapShapeIs2上nDimensions
from more_itertools import extract, pairwise, triplewise
from typing import TYPE_CHECKING
from Z0Z_tools import DOTitems, reverseLookup

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator, Sequence
	from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
	from mapFolding._e.theTypes import ChoicesLeaf, PinnedLeaves

#======== Reducing `ChoicesLeaf` ===============================

def _byCrease2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""I use this to enforce crease adjacency constraints.

	I use this constraint encoder to enforce that when a leaf is pinned at a pile and the
	adjacent pile has undetermined `ChoicesLeaf`, the adjacent pile can only contain leaves
	that are crease neighbors of the pinned leaf. I identify pinned-leaf-adjacent-to-undetermined
	configurations and restrict the undetermined pile to crease neighbors using `_reduceLeafSpace`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: choicesLeaf`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	"""
	# TODO (High value improvement) To generalize, at least one of the adjacent leaves must be a crease.
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		for (pile_k, leafSpace_k), (pile_r, leafSpace_r) in pairwise(permutationSpace.items()):
			if leaf吗(leafSpace_k) and choicesLeaf吗(leafSpace_r):
				pilesToUpdate: tuple[tuple[Pile, ChoicesLeaf]] = ((pile_r, leafSpace_r),)
				leavesCrease: Iterator[Leaf] = getLeavesCreasePost(state, leafSpace_k)  # DEVELOPMENT 2上nDimensional
			elif choicesLeaf吗(leafSpace_k) and leaf吗(leafSpace_r):
				pilesToUpdate = ((pile_k, leafSpace_k),)
				leavesCrease = getLeavesCreaseAnte(state, leafSpace_r)  # DEVELOPMENT 2上nDimensional
			else:
				continue

			permutationSpace = reduceLeafSpace(permutationSpace, pilesToUpdate
					, makeAntiChoicesLeaf(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease))
			)
			if not permutationSpace.valid:
				#=SIN= Early return.
				return permutationSpace

		if permutationSpace.leafCount < leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def _conditionalPredecessors2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace:
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
		A dictionary of `pile: leaf` and/or `pile: choicesLeaf`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.
	"""
	#-------------- Guard -------------------------------------------
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToRideThis=6):
		return permutationSpace

	#-------------- Initialize ------------------------------------
	leafAtPilePredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = getLeafPredecessors(state)
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		#-------------- Initialize again ------------------------------------
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		leavesPinned: PinnedLeaves = filterLeaf(leafAtPilePredecessors.__contains__, permutationSpace.pinnedLeaves(), factory=dict[Pile, Leaf])
		leavesPinned = filterLeaf(moreThanLeaf零吗, leavesPinned, factory=dict[Pile, Leaf])
		for pile, leaf in DOTitems(filterPile(partial(notPileLast, state.pileLast), leavesPinned)):
			if pile in leafAtPilePredecessors[leaf]:
				permutationSpace = reduceLeafSpace(permutationSpace
					, DOTitems(filterPile(pile.__lt__, permutationSpace.undeterminedPiles()))
					, makeAntiChoicesLeaf(state.leavesTotal, leafAtPilePredecessors[leaf][pile])
				)
				if not permutationSpace.valid:
					#=SIN= Early return.
					return permutationSpace

		if permutationSpace.leafCount < leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def _odd吗(_mapShape: tuple[int, ...], dimension: int) -> Callable[[tuple[Pile, Leaf]], bool]:
	def workhorse(pileLeaf: tuple[Pile, Leaf]) -> bool:
		return bool(oddLeaf2上nDimensional吗(dimension, leaf=pileLeaf[1]))
	return workhorse

def _crossedCreases2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""I use this to detect and eliminate crossed creases.

	I use this constraint encoder to detect configurations where two creases would cross physically
	and either invalidate `permutationSpace` or restrict forbidden pile positions for unpinned crease
	leaves. For each dimension, I partition pinned leaves by parity (even/odd coordinate in that
	dimension), identify crease pairs where one leaf is pinned and the other is not, and compute
	forbidden pile positions where the unpinned leaf cannot appear without causing a crease crossing.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: choicesLeaf`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.
	"""
	pileOf_kCrease: Pile = errorL33T
	pileOf_rCrease: Pile = errorL33T
	pilesForbidden: Iterable[Pile] = []
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		for dimension in range(state.dimensionsTotal):
			groupedByParity: dict[bool, list[tuple[Pile, Leaf]]] = toolz_groupby(_odd吗(state.mapShape, dimension), DOTitems(permutationSpace.pinnedLeaves()))

			for upDown, leftRight in ((False, True), (True, False)):
				leavesPinnedParityOpposite: PinnedLeaves = dict(get(upDown, groupedByParity, ()))

				for ((pileOf_k, leaf_k), (pileOf_r, leaf_r)) in combinations(sorted(get(leftRight, groupedByParity, ())), 2):
					leaf_kCrease: Leaf = int(bit_flip(leaf_k, dimension))
					leaf_rCrease: Leaf = int(bit_flip(leaf_r, dimension))

					if leaf_kCreaseIsPinned := leafPinned吗(leavesPinnedParityOpposite, leaf_kCrease):
						pileOf_kCrease = raiseIfNone(reverseLookup(leavesPinnedParityOpposite, leaf_kCrease))
					if leaf_rCreaseIsPinned := leafPinned吗(leavesPinnedParityOpposite, leaf_rCrease):
						pileOf_rCrease = raiseIfNone(reverseLookup(leavesPinnedParityOpposite, leaf_rCrease))

					if leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
						antiChoicesLeaf: ChoicesLeaf = makeAntiChoicesLeaf(state.leavesTotal, (leaf_rCrease,))

						if pileOf_k < pileOf_r < pileOf_kCrease:
							pilesForbidden = frozenset([*range(pileOf_k), *range(pileOf_kCrease + 1, state.pileLast + inclusive)])
						elif pileOf_kCrease < pileOf_r < pileOf_k:
							pilesForbidden = frozenset([*range(pileOf_kCrease), *range(pileOf_k + 1, state.pileLast + inclusive)])
						elif (pileOf_r < pileOf_kCrease < pileOf_k) or (pileOf_kCrease < pileOf_k < pileOf_r):
							pilesForbidden = range(pileOf_kCrease + 1, pileOf_k)
						elif (pileOf_r < pileOf_k < pileOf_kCrease) or (pileOf_k < pileOf_kCrease < pileOf_r):
							pilesForbidden = range(pileOf_k + 1, pileOf_kCrease)

					elif not leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
						antiChoicesLeaf = makeAntiChoicesLeaf(state.leavesTotal, (leaf_kCrease,))

						if pileOf_rCrease < pileOf_k < pileOf_r:
							pilesForbidden = frozenset([*range(pileOf_rCrease), *range(pileOf_r + 1, state.pileLast + inclusive)])
						elif pileOf_r < pileOf_k < pileOf_rCrease:
							pilesForbidden = frozenset([*range(pileOf_r), *range(pileOf_rCrease + 1, state.pileLast + inclusive)])
						elif (pileOf_k < pileOf_r < pileOf_rCrease) or (pileOf_r < pileOf_rCrease < pileOf_k):
							pilesForbidden = range(pileOf_r + 1, pileOf_rCrease)
						elif (pileOf_k < pileOf_rCrease < pileOf_r) or (pileOf_rCrease < pileOf_r < pileOf_k):
							pilesForbidden = range(pileOf_rCrease + 1, pileOf_r)

					elif leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
						if creaseViolation吗(pileOf_k, pileOf_r, pileOf_kCrease, pileOf_rCrease):
							#=SIN= Early return.
							permutationSpace.valid = False
							return permutationSpace
						continue

					else:
						continue

					permutationSpace = reduceLeafSpace(permutationSpace
							, filter(pileChoicesLeaf吗, extract(permutationSpace.items(), pilesForbidden))
							, antiChoicesLeaf
					)
					if not permutationSpace.valid:
						#=SIN= Early return.
						return permutationSpace

		if leafCount < permutationSpace.leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def _headsBeforeTails2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace:
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
		A dictionary of `pile: leaf` and/or `pile: choicesLeaf`.

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
		leafCount: int = permutationSpace.leafCount

		pile1stOpen: int = 2
		leavesPinned: PinnedLeaves = filterLeaf(moreThanLeaf零吗, permutationSpace.pinnedLeaves(), factory=dict[Pile, Leaf])
		for pile, leaf in DOTitems(filterPile(partial(notPileLast, state.pileLast), leavesPinned)):
			dimensionHead: int = dimensionNearest首(leaf)
			if 0 < dimensionHead:
				permutationSpace = reduceLeafSpace(permutationSpace
					, DOTitems(filterLeaf(choicesLeaf吗, filterPile(pile1stOpen.__le__, filterPile(pile.__gt__, permutationSpace))))
					, makeAntiChoicesLeaf(state.leavesTotal, range(state.mapShapeProducts[dimensionHead], state.leavesTotal, state.mapShapeProducts[dimensionHead]))
				)
				if not permutationSpace.valid:
					#=SIN= Early return.
					return permutationSpace

			dimensionTail: int = dimensionNearestTail(leaf)
			if 0 < dimensionTail:
				permutationSpace = reduceLeafSpace(permutationSpace
					, DOTitems(filterPile(pile.__lt__, permutationSpace.undeterminedPiles()))
					, makeAntiChoicesLeaf(state.leavesTotal, range(leafOrigin, state.mapShapeProductsSums[dimensionTail]))
				)
				if not permutationSpace.valid:
					#=SIN= Early return.
					return permutationSpace

		if permutationSpace.leafCount < leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def _noConsecutiveDimensions2上nDimensional(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""I use this to enforce non-consecutive dimension constraints.

	I use this constraint encoder to detect arithmetic progressions in pinned leaves and forbid
	the next term in the progression from appearing at the adjacent pile. When three consecutive
	piles contain leaves forming an arithmetic progression (leaf_k, leaf, leaf_r where
	`leaf - leaf_k == leaf_r - leaf`), the next term in the progression cannot appear at the
	next pile because map foldings cannot have four consecutive leaves in arithmetic progression.

	The function examines all triples of consecutive piles and identifies configurations where:
	1. Two adjacent piles have pinned leaves and the third has `ChoicesLeaf`, or
	2. The middle pile has `ChoicesLeaf` and the outer two have pinned leaves.

	For each pattern, I compute the forbidden leaf (the next term in the arithmetic progression)
	and remove that leaf from the undetermined pile using `_reduceLeafSpace`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: choicesLeaf`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		for (pile_k, leafSpace_k), (pile, leafSpace), (pile_r, leafSpace_r) in triplewise(sorted(DOTitems(permutationSpace))):
			if leaf吗(leafSpace_k) and leaf吗(leafSpace) and choicesLeaf吗(leafSpace_r):
				pilesToUpdate: tuple[tuple[Pile, ChoicesLeaf]] = ((pile_r, leafSpace_r),)
				leafForbidden: Leaf = leafSpace + (leafSpace - leafSpace_k)
			elif leaf吗(leafSpace_k) and choicesLeaf吗(leafSpace) and leaf吗(leafSpace_r):
				pilesToUpdate = ((pile, leafSpace),)
				leafForbidden = (leafSpace_k + leafSpace_r) // 2
			elif choicesLeaf吗(leafSpace_k) and leaf吗(leafSpace) and leaf吗(leafSpace_r):
				pilesToUpdate = ((pile_k, leafSpace_k),)
				leafForbidden = leafSpace - (leafSpace_r - leafSpace)
			else:
				continue

			if 0 <= leafForbidden < state.leavesTotal:
				permutationSpace = reduceLeafSpace(permutationSpace, pilesToUpdate, makeAntiChoicesLeaf(state.leavesTotal, [leafForbidden]))
				if not permutationSpace.valid:
					#=SIN= Early return.
					return permutationSpace

		if permutationSpace.leafCount < leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

# TODO The order of the functions can cause tests to fail. I don't think that ought to happen.
boxOfFunctionsReduction2上nDimensional: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace]] = (
	_noConsecutiveDimensions2上nDimensional,
	_crossedCreases2上nDimensional,
	_conditionalPredecessors2上nDimensional,
	_headsBeforeTails2上nDimensional,
	reducePermutationSpace_nakedSubset,
	reducePermutationSpace_leafDomainOf0or1,
	_byCrease2上nDimensional,
	reducePermutationSpace_LeafIsPinned,
)
