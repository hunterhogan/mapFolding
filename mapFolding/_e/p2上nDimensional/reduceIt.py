"""Provide the ordered coordinate-based reduction sequence.

You can select the reduction functions used to refine possible foldings for maps
whose axis lengths are `2`. The public sequence works with
`mapFolding._e.dataBaskets.StateElimination.reduceAllPermutationSpace` [1].

Contents
--------
Variables
	boxOfFunctionsReduction2上nDimensional
		Select the ordered reduction functions for maps whose axis lengths are two.

References
----------
[1] `mapFolding._e.dataBaskets.StateElimination.reduceAllPermutationSpace`
"""
from __future__ import annotations

from functools import partial
from gmpy2 import bit_flip
from humpy_cytoolz import get, groupby as toolz_groupby, keyfilter as filterPile, valfilter as filterLeaf
from hunterMakesPy import errorL33T, inclusive, raiseIfNone
from itertools import combinations
from mapFolding._e import makeAntiChoicesLeaf
from mapFolding._e.algorithms.iff import creaseViolation吗
from mapFolding._e.filters import choicesLeaf吗, leafPinned吗, leaf吗, notPileLast, pileChoicesLeaf吗
from mapFolding._e.p2上nDimensional import (
	getLeafPredecessors, getLeavesCreaseAnte, getLeavesCreasePost, moreThanLeaf零吗, 工dimensionTail, 工dimension首零)
from mapFolding._e.p2上nDimensional.filters import oddLeaf2上nDimensional吗
from mapFolding._e.reduceIt import (
	reduceLeafSpace, reducePermutationSpace_leafDomainOf0or1, reducePermutationSpace_LeafIsPinned, reducePermutationSpace_nakedSubset)
from mapFolding.beDRY import mapShapeIs2上nDimensions
from mapFolding.theTypes import Leaf, Pile
from more_itertools import extract, pairwise, triplewise
from typing import TYPE_CHECKING
from Z0Z_tools import DOTitems, reverseLookup

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator, Sequence
	from mapFolding._e.dataBaskets import PermutationSpace, StateElimination
	from mapFolding._e.theTypes import ChoicesLeaf, PinnedLeaves

#======== Reducing `ChoicesLeaf` ===============================

def _byCrease2上nDimensional(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Enforce crease adjacency constraints.

	You can restrict an undetermined pile adjacent to a pinned leaf to that leaf's crease neighbors. You
	identify pinned leaves next to undetermined `ChoicesLeaf` assignments and restrict those assignments
	with `_reduceLeafSpace`.

	Parameters
	----------
	state : StateElimination
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
					, makeAntiChoicesLeaf(state.totalLeaves, set(range(state.totalLeaves)).difference(leavesCrease))
			)
			if not permutationSpace.valid:
				#=SIN= Early return.
				return permutationSpace

		if permutationSpace.leafCount < leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def _conditionalPredecessors2上nDimensional(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Enforce conditional predecessor constraints.

	You can exclude conditional `Leaf` predecessors from piles after a pinned `Leaf`. You apply the
	restriction when the pinned `Leaf` has conditional predecessors at its `Pile`.

	You can reuse precomputed conditional `Leaf` predecessors for 2ⁿ-dimensional maps with `n ≥ 6`
	through `dictionaryConditionalLeafPredecessors`, avoiding repeated evaluation of the inefficient
	formulas.

	Parameters
	----------
	state : StateElimination
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
					, makeAntiChoicesLeaf(state.totalLeaves, leafAtPilePredecessors[leaf][pile])
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

def _crossedCreases2上nDimensional(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Detect and eliminate crossed creases.

	You can detect configurations where two creases would cross and either invalidate `permutationSpace`
	or restrict forbidden pile positions for unpinned crease leaves. For each dimension, you partition
	pinned leaves by coordinate parity, identify crease pairs with one pinned and one unpinned leaf, and
	compute forbidden pile positions where the unpinned leaf would cause a crossing.

	Parameters
	----------
	state : StateElimination
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

		for dimension in range(state.totalDimensions):
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
						antiChoicesLeaf: ChoicesLeaf = makeAntiChoicesLeaf(state.totalLeaves, (leaf_rCrease,))

						if pileOf_k < pileOf_r < pileOf_kCrease:
							pilesForbidden = frozenset([*range(pileOf_k), *range(pileOf_kCrease + 1, state.pileLast + inclusive)])
						elif pileOf_kCrease < pileOf_r < pileOf_k:
							pilesForbidden = frozenset([*range(pileOf_kCrease), *range(pileOf_k + 1, state.pileLast + inclusive)])
						elif (pileOf_r < pileOf_kCrease < pileOf_k) or (pileOf_kCrease < pileOf_k < pileOf_r):
							pilesForbidden = range(pileOf_kCrease + 1, pileOf_k)
						elif (pileOf_r < pileOf_k < pileOf_kCrease) or (pileOf_k < pileOf_kCrease < pileOf_r):
							pilesForbidden = range(pileOf_k + 1, pileOf_kCrease)

					elif not leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
						antiChoicesLeaf = makeAntiChoicesLeaf(state.totalLeaves, (leaf_kCrease,))

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

def _headsBeforeTails2上nDimensional(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Enforce head-before-tail ordering constraints.

	You can require leaves with large coordinates in a dimension (tail) to follow all leaves with small
	coordinates in that dimension (head). When a leaf with nonzero nearest head dimension is pinned, you
	remove leaves with larger coordinates in that dimension from preceding piles. When a leaf with
	nonzero nearest tail dimension is pinned, you remove leaves with smaller coordinates in that
	dimension from subsequent piles.

	Algorithm Details
	-----------------
	For each pinned leaf:

	1. Compute `工dimension首零(leaf)` [1] to identify the dimension with the smallest coordinate
		magnitude from the head.
	2. If nonzero, remove all leaves with larger coordinates in that dimension from piles before
		`pile`.
	3. Compute `工dimensionTail(leaf)` [2] to identify the dimension with the smallest coordinate
		magnitude from the tail.
	4. If nonzero, remove all leaves with smaller coordinates in that dimension from piles after
		`pile`.

	Parameters
	----------
	state : StateElimination
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: choicesLeaf`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	References
	----------
	[1] mapFolding._e.工dimension首零

	[2] mapFolding._e.工dimensionTail
	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		pile1stOpen: int = 2
		leavesPinned: PinnedLeaves = filterLeaf(moreThanLeaf零吗, permutationSpace.pinnedLeaves(), factory=dict[Pile, Leaf])
		for pile, leaf in DOTitems(filterPile(partial(notPileLast, state.pileLast), leavesPinned)):
			dimensionHead: int = 工dimension首零(leaf)
			if 0 < dimensionHead:
				permutationSpace = reduceLeafSpace(permutationSpace
					, DOTitems(filterLeaf(choicesLeaf吗, filterPile(pile1stOpen.__le__, filterPile(pile.__gt__, permutationSpace))))
					, makeAntiChoicesLeaf(state.totalLeaves, range(state.mapShapeProducts[dimensionHead], state.totalLeaves, state.mapShapeProducts[dimensionHead]))
				)
				if not permutationSpace.valid:
					#=SIN= Early return.
					return permutationSpace

			dimensionTail: int = 工dimensionTail(leaf)
			if 0 < dimensionTail:
				permutationSpace = reduceLeafSpace(permutationSpace
					, DOTitems(filterPile(pile.__lt__, permutationSpace.undeterminedPiles()))
					, makeAntiChoicesLeaf(state.totalLeaves, range(state.mapShapeProductsSums[dimensionTail]))
				)
				if not permutationSpace.valid:
					#=SIN= Early return.
					return permutationSpace

		if permutationSpace.leafCount < leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def _noConsecutiveDimensions2上nDimensional(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Enforce non-consecutive dimension constraints.

	You can detect arithmetic progressions in pinned leaves and exclude the next term from an adjacent
	pile. When three consecutive piles contain an arithmetic progression (leaf_k, leaf, leaf_r where
	`leaf - leaf_k == leaf_r - leaf`), you exclude the next term from the next pile because map foldings
	cannot have four consecutive leaves in arithmetic progression.

	You examine all triples of consecutive piles for the following configurations.
	1. Two adjacent piles have pinned leaves and the third has `ChoicesLeaf`, or
	2. The middle pile has `ChoicesLeaf` and the outer two have pinned leaves.

	For each pattern, you compute the forbidden leaf (the next term in the arithmetic progression) and
	remove it from the undetermined pile with `_reduceLeafSpace`.

	Parameters
	----------
	state : StateElimination
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

			if 0 <= leafForbidden < state.totalLeaves:
				permutationSpace = reduceLeafSpace(permutationSpace, pilesToUpdate, makeAntiChoicesLeaf(state.totalLeaves, [leafForbidden]))
				if not permutationSpace.valid:
					#=SIN= Early return.
					return permutationSpace

		if permutationSpace.leafCount < leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

# TODO The order of the functions can cause tests to fail. I don't think that ought to happen.
boxOfFunctionsReduction2上nDimensional: Sequence[Callable[[StateElimination, PermutationSpace], PermutationSpace]] = (
	_noConsecutiveDimensions2上nDimensional,
	_crossedCreases2上nDimensional,
	_conditionalPredecessors2上nDimensional,
	_headsBeforeTails2上nDimensional,
	reducePermutationSpace_nakedSubset,
	reducePermutationSpace_leafDomainOf0or1,
	_byCrease2上nDimensional,
	reducePermutationSpace_LeafIsPinned,
)
"""Select the ordered reduction functions for maps whose axis lengths are two.

(AI generated docstring)

You can pass this sequence to
`mapFolding._e.dataBaskets.StateElimination.reduceAllPermutationSpace` [1] to refine choice
foldings. Each function accepts an elimination state and one permutation space and returns the
resulting permutation space.

Reduction Order
---------------
The sequence checks consecutive dimensions, crossed creases, conditional predecessors, and
head-before-tail order, then applies naked-subset reduction [2], empty or singleton leaf domains
[3], crease constraints, and pinned-leaf propagation [4]. The order affects current results.

Examples
--------
From `easyRun/pinning.py`.

	```python
	state: StateElimination = StateElimination(
		(2,) * 5, boxOfFunctionsReduction=boxOfFunctionsReduction2上nDimensional
	)
	```

References
----------
[1] `mapFolding._e.dataBaskets.StateElimination.reduceAllPermutationSpace`

[2] `mapFolding._e.reduceIt.reducePermutationSpace_nakedSubset`

[3] `mapFolding._e.reduceIt.reducePermutationSpace_leafDomainOf0or1`

[4] `mapFolding._e.reduceIt.reducePermutationSpace_LeafIsPinned`
"""
