"""Reduce leaf sets in elimination permutation spaces.

(AI generated docstring)

You can propagate pinned leaves, restrict repeated leaf sets, and detect leaves with no remaining pile
or only one remaining pile. Use the default reduction sequence to also restrict leaf choices using
crease crossings.
The specialized reduction sequence additionally applies dimension-order and conditional-predecessor
rules for maps whose axis lengths are two.

Contents
--------
Functions
	reduceLeafSpace
		Restrict specified pile domains with an allowed-leaf mask.
	reducePermutationSpace_LeafIsPinned
		Remove pinned leaves from the remaining pile leaf sets.
	reducePermutationSpace_leafDomainOf0or1
		Pin leaves with one remaining pile and invalidate leaves with no pile.
	reducePermutationSpace_nakedSubset
		Reserve identical leaf sets for matching groups of piles.
Reduction sequences
	boxOfFunctionsReduction2上nDimensional
		Access the ordered reductions for maps whose axis lengths are two.
	boxOfFunctionsReductionDEFAULT
		Access the default sequence of permutation-space reductions.
"""

from __future__ import annotations

from collections import Counter
from functools import partial
from gmpy2 import bit_flip
from humpy_cytoolz import dissoc, get, groupby as toolz_groupby, keyfilter as filterPile, valfilter as filterLeaf, valfilter as filterValue
from hunterMakesPy import errorL33T, inclusive, raiseIfNone
from itertools import chain, combinations
from mapFolding.algorithms.iff import creaseViolation吗, getCreasePost, oddLeaf吗
from mapFolding.beDRY import (
	choicesLeafAND, distinguishChoicesLeaf, getIteratorOfLeaves, lengthChoicesLeaf, makeAntiChoicesLeaf, mapShapeIs2上nDimensions)
from mapFolding.conditionalOrdering import getLeafPredecessors
from mapFolding.creases import getLeavesCreaseAnte, getLeavesCreasePost
from mapFolding.filters import (
	choicesLeaf吗, leafInChoicesLeaf吗, leafPinned吗, leaf吗, moreThanLeaf零吗, notPileLast, oddLeaf2上nDimensional吗, pileChoicesLeaf吗)
from mapFolding.measure import 工dimensionTail, 工dimension首零
from mapFolding.theTypes import ChoicesLeaf, Leaf, Pile
from more_itertools import extract, first, one, pairwise, triplewise
from typing import TYPE_CHECKING
from Z0Z_tools import DOTitems, DOTvalues, reverseLookup

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator, Sequence
	from mapFolding.dataBaskets import PermutationSpace, StateElimination
	from mapFolding.theTypes import LeafSpace, PinnedLeaves, UndeterminedPiles

def reduceLeafSpace(permutationSpace: PermutationSpace, pilesToUpdate: Iterable[tuple[Pile, ChoicesLeaf]], antiChoicesLeaf: ChoicesLeaf) -> PermutationSpace:
	"""Restrict specified pile domains with an allowed-leaf mask.

	(AI generated docstring)

	You can intersect each supplied leaf set with `antiChoicesLeaf` and write the normalized result into
	`permutationSpace`. You pin a leaf when only one remains. With an empty result, you set the validity
	flag to `False` and return immediately.

	Domain preservation
	-------------------
	You must supply a `ChoicesLeaf` that appropriately restricts each pile. Check membership in the existing
	dictionary value yourself; your supplied `ChoicesLeaf` is not intersected with that value.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Permutation space to mutate.
	pilesToUpdate : Iterable[tuple[Pile, ChoicesLeaf]]
		Pile indices and `ChoicesLeaf` values to intersect with the mask. The supplied `ChoicesLeaf` values, rather than the
		current dictionary values, determine the new choices.
	antiChoicesLeaf : ChoicesLeaf
		Allowed-leaf mask including the sentinel. Cleared bits identify excluded leaves.

	Returns
	-------
	permutationSpace : PermutationSpace
		The same object with updated pile domains or a false validity flag. Earlier updates remain applied on
		failure.
	"""
	for pile, choicesLeaf in pilesToUpdate:
		leafSpace: LeafSpace | None = distinguishChoicesLeaf(choicesLeafAND(antiChoicesLeaf, choicesLeaf))
		if leafSpace is None:
			#=SIN= Early return.
			permutationSpace.valid = False
			return permutationSpace
		else:
			permutationSpace[pile] = leafSpace
	return permutationSpace

def _odd吗(mapShape: tuple[int, ...], dimension: int) -> Callable[[tuple[Pile, Leaf]], bool]:
	def workhorse(pileLeaf: tuple[Pile, Leaf]) -> bool:
		return bool(oddLeaf吗(mapShape, dimension=dimension, leaf=pileLeaf[1]))
	return workhorse

def _crossedCreases(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Detect and eliminate crossed creases.

	(AI generated docstring)

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
					leaf_kCrease: Leaf | None = getCreasePost(state.mapShape, leaf_k, dimension)
					if leaf_kCrease is None:
						continue
					leaf_rCrease: Leaf | None = getCreasePost(state.mapShape, leaf_r, dimension)
					if leaf_rCrease is None:
						continue

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
							permutationSpace.valid = False
							#=SIN= Early return.
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

def reducePermutationSpace_LeafIsPinned(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Remove pinned leaves from the remaining pile leaf sets.

	(AI generated docstring)

	You can enforce one pile per pinned leaf by removing pinned leaf indices from undetermined piles. You
	repeat when an exclusion pins another leaf and stop when no additional leaf is pinned or the
	permutation space becomes invalid.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the total leaf count.
	permutationSpace : PermutationSpace
		Permutation space to mutate.

	Returns
	-------
	permutationSpace : PermutationSpace
		The same object with propagated assignments or a false validity flag.
	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf and permutationSpace.valid:
		permutationSpaceHasNewLeaf = False
		leavesPinned, pilesUndetermined = permutationSpace.bifurcate()
		permutationSpace = reduceLeafSpace(permutationSpace, DOTitems(pilesUndetermined), makeAntiChoicesLeaf(state.totalLeaves, DOTvalues(leavesPinned)))
		if len(leavesPinned) < permutationSpace.leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def reducePermutationSpace_nakedSubset(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Reserve identical leaf sets for matching groups of piles.

	(AI generated docstring)

	You can find groups of at least two piles with the same `ChoicesLeaf`. When the leaf count equals the
	pile count, you exclude those leaves from every other undetermined pile.

	Subset coverage
	---------------
	You compare identical `ChoicesLeaf` values rather than arbitrary unions of leaf sets. The current repeat condition
	requires the pinned-leaf count to decrease, so use `StateElimination.reduceAllPermutationSpace`
	for further propagation.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the total leaf count.
	permutationSpace : PermutationSpace
		Permutation space to mutate.

	Returns
	-------
	permutationSpace : PermutationSpace
		The same object with reduced leaf sets or a false validity flag.

	"""
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf and permutationSpace.valid:
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		pilesUndetermined: UndeterminedPiles = permutationSpace.undeterminedPiles()
		groupByChoicesLeaf: dict[ChoicesLeaf, set[Pile]] = {}
		for pile, choicesLeaf in pilesUndetermined.items():
			groupByChoicesLeaf.setdefault(choicesLeaf, set()).add(pile)

		groupByChoicesLeaf = filterValue(lambda boxOfPiles: 1 < len(boxOfPiles), groupByChoicesLeaf)
		for choicesLeaf, boxOfPiles in groupByChoicesLeaf.items():
			if lengthChoicesLeaf(choicesLeaf) == len(boxOfPiles):

				permutationSpace = reduceLeafSpace(permutationSpace, DOTitems(dissoc(pilesUndetermined, *boxOfPiles))
					, makeAntiChoicesLeaf(state.totalLeaves, getIteratorOfLeaves(choicesLeaf))
				)

		if permutationSpace.leafCount < leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def reducePermutationSpace_leafDomainOf0or1(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Pin leaves with one remaining pile and invalidate leaves with no pile.

	(AI generated docstring)

	You can count each leaf's occurrences across pinned assignments and leaf sets. You invalidate the
	permutation space for a missing leaf, or pin an unpinned leaf with one possible pile and propagate
	pinned-leaf exclusions before counting again.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the total leaf count.
	permutationSpace : PermutationSpace
		Permutation space to reduce. Pinning can create a replacement object.

	Returns
	-------
	permutationSpace : PermutationSpace
		Reduced permutation space, possibly a replacement object, with a false validity flag if any leaf has
		no pile.
	"""
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf and permutationSpace.valid:
		permutationSpaceHasNewLeaf = False

		leavesPinned, pilesUndetermined = permutationSpace.bifurcate()

		countDomainSizes: Counter[Leaf] = Counter(chain(chain.from_iterable(map(getIteratorOfLeaves, DOTvalues(pilesUndetermined))), DOTvalues(leavesPinned)))

		if set(range(state.totalLeaves)).difference(countDomainSizes.keys()):
			permutationSpace.valid = False
		else:
			leaf: Leaf | None = first(set(filterValue((1).__eq__, countDomainSizes, factory=dict[Leaf, int])).difference(leavesPinned.values()).difference([state.totalLeaves]), None)
			if leaf is not None:
				permutationSpace = reducePermutationSpace_LeafIsPinned(state, permutationSpace.atPilePinLeaf(one(filterLeaf(partial(leafInChoicesLeaf吗, leaf), pilesUndetermined, factory=dict[Leaf, ChoicesLeaf])), leaf))
				permutationSpaceHasNewLeaf = True
	return permutationSpace

boxOfFunctionsReductionDEFAULT: Sequence[Callable[[StateElimination, PermutationSpace], PermutationSpace]] = (
	reducePermutationSpace_nakedSubset
	, reducePermutationSpace_leafDomainOf0or1
	, _crossedCreases
	, reducePermutationSpace_LeafIsPinned
)
"""Access the default sequence of permutation-space reductions.

(AI generated docstring)

You can restrict matching leaf sets, pin leaves with a single pile domain, remove crease crossings,
and propagate pinned leaves. With the state reduction method, you apply the sequence in reverse
order and restart when dictionary values change.

See Also
--------
`mapFolding.dataBaskets.StateElimination.reduceAllPermutationSpace`
	Apply this reduction sequence repeatedly to the state's permutation spaces.
"""

#======== Reducing `ChoicesLeaf` ===============================

def _byCrease2上nDimensional(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Enforce crease adjacency constraints.

	(AI generated docstring)

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

	(AI generated docstring)

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

def _odd2上nDimensional吗(_mapShape: tuple[int, ...], dimension: int) -> Callable[[tuple[Pile, Leaf]], bool]:
	def workhorse(pileLeaf: tuple[Pile, Leaf]) -> bool:
		return bool(oddLeaf2上nDimensional吗(dimension, leaf=pileLeaf[1]))
	return workhorse

def _crossedCreases2上nDimensional(state: StateElimination, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""Detect and eliminate crossed creases.

	(AI generated docstring)

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
			groupedByParity: dict[bool, list[tuple[Pile, Leaf]]] = toolz_groupby(_odd2上nDimensional吗(state.mapShape, dimension), DOTitems(permutationSpace.pinnedLeaves()))

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

	(AI generated docstring)

	You can require leaves with large coordinates in a dimension (tail) to follow all leaves with small
	coordinates in that dimension (head). When a leaf with nonzero nearest head dimension is pinned, you
	remove leaves with larger coordinates in that dimension from preceding piles. When a leaf with
	nonzero nearest tail dimension is pinned, you remove leaves with smaller coordinates in that
	dimension from subsequent piles.

	Algorithm Details
	-----------------
	For each pinned leaf:

	1. Compute `工dimension首零(leaf)` to identify the dimension with the smallest coordinate
		magnitude from the head.
	2. If nonzero, remove all leaves with larger coordinates in that dimension from piles before
		`pile`.
	3. Compute `工dimensionTail(leaf)` to identify the dimension with the smallest coordinate
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

	(AI generated docstring)

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
`mapFolding.dataBaskets.StateElimination.reduceAllPermutationSpace` to refine choice
foldings. Each function accepts an elimination state and one permutation space and returns the
resulting permutation space.

See Also
--------
`mapFolding.dataBaskets.StateElimination.reduceAllPermutationSpace`
	Apply this reduction sequence repeatedly to the state's permutation spaces.

Reduction Order
---------------
The sequence checks consecutive dimensions, crossed creases, conditional predecessors, and
head-before-tail order, then applies naked-subset reduction, empty or singleton leaf domains,
crease constraints, and pinned-leaf propagation. The order affects current results.

Examples
--------
From `easyRun/pinning.py`.

	```python
	state: StateElimination = StateElimination(
		(2,) * 5, boxOfFunctionsReduction=boxOfFunctionsReduction2上nDimensional
	)
	```

"""
