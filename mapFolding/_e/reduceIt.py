from __future__ import annotations

from collections import Counter
from functools import partial
from humpy_cytoolz import (
	get, groupby as toolz_groupby, itemfilter, keyfilter as filterPile, valfilter as filterLeaf, valfilter as filterLeafOptions,
	valfilter as filterValue)
from hunterMakesPy import errorL33T, inclusive, raiseIfNone
from itertools import chain, combinations
from mapFolding import _e
from mapFolding._e.algorithms.iff import creaseViolation吗, getCreasePost, oddLeaf吗
from mapFolding._e.filters import isPileLeafOptions吗, leafInLeafOptions吗, leafPinned吗
from more_itertools import extract, first, one
from typing import TYPE_CHECKING
from Z0Z_tools import DOTitems, DOTvalues, reverseLookup, thisNotHaveThat吗

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Sequence
	from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
	from mapFolding._e.theTypes import Leaf, LeafOptions, LeafSpace, Pile, PinnedLeaves, UndeterminedPiles

def reduceLeafSpace(permutationSpace: PermutationSpace, pilesToUpdate: Iterable[tuple[Pile, LeafOptions]], leafAntiOptions: LeafOptions) -> PermutationSpace:
	"""Update permutation space by removing forbidden leaves from specified piles.

	(AI generated docstring)

	You can use this shared subroutine to update a `PermutationSpace` by applying leaf exclusion
	constraints to specified piles. The function intersects each pile's domain with the complement
	of forbidden leaves, normalizes the result to a single leaf when possible, and invalidates the
	entire permutation space if any pile's domain becomes empty.

	This function implements the mechanical update logic used by all constraint-propagation
	functions in the reduction system. Constraint encoders should call this function rather than
	modifying `permutationSpace` directly to ensure consistent domain updates, proper normalization
	via `leafOptionsLeafNone` [1], and early detection of unsatisfiable constraints.

	The `pilesToUpdate` parameter contains explicit `(pile, leafOptions)` tuples because constraint
	encoders may need to restrict a different domain than the current `permutationSpace[pile]` value.
	For example, when enforcing crease adjacency, the encoder provides the specific crease-neighbor
	options to intersect with `leafAntiOptions`, not the broader current domain at that pile.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary mapping pile indices to leaf indices or `LeafOptions`. The function
		mutates this dictionary in place.
	pilesToUpdate : Iterable[tuple[Pile, LeafOptions]]
		Pile indices to update and their corresponding leaf domains to restrict. Each tuple contains
		a pile index and the `LeafOptions` bitset representing the domain to intersect with
		`leafAntiOptions`. The provided `LeafOptions` may differ from `permutationSpace[pile]` when
		the constraint encoder needs to restrict against a computed subset.
	leafAntiOptions : LeafOptions
		Bitset representing forbidden leaves to exclude from all updated piles. The function computes
		the intersection of each pile's domain with the complement of this bitset.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace
		The mutated `permutationSpace` with updated pile domains, or an empty dictionary if any pile's
		domain becomes empty after applying the constraints.

	Constraint Propagation Architecture
	------------------------------------
	This function is the shared update subroutine for the constraint-propagation system orchestrated
	by `reduceAllPermutationSpace` [2]. All constraint encoders (`reducePermutationSpace_*` functions)
	call this function to perform domain updates. Constraint encoders should not modify
	`permutationSpace` directly; they should identify forbidden leaves, construct `leafAntiOptions`,
	and delegate the actual update to this function.

	The function enforces two critical invariants:
	1. Domain reduction: Every update shrinks or maintains pile domains; domains never expand.
	2. Early failure: If any domain becomes empty, the function immediately returns an empty
		dictionary, signaling that the permutation space is unsatisfiable.

	References
	----------
	[1] mapFolding._e.leafOptionsLeafNone

	[2] mapFolding._e.pinIt.reduceAllPermutationSpace

	[3] mapFolding._e.leafOptionsAND

	[4] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	"""
	for pile, leafOptions in pilesToUpdate:
		leafSpace: LeafSpace | None = _e.leafOptionsLeafNone(_e.leafOptionsAND(leafAntiOptions, leafOptions))
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

def _crossedCreases(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace:
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
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

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
			groupedByParity: dict[bool, list[tuple[Pile, Leaf]]] = toolz_groupby(_odd吗(state.mapShape, dimension), DOTitems(permutationSpace.extractPinnedLeaves()))

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
						leafAntiOptions: LeafOptions = _e.makeLeafAntiOptions(state.leavesTotal, (leaf_rCrease,))

						if pileOf_k < pileOf_r < pileOf_kCrease:
							pilesForbidden = frozenset([*range(pileOf_k), *range(pileOf_kCrease + 1, state.pileLast + inclusive)])
						elif pileOf_kCrease < pileOf_r < pileOf_k:
							pilesForbidden = frozenset([*range(pileOf_kCrease), *range(pileOf_k + 1, state.pileLast + inclusive)])
						elif (pileOf_r < pileOf_kCrease < pileOf_k) or (pileOf_kCrease < pileOf_k < pileOf_r):
							pilesForbidden = range(pileOf_kCrease + 1, pileOf_k)
						elif (pileOf_r < pileOf_k < pileOf_kCrease) or (pileOf_k < pileOf_kCrease < pileOf_r):
							pilesForbidden = range(pileOf_k + 1, pileOf_kCrease)

					elif not leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
						leafAntiOptions = _e.makeLeafAntiOptions(state.leavesTotal, (leaf_kCrease,))

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
							, filter(isPileLeafOptions吗, extract(permutationSpace.items(), pilesForbidden))
							, leafAntiOptions
					)
					if not permutationSpace.valid:
						#=SIN= Early return.
						return permutationSpace

		if leafCount < permutationSpace.leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def reducePermutationSpace_LeafIsPinned(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""I use this to propagate leaf pinning constraints.

	I use this constraint encoder to enforce that every pinned leaf can appear at only one pile. For
	every leaf pinned at a pile, I remove that leaf from `LeafOptions` at all other piles. When
	`LeafOptions` at a pile reduces to a single leaf, I convert `pile: leafOptions` to `pile: leaf`
	(pinning the leaf).

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace
		The updated `permutationSpace`.

	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf and permutationSpace.valid:
		permutationSpaceHasNewLeaf = False
		leavesPinned, pilesUndetermined = permutationSpace.bifurcate()
		permutationSpace = reduceLeafSpace(permutationSpace, DOTitems(pilesUndetermined), _e.makeLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned)))
		if len(leavesPinned) < permutationSpace.leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def reducePermutationSpace_nakedSubset(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""I use this to detect and exploit naked subset constraints.

	I use this constraint encoder to detect naked subsets in the permutation space and remove
	subset leaves from all other piles. A naked subset occurs when `n` piles share the same
	`LeafOptions` containing exactly `n` leaves. Those `n` leaves can only appear in those `n`
	piles, so I remove those leaves from `LeafOptions` at all other piles using `_reduceLeafSpace`.

	Algorithm Details
	-----------------
	The function implements a specialized naked subset detector optimized for high throughput:

	1. Extract `UndeterminedPiles` (piles with `LeafOptions`).
	2. Group piles by their `LeafOptions` values.
	3. Filter groups where the number of leaves in `LeafOptions` equals the number of piles sharing that `LeafOptions` (the naked subset criterion).
	4. For each naked subset, remove subset leaves from all other piles.

	The function iterates until no new leaves are pinned. The function is not a comprehensive
	naked subset solver; the function prioritizes high throughput for a strong return on
	investment.

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
	leafOptionsKey: int = 0
	piles: int = 1
	while permutationSpaceHasNewLeaf and permutationSpace.valid:
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		groupByLeafSpace: dict[LeafSpace, set[Pile]] = {}
		for pile, leafOptions in permutationSpace.items():
			groupByLeafSpace.setdefault(leafOptions, set()).add(pile)

		groupByLeafOptions: dict[LeafOptions, set[Pile]] = filterValue(lambda setPiles: 1 < len(setPiles), groupByLeafSpace)  # pyright: ignore[reportUnknownVariableType, reportUnknownLambdaType, reportUnknownArgumentType, reportAssignmentType] # ty: ignore[invalid-assignment]
		for leafOptions, setPiles in DOTitems(
			itemfilter(lambda groupBy: (_e.howManyLeavesInLeafOptions(groupBy[leafOptionsKey])) == len(groupBy[piles]), groupByLeafOptions)
		):
			pilesUndetermined: UndeterminedPiles = permutationSpace.extractUndeterminedPiles()
			# TODO Z0Z_tools, Fix valfilter annotations, then clean up this code.
			pilesUndetermined: UndeterminedPiles = filterLeafOptions(thisNotHaveThat吗(set(pilesUndetermined.values())), pilesUndetermined)

			permutationSpace = reduceLeafSpace(permutationSpace
				, DOTitems(filterPile(thisNotHaveThat吗(setPiles), pilesUndetermined))
				, _e.makeLeafAntiOptions(state.leavesTotal, _e.getIteratorOfLeaves(leafOptions))
			)

		if permutationSpace.leafCount < leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def reducePermutationSpace_leafDomainOf1(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace:
	"""I use this to detect and pin leaves with domain size one.

	I use this constraint encoder to detect leaves that can appear at only one pile (domain size one)
	and pin those leaves. I compute the domain size for each leaf by counting how many piles contain
	that leaf (either pinned or in `LeafOptions`). When a leaf appears at exactly one pile, I pin that
	leaf at that pile using `PermutationSpace.atPilePinLeaf` [1] and propagate the pinning using
	`reducePermutationSpace_leafDomainOf1`.

	The function also validates that every leaf has nonzero domain size. When any leaf has zero domain
	(cannot appear anywhere), I invalidate `permutationSpace` by returning `None`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace
		The updated `permutationSpace` if valid; otherwise with `valid` set to `False`.

	References
	----------
	[1] mapFolding._e.dataBaskets.PermutationSpace.atPilePinLeaf
	"""
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf and permutationSpace.valid:
		permutationSpaceHasNewLeaf = False

		leavesPinned, pilesUndetermined = permutationSpace.bifurcate()

		counterLeafDomainSize: Counter[Leaf] = Counter(chain(chain.from_iterable(map(_e.getIteratorOfLeaves, DOTvalues(pilesUndetermined))), DOTvalues(leavesPinned)))

		if set(range(state.leavesTotal)).difference(counterLeafDomainSize.keys()):
			permutationSpace.valid = False
		else:
			# TODO Z0Z_tools, fix valfilter annotations, then clean up this code.
			leaf: Leaf | None = first(set(filterValue((1).__eq__, counterLeafDomainSize)).difference(leavesPinned.values()).difference([state.leavesTotal]), None)  # pyright: ignore[reportUnknownArgumentType]
			if leaf is not None:
				permutationSpace = reducePermutationSpace_LeafIsPinned(state, permutationSpace.atPilePinLeaf(one(filterLeaf(partial(leafInLeafOptions吗, leaf), pilesUndetermined)), leaf))  # pyright: ignore[reportUnknownArgumentType]
				permutationSpaceHasNewLeaf = True
	return permutationSpace

boxOfFunctionsReductionDEFAULT: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace]] = (
	reducePermutationSpace_nakedSubset
	, reducePermutationSpace_leafDomainOf1
	, _crossedCreases
	, reducePermutationSpace_LeafIsPinned
)
boxOfFunctionsReductionQuickDEFAULT: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace]] = (
	reducePermutationSpace_nakedSubset
	, reducePermutationSpace_leafDomainOf1
	, reducePermutationSpace_LeafIsPinned
)
