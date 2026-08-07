"""Generalized pinning functions in the "Elimination" algorithm for any `mapShape`.

Functions for 2^n-dimensional maps must go in other modules.
"""
from __future__ import annotations

from collections import Counter
from functools import partial
from gmpy2 import bit_clear
from humpy_cytoolz import (
	concat, get, groupby as toolz_groupby, itemfilter, keyfilter as filterPile, valfilter as filterLeaf, valfilter as filterLeafOptions,
	valfilter as filterValue)
from hunterMakesPy import errorL33T, inclusive, raiseIfNone
from itertools import chain, combinations, product as CartesianProduct
from mapFolding._e import (
	getIteratorOfLeaves, getLeafDomain, howManyLeavesInLeafOptions, leafOptionsAND, leafOptionsLeafNone, makeLeafAntiOptions)
from mapFolding._e.algorithms.iff import creaseViolation吗, getCreasePost, oddLeaf吗
from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
from mapFolding._e.filters import isLeafOptions吗, leafInLeafOptions吗, leafPinned吗
from mapFolding._e.theTypes import DimensionIndex, LeafOptions, PinnedLeaves
from more_itertools import extract, one
from operator import methodcaller
from typing import TYPE_CHECKING, TypeIs
from Z0Z_tools import between吗, DOTitems, DOTkeys, DOTvalues, reverseLookup, thisNotHaveThat吗

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator, Sequence
	from mapFolding._e.theTypes import Leaf, LeafSpace, Pile, UndeterminedPiles

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
		domain_k = getLeafDomain(state, leaf_k)
	for pile_k in reversed(tuple(domain_k)):
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

	state.removeCreaseViolations().reduceAllPermutationSpace()
	listPermutationSpaceUnchanged.extend(state.listPermutationSpace)
	state.listPermutationSpace = listExcludeLeaf_r
	del listExcludeLeaf_r, listPermutationSpace

	# DEVELOPMENT If I were to use `domainOf_leaf_r = domainOf_leaf_r or getLeafDomain(self,
	# leaf_r)`, then an empty `Iterable` would be replaced by `getLeafDomain(self, leaf_r)`. That
	# would be good if it prevents an error state. Or, allowing to pass an empty `Iterable` might
	# enable a no-op, which could be good. TODO make a conscious choice.
	if domainOf_leaf_r is None:
		domainOf_leaf_r = getLeafDomain(state, leaf_r)

	for pile_r in filter(between吗(0, pile_k - inclusive), domainOf_leaf_r):
		state.removeCreaseViolations().reduceAllPermutationSpace(quick=True)
		state.listPermutationSpace = excludeLeafAtPile(state.listPermutationSpace, leaf_r, pile_r)

	state.removeCreaseViolations().reduceAllPermutationSpace()

	state.listPermutationSpace.extend(listPermutationSpaceUnchanged)

	return state

def excludeLeafAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile) -> list[PermutationSpace]:
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

#======== Reducing `LeafOptions` ===============================
#-------- Shared logic -----------------------------------------

def reduceLeafSpace(
	permutationSpace: PermutationSpace
	, pilesToUpdate: Iterable[tuple[Pile, LeafOptions]]
	, leafAntiOptions: LeafOptions
) -> PermutationSpace:
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
		leafSpace: LeafSpace | None = leafOptionsLeafNone(leafOptionsAND(leafAntiOptions, leafOptions))
		if leafSpace is None:
			#=SIN= Early return.
			return PermutationSpace()
		else:
			permutationSpace[pile] = leafSpace
	return permutationSpace

#-------- Functions that use the shared logic -----------------------------------------

def isPileLeafOptions吗(pileLeafSpace: tuple[Pile, LeafSpace]) -> TypeIs[tuple[Pile, LeafOptions]]:
	return isLeafOptions吗(pileLeafSpace[1])

def _odd吗(state: EliminationState, dimension: int) -> Callable[[tuple[Pile, Leaf]], bool]:
	def workhorse(pileLeaf: tuple[Pile, Leaf]) -> bool:
		return bool(oddLeaf吗(state.mapShape, dimension=dimension, leaf=pileLeaf[1]))
	return workhorse

def _crossedCreases(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
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

	generators: list[CartesianProduct[tuple[DimensionIndex, PinnedLeaves, tuple[tuple[Pile, Leaf], tuple[Pile, Leaf]]]]] = []
	for dimension in range(state.dimensionsTotal):
		grouped: dict[bool, list[tuple[Pile, Leaf]]] = toolz_groupby(_odd吗(state, dimension), DOTitems(permutationSpace.extractPinnedLeaves()))
		parityEven: PinnedLeaves = dict(get(False, grouped, ()))
		parityOdd: PinnedLeaves = dict(get(True, grouped, ()))
		generators.append(CartesianProduct((dimension,), (parityOdd,), combinations(sorted(parityEven.items()), 2)))
		generators.append(CartesianProduct((dimension,), (parityEven,), combinations(sorted(parityOdd.items()), 2)))

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		for dimension, leavesPinnedParityOpposite, ((pileOf_k, leaf_k), (pileOf_r, leaf_r)) in concat(generators):
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
				leafAntiOptions: LeafOptions = makeLeafAntiOptions(state.leavesTotal, (leaf_rCrease,))

				if pileOf_k < pileOf_r < pileOf_kCrease:
					pilesForbidden = frozenset([*range(pileOf_k), *range(pileOf_kCrease + 1, state.pileLast + inclusive)])
				elif pileOf_kCrease < pileOf_r < pileOf_k:
					pilesForbidden = frozenset([*range(pileOf_kCrease), *range(pileOf_k + 1, state.pileLast + inclusive)])
				elif (pileOf_r < pileOf_kCrease < pileOf_k) or (pileOf_kCrease < pileOf_k < pileOf_r):
					pilesForbidden = range(pileOf_kCrease + 1, pileOf_k)
				elif (pileOf_r < pileOf_k < pileOf_kCrease) or (pileOf_k < pileOf_kCrease < pileOf_r):
					pilesForbidden = range(pileOf_k + 1, pileOf_kCrease)

			elif not leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
				leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, (leaf_kCrease,))

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
					return None
				continue

			else:
				continue

			if not (permutationSpace := reduceLeafSpace(permutationSpace
					, filter(isPileLeafOptions吗, extract(permutationSpace.items(), pilesForbidden))
					, leafAntiOptions
			)):
				#=SIN= Early return.
				return None

		if leafCount < permutationSpace.leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def reducePermutationSpace_LeafIsPinned(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
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
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		leavesPinned, pilesUndetermined = permutationSpace.bifurcate()
		#=EndNotes##walrus=
		if not (permutationSpace := reduceLeafSpace(
				permutationSpace, DOTitems(pilesUndetermined), makeLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned))
		)):
			#=SIN= Early return: an empty pile domain irreversibly invalidates the candidate.
			return None
		if len(leavesPinned) < permutationSpace.leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def reducePermutationSpace_nakedSubset(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
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
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		pilesUndetermined: UndeterminedPiles = permutationSpace.extractUndeterminedPiles()
		pilesUndetermined: UndeterminedPiles = filterLeafOptions(thisNotHaveThat吗(set(pilesUndetermined.values())), pilesUndetermined)

		groupByLeafOptions: dict[LeafOptions, set[Pile]] = {}
		for pile, leafOptions in pilesUndetermined.items():
			groupByLeafOptions.setdefault(leafOptions, set()).add(pile)

		for leafOptions, setPiles in DOTitems(itemfilter(lambda groupBy: (howManyLeavesInLeafOptions(groupBy[leafOptionsKey])) == len(groupBy[piles]), groupByLeafOptions)):

			if not (permutationSpace := reduceLeafSpace(permutationSpace
					, DOTitems(filterPile(thisNotHaveThat吗(setPiles), pilesUndetermined))
					, makeLeafAntiOptions(state.leavesTotal, getIteratorOfLeaves(leafOptions))
			)):
				#=SIN= Early return.
				return None

		if permutationSpace.leafCount < leafCount:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

#-------- Functions that do NOT use the shared logic -----------------------------------------

def reducePermutationSpace_leafDomainOf1(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
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
	updatedPermutationSpace : PermutationSpace | None
		The updated `permutationSpace` if valid; otherwise `None`.

	References
	----------
	[1] mapFolding._e.dataBaskets.PermutationSpace.atPilePinLeaf
	"""
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		leavesPinned, pilesUndetermined = permutationSpace.bifurcate()

		counterLeafDomainSize: Counter[Leaf] = Counter(chain(chain.from_iterable(map(getIteratorOfLeaves, DOTvalues(pilesUndetermined))), DOTvalues(leavesPinned)))

		if set(range(state.leavesTotal)).difference(counterLeafDomainSize.keys()):
			#=SIN= Early return: a leaf with no possible pile irreversibly invalidates the candidate.
			return None

		leavesWithDomainOf1: set[Leaf] = set(DOTkeys(filterValue((1).__eq__, counterLeafDomainSize))).difference(leavesPinned.values()).difference([state.leavesTotal])
		if leavesWithDomainOf1:
			leaf: Leaf = leavesWithDomainOf1.pop()
			sherpa: PermutationSpace | None = reducePermutationSpace_LeafIsPinned(state, permutationSpace.atPilePinLeaf(one(DOTkeys(filterLeaf(partial(leafInLeafOptions吗, leaf), pilesUndetermined))), leaf))
			if (sherpa is None) or (not sherpa):
				#=SIN= Early return: failed pin propagation irreversibly invalidates the candidate.
				return None
			else:
				permutationSpace = sherpa
			permutationSpaceHasNewLeaf = True
	return permutationSpace

listFunctionsReductionDEFAULT: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = (
	reducePermutationSpace_nakedSubset
	, reducePermutationSpace_leafDomainOf1
	, _crossedCreases
	, reducePermutationSpace_LeafIsPinned
)

listFunctionsReductionQuickDEFAULT: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = (
	reducePermutationSpace_nakedSubset
	, reducePermutationSpace_leafDomainOf1
	, reducePermutationSpace_LeafIsPinned
)
