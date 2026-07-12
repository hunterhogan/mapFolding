"""Generalized pinning functions in the "Elimination" algorithm for any `mapShape`.

Functions for 2^n-dimensional maps must go in other modules.

The development of this generalized module is severely hampered, however. Functions for 2^n-dimensional maps have a "beans and
cornbread" problem that was difficult for me to "solve"--due to my programming skills. If I were able to decouple the "beans and
cornbread" solution from the 2^n-dimensional functions, I would generalize more functions and move them here.
"""
from __future__ import annotations

from collections import Counter, deque
from functools import partial
from gmpy2 import bit_clear, bit_flip, bit_mask
from humpy_cytoolz import (
	assoc as associate, concat, curry as syntacticCurry, groupby as toolz_groupby, itemfilter, keyfilter as filterPile, unique,
	valfilter as filterLeaf, valfilter as filterLeafOptions, valfilter as filterValue)
# TODO One or more things is messed up with humpy_*toolz.*.map
from hunterMakesPy import errorL33T, inclusive, raiseIfNone
from itertools import chain, combinations, product as CartesianProduct, repeat
from mapFolding._e import (
	getIteratorOfLeaves, getLeafDomain, getLeafOptions, howManyLeavesInLeafOptions, JeanValjean, leafOptionsAND, makeLeafAntiOptions)
from mapFolding._e._2上nDimensional import dimensionNearest首
from mapFolding._e.algorithms.iff import creaseViolation吗, oddLeaf吗
from mapFolding._e.dataBaskets import PermutationSpace
from mapFolding._e.filters import leafInLeafOptions吗, leafPinned吗
from more_itertools import flatten, one
from typing import TYPE_CHECKING
from Z0Z_tools import between吗, DOTitems, DOTkeys, DOTvalues, reverseLookup, thisHasThat吗, thisNotHaveThat吗

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator, Sequence
	from mapFolding._e.dataBaskets import EliminationState
	from mapFolding._e.theTypes import DimensionIndex, Leaf, LeafOptions, LeafSpace, Pile, PinnedLeaves, UndeterminedPiles

#======== Boolean filters =======================

@syntacticCurry
def disqualifyPinningLeafAtPile(state: EliminationState, leaf: Leaf) -> bool:
	return any((
		state.permutationSpace.leafPinned吗(leaf)
		, state.permutationSpace.pilePinned吗(state.pile)
		, state.pile not in getLeafDomain(state, leaf)
	))

#======== Group by =======================

def segregateLeafPinnedAtPile(listPermutationSpace: Sequence[PermutationSpace], leaf: Leaf, pile: Pile) -> tuple[list[PermutationSpace], list[PermutationSpace]]:
	"""Partition `listPermutationSpace` into (notPinned, isPinned) groups for `leaf` pinned at `pile`.

	Parameters
	----------
	listPermutationSpace : Sequence[PermutationSpace]
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

#======== Pin a `Leaf` in a `PermutationSpace` or `Folding` =======================
# NOTE The ONLY valid way to pin a `Leaf` in a `PermutationSpace` or `Folding` is to call a method of `PermutationSpace`.

#======== Bulk modifications =======================

def deconstructListPermutationSpaceAtPile(listPermutationSpace: Iterable[PermutationSpace], pile: Pile, leavesToPin: Iterable[Leaf]) -> Iterator[PermutationSpace]:
	"""Expand every dictionary in `listPermutationSpace` at `pile` into all pinning variants.

	Applies `PermutationSpace.deconstructPermutationSpaceAtPile` element-wise, then flattens the nested value collections (each a mapping leaf -> dictionary)
	into a single list of dictionaries, discarding the intermediate keyed structure.

	Parameters
	----------
	listPermutationSpace : Iterable[PermutationSpace]
		Partial folding dictionaries.
	pile : int
		`pile` index to expand.
	leavesToPin : Iterable[int]
		`leaf` indices to pin at `pile`.

	Returns
	-------
	listPermutationSpace : Iterator[PermutationSpace]
		Flat iterator of expanded dictionaries covering all possible `leaf` assignments at `pile`.

	See Also
	--------
	PermutationSpace.deconstructPermutationSpaceAtPile
	"""
	return flatten(map(DOTvalues, map(PermutationSpace.deconstructAtPile, listPermutationSpace, repeat(pile), repeat(leavesToPin))))

# TODO Fix this moronic bullshit created by an AI assistant that refused to follow instructions.
def excludeLeaf_rBeforeLeaf_kAtPile_k(state: EliminationState, leaf_k: Leaf, leaf_r: Leaf, pile_k: Pile, domain_r: Iterable[Pile] | None = None, rangePile_k: Iterable[Leaf] | None = None) -> EliminationState:
	"""Exclude `leaf_r` from appearing before `leaf_k` at `pile_k`.

	Parameters
	----------
	state : EliminationState
		Mutable elimination state (provides `leavesTotal`, `pileLast`).
	leaf_k : int
		Reference leaf index derived from `productOfDimensions` for a dimension.
	leaf_r : int
		Leaf that must not appear before `leaf_k` (also dimension-derived).
	pile_k : int
		Pile index currently under consideration for leaf `leaf_k`.
	domain_r : Iterable[int] | None = None
		Optional domain of piles for leaf `leaf_r`. If `None`, the full domain from `state` is used.
	rangePile_k : Iterable[int] | None = None
		Optional range of leaves for pile `pile_k`. If `None`, the full range from `state` is used.

	Returns
	-------
	state : EliminationState
		Same state instance, mutated with updated `listPermutationSpace`.

	See Also
	--------
	excludeLeafRBeforeLeafK, theorem4, theorem2b
	"""
	listPermutationSpace: deque[PermutationSpace] = deque()

	if domain_r is None:
		domain_r = getLeafDomain(state, leaf_r)
	domain_r = tuple(filter(between吗(0, pile_k - inclusive), domain_r))

	if rangePile_k is None:
		rangePile_k = getIteratorOfLeaves(getLeafOptions(state, pile_k))
	rangePile_k = frozenset(rangePile_k)

	for permutationSpace in state.listPermutationSpace:
		listPermutationSpace_kPinnedAt_pile_k: list[PermutationSpace] = []
		listPermutationSpaceCompleted: list[PermutationSpace] = []

		if permutationSpace.leafPinnedAtPile吗(leaf_k, pile_k):
			listPermutationSpace_kPinnedAt_pile_k.append(permutationSpace)
		elif permutationSpace.leafPinned吗(leaf_k) or permutationSpace.pilePinned吗(pile_k) or leaf_k not in rangePile_k:
			listPermutationSpaceCompleted.append(permutationSpace)
		else:
			leafOptionsAt_pile_k: LeafOptions = raiseIfNone(permutationSpace.getLeafOptions(pile_k, default=bit_mask(len(permutationSpace))))
			if leafInLeafOptions吗(leaf_k, leafOptionsAt_pile_k):
				listPermutationSpace_kPinnedAt_pile_k.append(permutationSpace.atPilePinLeaf(pile_k, leaf_k))
				leafSpaceWithoutLeaf_k: Leaf | LeafOptions | None = JeanValjean(bit_clear(leafOptionsAt_pile_k, leaf_k))
				if leafSpaceWithoutLeaf_k is not None:
					listPermutationSpaceCompleted.append(PermutationSpace(associate(permutationSpace, pile_k, leafSpaceWithoutLeaf_k)))
					# listPermutationSpaceCompleted.append(associate(permutationSpace, pile_k, leafSpaceWithoutLeaf_k, factory=PermutationSpace))  # ruff:ignore[commented-out-code]
					# TODO `class PermutationSpace(dict[Pile, LeafSpace])` but Pylance and ty say:
					# Argument of type "dict[Pile, LeafSpace]" cannot be assigned to parameter "object" of type "PermutationSpace" in function "append"
					# 	"dict[Pile, LeafSpace]" is not assignable to "PermutationSpace"
					# Argument to bound method `list.append` is incorrect: Expected `PermutationSpace`, found `dict[Pile, int | mpz]`
			else:
				listPermutationSpaceCompleted.append(permutationSpace)

		iterator_kPinnedAt_pile_k: Iterable[PermutationSpace] = listPermutationSpace_kPinnedAt_pile_k
		for pile_r in domain_r:
			iterator_kPinnedAt_pile_k = excludeLeafAtPile(iterator_kPinnedAt_pile_k, leaf_r, pile_r, ())

		listPermutationSpace.extend(listPermutationSpaceCompleted)
		listPermutationSpace.extend(iterator_kPinnedAt_pile_k)

	state.listPermutationSpace = listPermutationSpace

	return state.reduceAllPermutationSpace(listFunctionsReduction)

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
		state = excludeLeaf_rBeforeLeaf_kAtPile_k(state, leaf_k, leaf_r, pile_k, domain_r=domain_r)
	return state

def excludeLeafAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile, leavesToPin: Iterable[Leaf]) -> Iterator[PermutationSpace]:
	"""Return a new list of pinned-leaves dictionaries that forbid `leaf` at `pile`.

	Parameters
	----------
	listPermutationSpace : Iterable[PermutationSpace]
		Collection of partial pinning dictionaries to transform.
	leaf : int
		`leaf` to exclude from `pile`.
	pile : int
		`pile` at which `leaf` must not be fixed.
	leavesToPin : Iterable[int]
		List of leaves available for pinning at `pile`. Don't include `leaf`.

	Yields
	------
	listPermutationSpace : Iterable[PermutationSpace]
		Expanded / filtered list respecting the exclusion constraint.

	See Also
	--------
	PermutationSpace.deconstructPermutationSpaceAtPile : Performs the expansion for one dictionary.
	requireLeafPinnedAtPile : Complementary operation that forces a `leaf` at a `pile`.
	"""
	#=SIN= Unused parameter: the shared bulk-modification callable contract requires `leavesToPin`.
	del leavesToPin

	for permutationSpace in listPermutationSpace:
		if permutationSpace.leafPinnedAtPile吗(leaf, pile):
			continue

		if (leafOptionsAtPile := permutationSpace.getLeafOptions(pile)) is None:
			yield permutationSpace
			continue

		if leafInLeafOptions吗(leaf, leafOptionsAtPile):
			leafSpaceWithoutLeaf: Leaf | LeafOptions | None = JeanValjean(bit_clear(leafOptionsAtPile, leaf))
			if leafSpaceWithoutLeaf is not None:
				yield PermutationSpace(associate(permutationSpace, pile, leafSpaceWithoutLeaf))
		else:
			yield permutationSpace

def requireLeafPinnedAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile) -> deque[PermutationSpace]:
	"""In every `PermutationSpace` dictionary, ensure `leaf`, and *only* `leaf`, is pinned at `pile`: excluding every other `leaf` at `pile`.

	Parameters
	----------
	listPermutationSpace : Iterable[PermutationSpace]
		Collection of partial pinning dictionaries to transform.
	leaf : int
		`leaf` required at `pile`.
	pile : int
		`pile` at which to pin the leaf.

	Returns
	-------
	listLeafAtPile : deque[PermutationSpace]
		`deque` of `PermutationSpace` dictionaries with `leaf` pinned at `pile`.

	See Also
	--------
	PermutationSpace.deconstructPermutationSpaceAtPile, excludeLeafAtPile
	"""
	listLeafAtPile: deque[PermutationSpace] = deque()

	for permutationSpace in listPermutationSpace:
		if permutationSpace.leafPinnedAtPile吗(leaf, pile):
			listLeafAtPile.append(permutationSpace)
		elif permutationSpace.leafPinned吗(leaf) or permutationSpace.pilePinned吗(pile):
			continue
		else:
			leafOptionsAtPile: LeafOptions = raiseIfNone(permutationSpace.getLeafOptions(pile, default=bit_mask(len(permutationSpace))))
			if leafInLeafOptions吗(leaf, leafOptionsAtPile):
				listLeafAtPile.append(permutationSpace.atPilePinLeaf(pile, leaf))

	return listLeafAtPile

def segregateLeafByDeconstructingListPermutationSpaceAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile, leavesToPin: Iterable[Leaf]) -> Iterator[tuple[PermutationSpace, tuple[PermutationSpace, ...]]]:
	for permutationSpace in listPermutationSpace:
		deconstructedPermutationSpaceAtPile: dict[Leaf, PermutationSpace] = permutationSpace.deconstructAtPile(pile, leavesToPin)
		leafPinnedAtPile: PermutationSpace = deconstructedPermutationSpaceAtPile.pop(leaf)
		yield (leafPinnedAtPile, tuple(deconstructedPermutationSpaceAtPile.values()))

#======== Reducing `LeafOptions` ===============================
#-------- Shared logic -----------------------------------------

#=SIN= Ruff suppression: the shared reduction callable contract requires the unused `state` parameter.
def reduceLeafSpace(
	state: EliminationState  # noqa: ARG001
	, permutationSpace: PermutationSpace
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
	via `JeanValjean` [1], and early detection of unsatisfiable constraints.

	The `pilesToUpdate` parameter contains explicit `(pile, leafOptions)` tuples because constraint
	encoders may need to restrict a different domain than the current `permutationSpace[pile]` value.
	For example, when enforcing crease adjacency, the encoder provides the specific crease-neighbor
	options to intersect with `leafAntiOptions`, not the broader current domain at that pile.

	Parameters
	----------
	state : EliminationState
		Data basket containing computed properties such as `leavesTotal`. Currently unused by the
		function but included for signature consistency with other reduction functions.
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
	[1] mapFolding._e.JeanValjean

	[2] mapFolding._e.pinIt.reduceAllPermutationSpace

	[3] mapFolding._e.leafOptionsAND

	[4] gmpy2 - Integer arithmetic
		https://gmpy2.readthedocs.io/en/latest/
	"""
	for pile, leafOptions in pilesToUpdate:
		leafSpace: LeafSpace | None = JeanValjean(leafOptionsAND(leafAntiOptions, leafOptions))
		if leafSpace is None:
			#=SIN= Early return: an empty pile domain irreversibly invalidates the candidate.
			return PermutationSpace()
		else:
			permutationSpace[pile] = leafSpace
	return permutationSpace

#-------- Functions that use the shared logic -----------------------------------------

def reducePermutationSpace_CrossedCreases(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
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
	#=SIN= Sentinel: type checkers cannot infer that pin-state guards precede every read of `pileOf_kCrease`.
	pileOf_kCrease: Pile = errorL33T
	#=SIN= Sentinel: type checkers cannot infer that pin-state guards precede every read of `pileOf_rCrease`.
	pileOf_rCrease: Pile = errorL33T
	pilesForbidden: Iterable[Pile] = []
	permutationSpaceHasNewLeaf: bool = True

	generators: deque[CartesianProduct[tuple[DimensionIndex, PinnedLeaves, tuple[tuple[Pile, Leaf], tuple[Pile, Leaf]]]]] = deque()
	for dimension in range(state.dimensionsTotal):
		parityEven: PinnedLeaves = {}
		parityOdd: PinnedLeaves = {}
		for pileLeaf in DOTitems(permutationSpace.extractPinnedLeaves()):
			if oddLeaf吗(state.mapShape, pileLeaf[1], dimension):
				parityOdd.update((pileLeaf,))
			else:
				parityEven.update((pileLeaf,))
		generators.append(CartesianProduct((dimension,), (parityOdd,), combinations(parityEven.items(), 2)))
		generators.append(CartesianProduct((dimension,), (parityEven,), combinations(parityOdd.items(), 2)))

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		leafCount: int = permutationSpace.leafCount

		for dimension, leavesPinnedParityOpposite, ((pileOf_k, leaf_k), (pileOf_r, leaf_r)) in concat(generators):
			leaf_kCrease: Leaf = int(bit_flip(leaf_k, dimension))
			leaf_rCrease: Leaf = int(bit_flip(leaf_r, dimension))

			if leaf_kCreaseIsPinned := leafPinned吗(leavesPinnedParityOpposite, leaf_kCrease):
				pileOf_kCrease = raiseIfNone(reverseLookup(permutationSpace, leaf_kCrease))
			if leaf_rCreaseIsPinned := leafPinned吗(leavesPinnedParityOpposite, leaf_rCrease):
				pileOf_rCrease = raiseIfNone(reverseLookup(permutationSpace, leaf_rCrease))

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
					#=SIN= Early return: crossed pinned creases irreversibly invalidate the candidate.
					return None
				continue

			else:  # elif not leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
				continue

			if not (permutationSpace := reduceLeafSpace(state, permutationSpace
					, DOTitems(filterPile(thisHasThat吗(pilesForbidden), permutationSpace.extractUndeterminedPiles()))
					, leafAntiOptions
			)):
				#=SIN= Early return: an empty pile domain irreversibly invalidates the candidate.
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
		# TODO create an "end notes" system to collect all general development notes in one place.
		# NOTE using the walrus operator here `if not (permutationSpace := _reduceLeafSpace...` means
		# that type checkers are ok with `permutationSpace: PermutationSpace`. If I assigned without
		# the `if` check, `permutationSpace = _reduceLeafSpace...`, then the annotation would need to
		# be `permutationSpace: PermutationSpace | None` because `_reduceLeafSpace` can return `None`.
		# Furthermore, not creating an intermediate variable is more efficient.
		if not (permutationSpace := reduceLeafSpace(
				state, permutationSpace, DOTitems(pilesUndetermined), makeLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned))
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

		groupByLeafOptions: dict[LeafOptions, set[Pile]] = {}
		for pile, leafOptions in DOTitems(filterLeafOptions(thisNotHaveThat吗(unique(pilesUndetermined.values())), pilesUndetermined)):
			groupByLeafOptions.setdefault(leafOptions, set()).add(pile)

		for leafOptions, setPiles in DOTitems(itemfilter(lambda groupBy: (howManyLeavesInLeafOptions(groupBy[leafOptionsKey])) == len(groupBy[piles]), groupByLeafOptions)):

			if not (permutationSpace := reduceLeafSpace(state, permutationSpace
					, DOTitems(filterPile(thisNotHaveThat吗(setPiles), pilesUndetermined))
					, makeLeafAntiOptions(state.leavesTotal, getIteratorOfLeaves(leafOptions))
			)):
				#=SIN= Early return: an empty pile domain irreversibly invalidates the candidate.
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
			sherpa: PermutationSpace | None = reducePermutationSpace_LeafIsPinned(state, permutationSpace.atPilePinLeaf(one(DOTkeys(filterLeaf(leafInLeafOptions吗(leaf), pilesUndetermined))), leaf))
			if (sherpa is None) or (not sherpa):
				#=SIN= Early return: failed pin propagation irreversibly invalidates the candidate.
				return None
			else:
				permutationSpace = sherpa
			permutationSpaceHasNewLeaf = True
	return permutationSpace

listFunctionsReduction: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = (
	reducePermutationSpace_LeafIsPinned,
	reducePermutationSpace_leafDomainOf1,
	reducePermutationSpace_nakedSubset,
	# TODO I cannot think of a reason why this ought not to be a general function. So, the
	# test failures suggest an implementation error or bug. A001415(3) and A001416(2), which
	# happen to be the same mapShape, (2, 3).
	reducePermutationSpace_CrossedCreases,
)
