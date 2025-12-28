from operator import getitem
from collections.abc import Callable, Iterable, Iterator
from cytoolz.dicttoolz import assoc as associate, valfilter as leafFilter
from cytoolz.functoolz import curry as syntacticCurry
from cytoolz.itertoolz import groupby as toolz_groupby
from itertools import repeat
from mapFolding import inclusive
from mapFolding._e import (
	between, DOTvalues, Folding, get_mpzAntiPileRangeOfLeaves, getLeaf, getLeafDomain, getPileRange, leafIsNotPinned,
	LeafOrPileRangeOfLeaves, oopsAllLeaves, PermutationSpace, pileIsOpen, pileRangeOfLeavesAND, thisIsALeaf)
from mapFolding._e.dataBaskets import EliminationState
from more_itertools import flatten, map_if

# ======= Boolean filters =======================

@syntacticCurry
def atPilePinLeafSafetyFilter(leavesPinned: PermutationSpace, pile: int, leaf: int) -> bool:
	"""Return `True` if it is safe to call `atPilePinLeafSafetyFilter(leavesPinned, pile, leaf)`.

	For performance, you probably can and probably *should* create a set of filters for your circumstances.

	Parameters
	----------
	leavesPinned : PermutationSpace
		A mapping of each `pile` with a pinned `leaf`.
	pile : int
		`pile` at which to pin.
	leaf : int
		`leaf` to pin.

	Returns
	-------
	isSafeToPin : bool
		True if it is safe to pin `leaf` at `pile` in `leavesPinned`.
	"""
	return isPinnedAtPile(leavesPinned, leaf, pile) or (pileIsOpen(leavesPinned, pile) and leafIsNotPinned(leavesPinned, leaf))

@syntacticCurry
def isPinnedAtPile(leavesPinned: PermutationSpace, leaf: int, pile: int) -> bool:
	"""Return `True` if `leaf` is presently pinned at `pile` in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` whose presence at `pile` is being checked.
	pile : int
		`pile` index.

	Returns
	-------
	leafIsPinnedAtPile : bool
		True if the mapping includes `pile: leaf`.
	"""
	return leaf == leavesPinned.get(pile)

# ======= Group by =======================

def _segregateLeafPinnedAtPile(listPermutationSpace: list[PermutationSpace], leaf: int, pile: int) -> tuple[list[PermutationSpace], list[PermutationSpace]]:
	"""Partition `listPermutationSpace` into (notPinned, isPinned) groups for `leaf` pinned at `pile`.

	Parameters
	----------
	listPermutationSpace : list[PermutationSpace]
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
	isPinned: Callable[[PermutationSpace], bool] = isPinnedAtPile(leaf=leaf, pile=pile)
	grouped: dict[bool, list[PermutationSpace]] = toolz_groupby(isPinned, listPermutationSpace)
	return (grouped.get(False, []), grouped.get(True, []))

# ======= Pin one or more leaves in `leavesPinned` or `folding` =======================
# NOTE The only functions that actually pin a leaf in a dictionary or folding ought to be in this section.

@syntacticCurry
def atPilePinLeaf(leavesPinned: PermutationSpace, pile: int, leaf: int) -> PermutationSpace:
	"""Return a new `PermutationSpace` with `leaf` pinned at `pile` without modifying `leavesPinned`.

	Warning
	-------
	This function assumes either 1. `leaf` is not pinned and `pile` is open or 2. `leaf` is pinned at `pile`. Overwriting a
	different `leaf` pinned at `pile` corrupts the permutation space.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Dictionary of `pile` with pinned `leaf` or pile-range of leaves, if a `leaf` is pinned at `pile` or the pile-range of
		leaves is defined.
	pile : int
		`pile` at which to pin `leaf`.
	leaf : int
		`leaf` to pin.

	Returns
	-------
	dictionaryLeavesPinned : PermutationSpace
		New dictionary with `pile` mapped to `leaf`.

	See Also
	--------
	deconstructLeavesPinned
	"""
	return associate(leavesPinned, pile, leaf)

# TODO more flexible.
def makeFolding(leavesPinned: PermutationSpace, leavesToInsert: tuple[int, ...]) -> Folding:
	permutand: Iterator[int] = iter(leavesToInsert)
	pilesTotal: int = len(oopsAllLeaves(leavesPinned)) + len(leavesToInsert)
	return tuple([
		leafOrLeafRange if (leafOrLeafRange := getLeaf(leavesPinned, pile)) else next(permutand)
		for pile in range(pilesTotal)
	])

# ======= Deconstruct a `PermutationSpace` dictionary =======
# This function returns `deconstructedLeavesPinned : dict[int, LeavesPinned]` for pragmatic reasons. I typically deconstruct a
# pile because I want to pin one leaf at the pile without invalidating the permutation space. So, the dictionary makes it easy for
# me to segregate the one leaf I am working on from the umpteen other new dictionaries, each with a leaf pinned at the pile.
# If I change to representing the range of a pile with "bit packing", then I only need to make two dictionaries: one dictionary
# with the desired leaf pinned at pile, and another dictionary with a modified range at pile that excludes the desired leaf.
# The two systems are complementary: I can completely deconstruct a pile into only pinned leaves, bifurcate into one pinned
# dictionary and one pile-range dictionary, or any intermediate option.
def deconstructPermutationSpaceAtPile(leavesPinned: PermutationSpace, pile: int, leavesToPin: Iterable[int]) -> dict[int, PermutationSpace]:
	"""Deconstruct an open `pile` to the `leaf` range of `pile`.

	Return a dictionary of `PermutationSpace` with either `leavesPinned` because it already has a `leaf` pinned at `pile` or one
	`PermutationSpace` for each `leaf` in `leavesToPin` pinned at `pile`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Dictionary to divide and replace.
	pile : int
		`pile` at which to pin a `leaf`.
	leavesToPin : list[int]
		List of `leaves` to pin at `pile`.

	Returns
	-------
	deconstructedLeavesPinned : dict[int, LeavesPinned]
		Dictionary mapping from `leaf` pinned at `pile` to the `PermutationSpace` dictionary with the `leaf` pinned at `pile`.
	"""
	if thisIsALeaf(leaf := leavesPinned.get(pile)):
		deconstructedLeavesPinned: dict[int, PermutationSpace] = {leaf: leavesPinned}
	else:
		pin: Callable[[int], PermutationSpace] = atPilePinLeaf(leavesPinned, pile)
		leafCanBePinned: Callable[[int], bool] = leafIsNotPinned(leavesPinned)
		deconstructedLeavesPinned = {leaf: pin(leaf) for leaf in filter(leafCanBePinned, leavesToPin)}
	return deconstructedLeavesPinned

def deconstructPermutationSpaceByDomainOfLeaf(leavesPinned: PermutationSpace, leaf: int, leafDomain: Iterable[int]) -> list[PermutationSpace]:
	"""Pin `leaf` at each open `pile` in the domain of `leaf`.

	Return a `list` of `PermutationSpace` with either `leavesPinned` because `leaf` is already pinned or one `PermutationSpace` for each
	open `pile` in `leafDomain` with leaf pinned at `pile`.

	Parameters
	----------
	leavesPinned : PermutationSpace
		Dictionary to divide and replace.
	leaf : int
		`leaf` to pin.
	leafDomain : Iterable[int]
		Domain of `pile` indices for `leaf`.

	Returns
	-------
	deconstructedLeavesPinned : list[PermutationSpace]
		List of `PermutationSpace` dictionaries with `leaf` pinned at each open `pile` in `leafDomain`.
	"""
	if leafIsNotPinned(leavesPinned, leaf):
		pinLeafAt: Callable[[int], PermutationSpace] = atPilePinLeaf(leavesPinned, leaf=leaf)
		pileAvailable: Callable[[int], bool] = pileIsOpen(leavesPinned)
		deconstructedLeavesPinned: list[PermutationSpace] = list(map(pinLeafAt, filter(pileAvailable, leafDomain)))
	else:
		deconstructedLeavesPinned = [leavesPinned]
	return deconstructedLeavesPinned

def deconstructPermutationSpaceByDomainsCombined(leavesPinned: PermutationSpace, leaves: tuple[int, ...], leavesDomain: Iterable[tuple[int, ...]]) -> list[PermutationSpace]:
	"""Prototype."""
	deconstructedLeavesPinned: list[PermutationSpace] = []

	def pileOpenByIndex(index: int) -> Callable[[tuple[int, ...]], bool]:
		def workhorse(domain: tuple[int, ...]) -> bool:
			return pileIsOpen(leavesPinned, domain[index])
		return workhorse

	def isPinnedAtPileByIndex(leaf: int, index: int) -> Callable[[tuple[int, ...]], bool]:
		def workhorse(domain: tuple[int, ...]) -> bool:
			return isPinnedAtPile(leavesPinned, leaf, domain[index])
		return workhorse

	if any(map(leafIsNotPinned(leavesPinned), leaves)):
		for index in range(len(leaves)):
			"""Redefine leavesDomain by filtering out domains that are not possible with the current `PermutationSpace`."""
			if leafIsNotPinned(leavesPinned, leaves[index]):
				"""`leaves[index]` is not pinned, so it needs a pile.
				In each iteration of `leavesDomain`, `listOfPiles`, the pile it needs is `listOfPiles[index]`.
				Therefore, if `listOfPiles[index]` is open, filter in the iteration. If `listOfPiles[index]` is occupied, filter out the iteration."""
				leavesDomain = filter(pileOpenByIndex(index), leavesDomain)
			else:
				"""`leaves[index]` is pinned.
				In each iteration of `leavesDomain`, `listOfPiles`, the pile in which `leaves[index]` is pinned must match `listOfPiles[index]`.
				Therefore, if the pile in which `leaves[index]` is pinned matches `listOfPiles[index]`, filter in the iteration. Otherwise, filter out the iteration."""
				leavesDomain = filter(isPinnedAtPileByIndex(leaves[index], index), leavesDomain)

		for listOfPiles in leavesDomain:
			"""Properly and safely deconstruct `leavesPinned` by the combined domain of leaves.
			The parameter `leavesDomain` is the full domain of the leaves, so deconstructing with `leavesDomain` preserves the permutation space.
			For each leaf in leaves, I filter out occupied piles, so I will not overwrite any pinned leaves--that would invalidate the permutation space.
			I apply filters that prevent pinning the same leaf twice.
			Therefore, for each domain in `leavesDomain`, I can safely pin `leaves[index]` at `listOfPiles[index]` without corrupting the permutation space."""
			leavesPinnedForListOfPiles: PermutationSpace = leavesPinned.copy()
			for index in range(len(leaves)):
				leavesPinnedForListOfPiles = atPilePinLeaf(leavesPinnedForListOfPiles, listOfPiles[index], leaves[index])
			deconstructedLeavesPinned.append(leavesPinnedForListOfPiles)
	else:
		deconstructedLeavesPinned = [leavesPinned]

	return deconstructedLeavesPinned

# ======= Bulk modifications =======================

def deconstructListPermutationSpaceAtPile(listPermutationSpace: Iterable[PermutationSpace], pile: int, leavesToPin: Iterable[int]) -> Iterator[PermutationSpace]:
	"""Expand every dictionary in `listPermutationSpace` at `pile` into all pinning variants.

	Applies `deconstructLeavesPinned` element-wise, then flattens the nested value collections (each a mapping leaf -> dictionary)
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
	deconstructLeavesPinned
	"""
	return  flatten(map(DOTvalues, map(deconstructPermutationSpaceAtPile, listPermutationSpace, repeat(pile), repeat(leavesToPin))))

def excludeLeaf_rBeforeLeaf_kAtPile_k(state: EliminationState, k: int, r: int, pile_k: int, domain_r: Iterable[int] | None = None, rangePile_k: Iterable[int] | None = None) -> EliminationState:
	"""Exclude leaf `r` before leaf `k` at pile `k`.

	Parameters
	----------
	state : EliminationState
		Mutable elimination state (provides `leavesTotal`, `pileLast`).
	k : int
		Reference leaf index derived from `productOfDimensions` for a dimension.
	r : int
		Leaf that must not appear before `k` (also dimension-derived).
	pile_k : int
		Pile index currently under consideration for leaf `k`.
	domain_r : Iterable[int] | None = None
		Optional domain of piles for leaf `r`. If `None`, the full domain from `state` is used.
	rangePile_k : Iterable[int] | None = None
		Optional range of leaves for pile `k`. If `None`, the full range from `state` is used.

	Returns
	-------
	state : EliminationState
		Same state instance, mutated with updated `listPermutationSpace`.

	See Also
	--------
	excludeLeafRBeforeLeafK, theorem4, theorem2b
	"""
	iterator_kPinnedAt_pile_k: Iterable[PermutationSpace] = []
	iterable_pile_kOccupied: Iterable[PermutationSpace] = []
	listPermutationSpace: list[PermutationSpace] = []
	listPermutationSpaceCompleted: list[PermutationSpace] = []

	pile_kIsOpen: Callable[[PermutationSpace], bool] = pileIsOpen(pile=pile_k)
	kIsNotPinned: Callable[[PermutationSpace], bool] = leafIsNotPinned(leaf=k)
	def notLeaf_r(comparand: int, r: int = r) -> bool:
		return comparand != r

	listPermutationSpace, iterator_kPinnedAt_pile_k = _segregateLeafPinnedAtPile(state.listPermutationSpace, k, pile_k)
	state.listPermutationSpace = []

	grouped: dict[bool, list[PermutationSpace]] = toolz_groupby(kIsNotPinned, listPermutationSpace)
	listPermutationSpaceCompleted, listPermutationSpace = grouped.get(False, []), grouped.get(True, [])
	grouped = toolz_groupby(pile_kIsOpen, listPermutationSpace)
	iterable_pile_kOccupied, listPermutationSpace = grouped.get(False, []), grouped.get(True, [])
	listPermutationSpaceCompleted.extend(iterable_pile_kOccupied)

	if domain_r is None:
		domain_r = getLeafDomain(state, r)

	domain_r = filter(between(0, pile_k - inclusive), domain_r)

	if rangePile_k is None:
		rangePile_k = getPileRange(state, pile_k)

	rangePile_k = frozenset(rangePile_k)

	if k in rangePile_k:
		for leavesPinned_kPinnedAt_pile_k, iterable_pile_kOccupied in segregateLeafByDeconstructingListPermutationSpaceAtPile(listPermutationSpace, k, pile_k, rangePile_k):
			listPermutationSpaceCompleted.extend(iterable_pile_kOccupied)
			iterator_kPinnedAt_pile_k.append(leavesPinned_kPinnedAt_pile_k)
	else:
		listPermutationSpaceCompleted.extend(listPermutationSpace)

	for pile_r in domain_r:
		iterator_kPinnedAt_pile_k = excludeLeafAtPile(iterator_kPinnedAt_pile_k, r, pile_r, tuple(filter(notLeaf_r, getPileRange(state, pile_r))))

	state.listPermutationSpace.extend(listPermutationSpaceCompleted)
	state.listPermutationSpace.extend(iterator_kPinnedAt_pile_k)

	return state

def excludeLeaf_rBeforeLeaf_k(state: EliminationState, k: int, r: int, domain_k: Iterable[int] | None = None, domain_r: Iterable[int] | None = None) -> EliminationState:
	"""Apply a `leaf` ordering exclusion (`r` cannot precede `k`) at every `pile`.

	Parameters
	----------
	state : EliminationState
		Data basket, state of the local context, and state of the global context.
	k : int
		`leaf` that must be in a `pile` preceding the `pile` of `r`.
	r : int
		`leaf` that must be in a `pile` succeeding the `pile` of `k`.
	domain_k : Iterable[int] | None = None
		The domain of each `pile` at which `k` can be pinned. If `None`, every `pile` is in the domain.
	domain_r : Iterable[int] | None = None
		The domain of each `pile` at which `r` can be pinned. If `None`, every `pile` is in the domain.

	Returns
	-------
	EliminationState
		Same state instance, mutated with updated `listPermutationSpace`.

	See Also
	--------
	_excludeLeafRBeforeLeafK, theorem4, theorem2b
	"""
	if domain_k is None:
		domain_k = getLeafDomain(state, k)
	for pile_k in reversed(tuple(domain_k)):
		state = excludeLeaf_rBeforeLeaf_kAtPile_k(state, k, r, pile_k, domain_r=domain_r)
	return state

def excludeLeafAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: int, pile: int, leavesToPin: Iterable[int]) -> Iterator[PermutationSpace]:
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

	Returns
	-------
	listPermutationSpace : Iterable[PermutationSpace]
		Expanded / filtered list respecting the exclusion constraint.

	See Also
	--------
	deconstructLeavesPinned : Performs the expansion for one dictionary.
	pinLeafAtPile : Complementary operation that forces a leaf at a pile.
	"""
	return deconstructListPermutationSpaceAtPile(getitem(_segregateLeafPinnedAtPile(list(listPermutationSpace), leaf, pile), 0), pile, leavesToPin)

def requireLeafPinnedAtPile(listPermutationSpace: list[PermutationSpace], leaf: int, pile: int) -> list[PermutationSpace]:
	"""In every `PermutationSpace` dictionary, ensure `leaf`, and *only* `leaf`, is pinned at `pile`: excluding every other `leaf` at `pile`.

	Parameters
	----------
	listPermutationSpace : list[PermutationSpace]
		Collection of partial pinning dictionaries to transform.
	leaf : int
		`leaf` required at `pile`.
	pile : int
		`pile` at which to pin the leaf.

	Returns
	-------
	listLeafAtPile : list[PermutationSpace]
		`list` of `PermutationSpace` dictionaries with `leaf` pinned at `pile`.

	See Also
	--------
	deconstructPermutationSpace, excludeLeafAtPile
	"""
	listPermutationSpace, listLeafAtPile = _segregateLeafPinnedAtPile(listPermutationSpace, leaf, pile)
	listLeafAtPile.extend(map(atPilePinLeaf(pile=pile, leaf=leaf), filter(pileIsOpen(pile=pile), filter(leafIsNotPinned(leaf=leaf), listPermutationSpace))))
	return listLeafAtPile

def segregateLeafByDeconstructingListPermutationSpaceAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: int, pile: int, leavesToPin: Iterable[int]) -> Iterator[tuple[PermutationSpace, tuple[PermutationSpace, ...]]]:
	for leavesPinned in listPermutationSpace:
		deconstructedLeavesPinnedAtPile: dict[int, PermutationSpace] = deconstructPermutationSpaceAtPile(leavesPinned, pile, leavesToPin)
		leafPinnedAtPile: PermutationSpace = deconstructedLeavesPinnedAtPile.pop(leaf)
		yield (leafPinnedAtPile, tuple(deconstructedLeavesPinnedAtPile.values()))

