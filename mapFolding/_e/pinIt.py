from collections.abc import Callable, Iterable, Iterator
from cytoolz.dicttoolz import assoc as associate, valfilter as leafFilter
from cytoolz.functoolz import curry as syntacticCurry
from cytoolz.itertoolz import groupby as toolz_groupby
from gmpy2 import bit_mask, xmpz
from hunterMakesPy import raiseIfNone
from itertools import repeat
from mapFolding import between, DOTvalues, inclusive, LeafOrPileRangeOfLeaves, PinnedLeaves
from mapFolding._e import getLeafDomain, getPileRange, 零
from mapFolding.dataBaskets import EliminationState
from more_itertools import flatten
from operator import iand
from typing import TypeGuard

# ======= Boolean filters =======================

@syntacticCurry
def atPilePinLeafSafetyFilter(leavesPinned: PinnedLeaves, pile: int, leaf: int) -> bool:
	"""Return `True` if it is safe to call `atPilePinLeafSafetyFilter(leavesPinned, pile, leaf)`.

	For performance, you probably can and probably *should* create a set of filters for your circumstances.

	Parameters
	----------
	leavesPinned : PinnedLeaves
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
def isPinnedAtPile(leavesPinned: PinnedLeaves, leaf: int, pile: int) -> bool:
	"""Return `True` if `leaf` is presently pinned at `pile` in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PinnedLeaves
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

@syntacticCurry
def leafIsNotPinned(leavesPinned: PinnedLeaves, leaf: int) -> bool:
	"""Return True if `leaf` is not presently pinned in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PinnedLeaves
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` index.

	Returns
	-------
	leafIsNotPinned : bool
		True if the mapping does not include `leaf`.
	"""
	return leaf not in leavesPinned.values()

@syntacticCurry
def leafIsPinned(leavesPinned: PinnedLeaves, leaf: int) -> bool:
	"""Return True if `leaf` is pinned in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PinnedLeaves
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` index.

	Returns
	-------
	leafIsPinned : bool
		True if the mapping includes `leaf`.
	"""
	return leaf in leavesPinned.values()

def notLeafOriginOrLeaf零(leaf: int) -> bool:
	return 零 < leaf

@syntacticCurry
def notPileLast(pileLast: int, pile: int) -> bool:
	"""Return True if `pile` is not the last pile.

	Parameters
	----------
	pileLast : int
		Index of the last pile.
	pile : int
		`pile` index.

	Returns
	-------
	notPileLast : bool
		True if `pile` is not equal to `pileLast`.
	"""
	return pileLast != pile

@syntacticCurry
def pileIsOpen(leavesPinned: PinnedLeaves, pile: int) -> bool:
	"""Return True if `pile` is not presently pinned in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PinnedLeaves
		Partial folding mapping from pile -> leaf.
	pile : int
		`pile` index.

	Returns
	-------
	pileIsOpen : bool
		True if either `pile` is not a key in `leavesPinned` or `leavesPinned[pile]` is a pile-range (`xmpz`).
	"""
	return not thisIsALeaf(leavesPinned.get(pile))

# ======= Group by =======================

def _segregateLeafPinnedAtPile(listPinnedLeaves: list[PinnedLeaves], leaf: int, pile: int) -> tuple[list[PinnedLeaves], list[PinnedLeaves]]:
	"""Partition `listPinnedLeaves` into (notPinned, isPinned) groups for `leaf` pinned at `pile`.

	Parameters
	----------
	listPinnedLeaves : list[PinnedLeaves]
		Collection of partial folding dictionaries.
	leaf : int
		`leaf` to test.
	pile : int
		`pile` index.

	Returns
	-------
	segregatedLists : tuple[list[PinnedLeaves], list[PinnedLeaves]]
		First element: dictionaries where `leaf` is NOT pinned at `pile`.
		Second element: dictionaries where `leaf` IS pinned at `pile`.
	"""
	isPinned: Callable[[PinnedLeaves], bool] = isPinnedAtPile(leaf=leaf, pile=pile)
	grouped: dict[bool, list[PinnedLeaves]] = toolz_groupby(isPinned, listPinnedLeaves)
	return (grouped.get(False, []), grouped.get(True, []))

# ======= Working with variables that are a leaf or a pile's domain of leaves =======================
# https://gmpy2.readthedocs.io/en/latest/advmpz.html

def getLeaf(leavesPinned: PinnedLeaves, pile: int, default: int | None = None) -> int | None:
	if thisIsALeaf(ImaLeaf := leavesPinned.get(pile)):
		return ImaLeaf
	return default

def getIteratorOfLeaves(pileRangeOfLeaves: xmpz) -> Iterator[int]:
	"""Return an iterator of leaves in `pileRangeOfLeaves`.

	Parameters
	----------
	pileRangeOfLeaves : xmpz
		`xmpz` representing the range of leaves in a pile.

	Returns
	-------
	iteratorOfLeaves : Iterator[int]
		Iterator of `leaves` in `pileRangeOfLeaves`.
	"""
	pileRangeOfLeaves[-1] = 0
	return pileRangeOfLeaves.iter_set()

def getXmpzAntiPileRange(leavesTotal: int, leaves: Iterable[int]) -> xmpz:
	antiPileRange = xmpz(bit_mask(leavesTotal))
	for leaf in leaves:
		antiPileRange[leaf] = 0
	return antiPileRange

def getXmpzPileRangeOfLeaves(leavesTotal: int, leaves: Iterable[int]) -> xmpz:
	pileRangeOfLeaves: xmpz = xmpz(0)
	pileRangeOfLeaves[leavesTotal] = 1
	for leaf in leaves:
		pileRangeOfLeaves[leaf] = 1
	return pileRangeOfLeaves

def thisIsALeaf(leafOrPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None) -> TypeGuard[int]:
	"""Return True if `leafOrPileRangeOfLeaves` is a `leaf`.

	Parameters
	----------
	leafOrPileRangeOfLeaves : LeafOrPileRangeOfLeaves | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	intIsProbablyALeaf : TypeGuard[int]
		Technically, we only know the type is `int`.
	"""
	return (leafOrPileRangeOfLeaves is not None) and isinstance(leafOrPileRangeOfLeaves, int)

def thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None) -> TypeGuard[xmpz]:
	"""Return True if `leafOrPileRangeOfLeaves` is a pile's range of leaves.

	Parameters
	----------
	leafOrPileRangeOfLeaves : LeafOrPileRangeOfLeaves | None
		`leaf`, `pile`-range, or `None` to check.

	Returns
	-------
	youHaveAPileRange : TypeGuard[xmpz]
		Congrats, you have a pile range!
	"""
	return (leafOrPileRangeOfLeaves is not None) and isinstance(leafOrPileRangeOfLeaves, xmpz)

def oopsAllLeaves(leavesPinned: PinnedLeaves) -> dict[int, int]:
	"""Return a dictionary of all pinned leaves in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PinnedLeaves
		Dictionary of `pile` with pinned `leaf` or pile-range of leaves, if a `leaf` is pinned at `pile` or the pile-range of
		leaves is defined.

	Returns
	-------
	dictionaryOfPinnedLeaves : dict[int, int]
		Dictionary mapping from `pile` to pinned `leaf` for every pinned leaf in `leavesPinned`.
	"""
	return leafFilter(thisIsALeaf, leavesPinned) # pyright: ignore[reportReturnType]

def oopsAllPileRangesOfLeaves(leavesPinned: PinnedLeaves) -> dict[int, xmpz]:
	"""Return a dictionary of all pile-ranges of leaves in `leavesPinned`.

	Parameters
	----------
	leavesPinned : PinnedLeaves
		Dictionary of `pile` with pinned `leaf` or pile-range of leaves, if a `leaf` is pinned at `pile` or the pile-range of
		leaves is defined.

	Returns
	-------
	dictionaryOfPinnedLeaves : dict[int, xmpz]
		Dictionary mapping from `pile` to pinned `leaf` for every pinned leaf in `leavesPinned`.
	"""
	return leafFilter(thisIsAPileRangeOfLeaves, leavesPinned) # pyright: ignore[reportReturnType]

@syntacticCurry
def pileRangeAND(antiPileRange: xmpz, pileRangeOfLeaves: xmpz) -> xmpz:
	return iand(pileRangeOfLeaves, antiPileRange)

def Z0Z_idk(leavesPinned: PinnedLeaves, antiPileRange: xmpz) -> PinnedLeaves:
	for pile, pileRangeOfLeaves in leafFilter(thisIsAPileRangeOfLeaves, leavesPinned).items():
		leavesPinned[pile] = pileRangeAND(antiPileRange, pileRangeOfLeaves)
	return leavesPinned

@syntacticCurry
def Z0Z_updateDomains(leavesTotal: int, leavesPinned: PinnedLeaves) -> PinnedLeaves:
	"""Updates elements but does not check for duplicates or validity."""
	keepGoing = True
	while keepGoing:
		keepGoing = False
		leaves = oopsAllLeaves(leavesPinned).values()
		antiPileRange = getXmpzAntiPileRange(leavesTotal, leaves)
		leavesPinned = Z0Z_idk(leavesPinned, antiPileRange)
	return leavesPinned

def Z0Z_JeanValjean(p24601: LeafOrPileRangeOfLeaves) -> LeafOrPileRangeOfLeaves:
	whoAmI: LeafOrPileRangeOfLeaves = p24601
	if thisIsAPileRangeOfLeaves(p24601):
		if p24601.bit_count() == 1:
			# dammit. this is wrong. this means the pile-range is null and the dictionary should be discarded.
			whoAmI = 0
		elif p24601.bit_count() == 2:
			whoAmI = raiseIfNone(p24601.bit_scan1())
	return whoAmI

# ======= Pin one or more leaves in `leavesPinned` or `folding` =======================
# NOTE The only functions that actually pin a leaf in a dictionary or folding ought to be in this section.

@syntacticCurry
def atPilePinLeaf(leavesPinned: PinnedLeaves, pile: int, leaf: int) -> PinnedLeaves:
	"""Return a new `PinnedLeaves` with `leaf` pinned at `pile` without modifying `leavesPinned`.

	Warning
	-------
	This function assumes either 1. `leaf` is not pinned and `pile` is open or 2. `leaf` is pinned at `pile`. Overwriting a
	different `leaf` pinned at `pile` corrupts the permutation space.

	Parameters
	----------
	leavesPinned : PinnedLeaves
		Dictionary of `pile` with pinned `leaf` or pile-range of leaves, if a `leaf` is pinned at `pile` or the pile-range of
		leaves is defined.
	pile : int
		`pile` at which to pin `leaf`.
	leaf : int
		`leaf` to pin.

	Returns
	-------
	dictionaryLeavesPinned : PinnedLeaves
		New dictionary with `pile` mapped to `leaf`.

	See Also
	--------
	deconstructLeavesPinned
	"""
	return associate(leavesPinned, pile, leaf)

def makeFolding(leavesPinned: PinnedLeaves, leavesToInsert: tuple[int, ...]) -> tuple[int, ...]:
	permutand: Iterator[int] = iter(leavesToInsert)
	pilesTotal: int = sum(map(thisIsALeaf, leavesPinned.values())) + len(leavesToInsert)
	return tuple([
		leafOrLeafRange if thisIsALeaf(leafOrLeafRange := leavesPinned.get(pile)) else next(permutand)
		for pile in range(pilesTotal)
	])

# ======= Deconstruct a `PinnedLeaves` dictionary =======
# This function returns `deconstructedLeavesPinned : dict[int, LeavesPinned]` for pragmatic reasons. I typically deconstruct a
# pile because I want to pin one leaf at the pile without invalidating the permutation space. So, the dictionary makes it easy for
# me to segregate the one leaf I am working on from the umpteen other new dictionaries, each with a leaf pinned at the pile.
# If I change to representing the range of a pile with "bit packing", then I only need to make two dictionaries: one dictionary
# with the desired leaf pinned at pile, and another dictionary with a modified range at pile that excludes the desired leaf.
# The two systems are complementary: I can completely deconstruct a pile into only pinned leaves, bifurcate into one pinned
# dictionary and one pile-range dictionary, or any intermediate option.
def deconstructPinnedLeavesAtPile(leavesPinned: PinnedLeaves, pile: int, leavesToPin: Iterable[int]) -> dict[int, PinnedLeaves]:
	"""Deconstruct an open `pile` to the `leaf` range of `pile`.

	Return a dictionary of `PinnedLeaves` with either `leavesPinned` because it already has a `leaf` pinned at `pile` or one
	`PinnedLeaves` for each `leaf` in `leavesToPin` pinned at `pile`.

	Parameters
	----------
	leavesPinned : PinnedLeaves
		Dictionary to divide and replace.
	pile : int
		`pile` at which to pin a `leaf`.
	leavesToPin : list[int]
		List of `leaves` to pin at `pile`.

	Returns
	-------
	deconstructedLeavesPinned : dict[int, LeavesPinned]
		Dictionary mapping from `leaf` pinned at `pile` to the `PinnedLeaves` dictionary with the `leaf` pinned at `pile`.
	"""
	if thisIsALeaf(leaf := leavesPinned.get(pile)):
		deconstructedLeavesPinned: dict[int, PinnedLeaves] = {leaf: leavesPinned}
	else:
		pin: Callable[[int], PinnedLeaves] = atPilePinLeaf(leavesPinned, pile)
		leafCanBePinned: Callable[[int], bool] = leafIsNotPinned(leavesPinned)
		deconstructedLeavesPinned = {leaf: pin(leaf) for leaf in filter(leafCanBePinned, leavesToPin)}
	return deconstructedLeavesPinned

def deconstructPinnedLeavesByDomainOfLeaf(leavesPinned: PinnedLeaves, leaf: int, leafDomain: Iterable[int]) -> list[PinnedLeaves]:
	"""Pin `leaf` at each open `pile` in the domain of `leaf`.

	Return a `list` of `PinnedLeaves` with either `leavesPinned` because `leaf` is already pinned or one `PinnedLeaves` for each
	open `pile` in `leafDomain` with leaf pinned at `pile`.

	Parameters
	----------
	leavesPinned : PinnedLeaves
		Dictionary to divide and replace.
	leaf : int
		`leaf` to pin.
	leafDomain : Iterable[int]
		Domain of `pile` indices for `leaf`.

	Returns
	-------
	deconstructedLeavesPinned : list[PinnedLeaves]
		List of `PinnedLeaves` dictionaries with `leaf` pinned at each open `pile` in `leafDomain`.
	"""
	if leafIsNotPinned(leavesPinned, leaf):
		pinLeafAt: Callable[[int], PinnedLeaves] = atPilePinLeaf(leavesPinned, leaf=leaf)
		pileAvailable: Callable[[int], bool] = pileIsOpen(leavesPinned)
		deconstructedLeavesPinned: list[PinnedLeaves] = list(map(pinLeafAt, filter(pileAvailable, leafDomain)))
	else:
		deconstructedLeavesPinned = [leavesPinned]
	return deconstructedLeavesPinned

def deconstructPinnedLeavesByDomainsCombined(leavesPinned: PinnedLeaves, leaves: tuple[int, ...], leavesDomain: Iterable[tuple[int, ...]]) -> list[PinnedLeaves]:
	"""Prototype."""
	deconstructedLeavesPinned: list[PinnedLeaves] = []

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
			"""Redefine leavesDomain by filtering out domains that are not possible with the current `PinnedLeaves`."""
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
			leavesPinnedForListOfPiles: PinnedLeaves = leavesPinned.copy()
			for index in range(len(leaves)):
				leavesPinnedForListOfPiles = atPilePinLeaf(leavesPinnedForListOfPiles, listOfPiles[index], leaves[index])
			deconstructedLeavesPinned.append(leavesPinnedForListOfPiles)
	else:
		deconstructedLeavesPinned = [leavesPinned]

	return deconstructedLeavesPinned

# ======= Bulk modifications =======================

def deconstructListPinnedLeavesAtPile(listPinnedLeaves: Iterable[PinnedLeaves], pile: int, leavesToPin: Iterable[int]) -> Iterator[PinnedLeaves]:
	"""Expand every dictionary in `listPinnedLeaves` at `pile` into all pinning variants.

	Applies `deconstructLeavesPinned` element-wise, then flattens the nested value collections (each a mapping leaf -> dictionary)
	into a single list of dictionaries, discarding the intermediate keyed structure.

	Parameters
	----------
	listPinnedLeaves : Iterable[PinnedLeaves]
		Partial folding dictionaries.
	pile : int
		`pile` index to expand.
	leavesToPin : Iterable[int]
		`leaf` indices to pin at `pile`.

	Returns
	-------
	listPinnedLeaves : Iterator[PinnedLeaves]
		Flat iterator of expanded dictionaries covering all possible `leaf` assignments at `pile`.

	See Also
	--------
	deconstructLeavesPinned
	"""
	return  flatten(map(DOTvalues, map(deconstructPinnedLeavesAtPile, listPinnedLeaves, repeat(pile), repeat(leavesToPin))))

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
		Same state instance, mutated with updated `listPinnedLeaves`.

	See Also
	--------
	excludeLeafRBeforeLeafK, theorem4, theorem2b
	"""
	iterator_kPinnedAt_pile_k: Iterable[PinnedLeaves] = []
	iterable_pile_kOccupied: Iterable[PinnedLeaves] = []
	listPinnedLeaves: list[PinnedLeaves] = []
	listPinnedLeavesCompleted: list[PinnedLeaves] = []

	pile_kIsOpen: Callable[[PinnedLeaves], bool] = pileIsOpen(pile=pile_k)
	kIsNotPinned: Callable[[PinnedLeaves], bool] = leafIsNotPinned(leaf=k)
	def notLeaf_r(comparand: int, r: int = r) -> bool:
		return comparand != r

	listPinnedLeaves, iterator_kPinnedAt_pile_k = _segregateLeafPinnedAtPile(state.listPinnedLeaves, k, pile_k)
	state.listPinnedLeaves = []

	grouped: dict[bool, list[PinnedLeaves]] = toolz_groupby(kIsNotPinned, listPinnedLeaves)
	listPinnedLeavesCompleted, listPinnedLeaves = grouped.get(False, []), grouped.get(True, [])
	grouped = toolz_groupby(pile_kIsOpen, listPinnedLeaves)
	iterable_pile_kOccupied, listPinnedLeaves = grouped.get(False, []), grouped.get(True, [])
	listPinnedLeavesCompleted.extend(iterable_pile_kOccupied)

	if domain_r is None:
		domain_r = getLeafDomain(state, r)

	domain_r = filter(between(0, pile_k - inclusive), domain_r)

	if rangePile_k is None:
		rangePile_k = getPileRange(state, pile_k)

	rangePile_k = set(rangePile_k)

	if k in rangePile_k:
		for leavesPinned_kPinnedAt_pile_k, iterable_pile_kOccupied in segregateLeafByDeconstructingListPinnedLeavesAtPile(listPinnedLeaves, k, pile_k, rangePile_k):
			listPinnedLeavesCompleted.extend(iterable_pile_kOccupied)
			iterator_kPinnedAt_pile_k.append(leavesPinned_kPinnedAt_pile_k)
	else:
		listPinnedLeavesCompleted.extend(listPinnedLeaves)

	for pile_r in domain_r:
		iterator_kPinnedAt_pile_k = excludeLeafAtPile(iterator_kPinnedAt_pile_k, r, pile_r, tuple(filter(notLeaf_r, getPileRange(state, pile_r))))

	state.listPinnedLeaves.extend(listPinnedLeavesCompleted)
	state.listPinnedLeaves.extend(iterator_kPinnedAt_pile_k)

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
		Same state instance, mutated with updated `listPinnedLeaves`.

	See Also
	--------
	_excludeLeafRBeforeLeafK, theorem4, theorem2b
	"""
	if domain_k is None:
		domain_k = getLeafDomain(state, k)
	for pile_k in reversed(tuple(domain_k)):
		state = excludeLeaf_rBeforeLeaf_kAtPile_k(state, k, r, pile_k, domain_r=domain_r)
	return state

def excludeLeafAtPile(listPinnedLeaves: Iterable[PinnedLeaves], leaf: int, pile: int, leavesToPin: Iterable[int]) -> Iterator[PinnedLeaves]:
	"""Return a new list of pinned-leaves dictionaries that forbid `leaf` at `pile`.

	Parameters
	----------
	listPinnedLeaves : Iterable[PinnedLeaves]
		Collection of partial pinning dictionaries to transform.
	leaf : int
		`leaf` to exclude from `pile`.
	pile : int
		`pile` at which `leaf` must not be fixed.
	leavesToPin : Iterable[int]
		List of leaves available for pinning at `pile`. Don't include `leaf`.

	Returns
	-------
	listPinnedLeaves : Iterable[PinnedLeaves]
		Expanded / filtered list respecting the exclusion constraint.

	See Also
	--------
	deconstructLeavesPinned : Performs the expansion for one dictionary.
	pinLeafAtPile : Complementary operation that forces a leaf at a pile.
	"""
	return deconstructListPinnedLeavesAtPile(_segregateLeafPinnedAtPile(list(listPinnedLeaves), leaf, pile)[0], pile, leavesToPin)

def requireLeafPinnedAtPile(listPinnedLeaves: list[PinnedLeaves], leaf: int, pile: int) -> list[PinnedLeaves]:
	"""In every `PinnedLeaves` dictionary, ensure `leaf`, and *only* `leaf`, is pinned at `pile`: excluding every other `leaf` at `pile`.

	Parameters
	----------
	listPinnedLeaves : list[PinnedLeaves]
		Collection of partial pinning dictionaries to transform.
	leaf : int
		`leaf` required at `pile`.
	pile : int
		`pile` at which to pin the leaf.

	Returns
	-------
	listLeafAtPile : list[PinnedLeaves]
		`list` of `PinnedLeaves` dictionaries with `leaf` pinned at `pile`.

	See Also
	--------
	deconstructPinnedLeaves, excludeLeafAtPile
	"""
	listPinnedLeaves, listLeafAtPile = _segregateLeafPinnedAtPile(listPinnedLeaves, leaf, pile)
	listLeafAtPile.extend(map(atPilePinLeaf(pile=pile, leaf=leaf), filter(pileIsOpen(pile=pile), filter(leafIsNotPinned(leaf=leaf), listPinnedLeaves))))
	return listLeafAtPile

def segregateLeafByDeconstructingListPinnedLeavesAtPile(listPinnedLeaves: Iterable[PinnedLeaves], leaf: int, pile: int, leavesToPin: Iterable[int]) -> Iterator[tuple[PinnedLeaves, tuple[PinnedLeaves, ...]]]:
	for leavesPinned in listPinnedLeaves:
		deconstructedLeavesPinnedAtPile: dict[int, PinnedLeaves] = deconstructPinnedLeavesAtPile(leavesPinned, pile, leavesToPin)
		leafPinnedAtPile: PinnedLeaves = deconstructedLeavesPinnedAtPile.pop(leaf)
		yield (leafPinnedAtPile, tuple(deconstructedLeavesPinnedAtPile.values()))

