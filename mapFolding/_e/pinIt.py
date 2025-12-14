from collections.abc import Callable, Iterable, Iterator
from cytoolz.dicttoolz import assoc as associate
from cytoolz.functoolz import complement, curry as syntacticCurry
from cytoolz.itertoolz import groupby as toolz_groupby
from itertools import repeat
from mapFolding import DOTvalues, mappingHasKey
from mapFolding._e import getLeafDomain, getPileRange, PinnedLeaves, 零
from mapFolding.dataBaskets import EliminationState
from more_itertools import flatten

"""Rules for maintaining a valid permutation space:

1. In `leavesPinned`, if `leaf` is not pinned, deconstruct `leavesPinned` by the `pile` domain of `leaf`.
	A. For each `pile` in the domain of `leaf`, if `pile` in `leavesPinned` is not occupied, create a new `PinnedLeaves` dictionary by appending `leaf` pinned at `pile` to `leavesPinned`.
	B. Replace `leavesPinned` with the group of newly created `PinnedLeaves` dictionaries.
2. In `leavesPinned`, if a `pile` is not pinned, deconstruct `leavesPinned` by the `leaf` range (mathematical range) of `pile`.
	A. For each `leaf` in the range of `pile`, if `leaf` is not already pinned in `leavesPinned`, create a new `PinnedLeaves` dictionary by appending `leaf` pinned at `pile` to `leavesPinned`.
	B. Replace `leavesPinned` with the group of newly created `PinnedLeaves` dictionaries.
3. Do not overwrite or delete a dictionary's pinned leaves because that could cause the dictionary's permutation space to overlap with a different dictionary's permutation space.
"""
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
		True if `pile` is not a key in `leavesPinned`.
	"""
	return complement(mappingHasKey)(leavesPinned, pile)

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
		Dictionary of `pile` with pinned `leaf`, if a `leaf` is pinned at `pile`.
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
	return tuple([leavesPinned[pile] if pile in leavesPinned else next(permutand) for pile in range(len(leavesPinned) + len(leavesToInsert))])

# ======= Deconstruct a `PinnedLeaves` dictionary =======
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
	if not pileIsOpen(leavesPinned, pile):
		deconstructedLeavesPinned: dict[int, PinnedLeaves] = {leavesPinned[pile]: leavesPinned}
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

	listPinnedLeaves, iterator_kPinnedAt_pile_k = _segregateLeafPinnedAtPile(state.listPinnedLeaves, k, pile_k)
	state.listPinnedLeaves = []

	grouped: dict[bool, list[PinnedLeaves]] = toolz_groupby(kIsNotPinned, listPinnedLeaves)
	listPinnedLeavesCompleted, listPinnedLeaves = grouped.get(False, []), grouped.get(True, [])
	grouped = toolz_groupby(pile_kIsOpen, listPinnedLeaves)
	iterable_pile_kOccupied, listPinnedLeaves = grouped.get(False, []), grouped.get(True, [])
	listPinnedLeavesCompleted.extend(iterable_pile_kOccupied)

	if domain_r is None:
		domain_r = getLeafDomain(state, r)

	domain_r = filter(lambda pile_r: pile_r < pile_k, domain_r)

	if rangePile_k is None:
		rangePile_k = getPileRange(state, pile_k)

	rangePile_k = tuple(rangePile_k)

	if k in rangePile_k:
		for leavesPinned_kPinnedAt_pile_k, iterable_pile_kOccupied in segregateLeafByDeconstructingListPinnedLeavesAtPile(listPinnedLeaves, k, pile_k, rangePile_k):
			listPinnedLeavesCompleted.extend(iterable_pile_kOccupied)
			iterator_kPinnedAt_pile_k.append(leavesPinned_kPinnedAt_pile_k)
	else:
		listPinnedLeavesCompleted.extend(listPinnedLeaves)

	for pile_r in domain_r:
		iterator_kPinnedAt_pile_k = excludeLeafAtPile(iterator_kPinnedAt_pile_k, r, pile_r, tuple(filter(lambda leaf: leaf != r, getPileRange(state, pile_r))))

	state.listPinnedLeaves.extend(listPinnedLeavesCompleted)
	state.listPinnedLeaves.extend(iterator_kPinnedAt_pile_k)

	return state

def excludeLeaf_rBeforeLeaf_k(state: EliminationState, k: int, r: int, domain_k: Iterable[int] | None = None, domain_r: Iterable[int] | None = None) -> EliminationState:
	"""Apply the leaf ordering exclusion (r cannot precede k) over all piles.

	Iterates `pileK` downward from `pileLast` to 1, invoking `_excludeLeafRBeforeLeafK` which performs localized expansion,
	segregation, and exclusion. The descending order preserves correctness because earlier piles may depend on knowledge of
	later pinning positions.

	Parameters
	----------
	state : EliminationState
		Mutable elimination state.
	k : int
		Leaf index representing the first of a dimension-derived pair.
	r : int
		Leaf index representing the second (excluded before `k`).

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

