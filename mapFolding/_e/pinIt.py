from collections.abc import Callable, Iterable
from cytoolz.dicttoolz import assoc as associate
from cytoolz.functoolz import complement, curry as syntacticCurry
from cytoolz.itertoolz import groupby as toolz_groupby
from itertools import repeat
from mapFolding import decreasing, DOTvalues, inclusive
from mapFolding._e import PinnedLeaves
from mapFolding.dataBaskets import EliminationState
from more_itertools import flatten
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator

"""Maintaining a valid permutation space:

- If a pile is not pinned, deconstruct the dictionary into multiple dictionaries: for each unpinned leaf, pin it a pile.
- If I know the domain of a leaf, deconstruct the dictionary into multiple dictionaries: for each pile in the domain, pin the leaf at that pile.

- Do not overwrite or delete a dictionary's pinned leaves because that could cause the permutation space to overlap with a different dictionary's permutation space.
"""
# ======= Deconstruct a `pinnedLeaves` dictionary =======

def deconstructPinnedLeavesAtPile(pinnedLeaves: PinnedLeaves, pile: int, leavesToPin: Iterable[int]) -> dict[int, PinnedLeaves]:
	"""Return a dictionary of `PinnedLeaves` with either `pinnedLeaves` because it already has a `leaf` pinned at `pile` or one `PinnedLeaves` for each `leaf` in `leavesToPin` pinned at `pile`.

	Parameters
	----------
	pinnedLeaves : PinnedLeaves
		Dictionary to divide and replace.
	pile : int
		`pile` at which to pin a `leaf`.
	leavesToPin : list[int]
		List of `leaves` to pin at `pile`.

	Returns
	-------
	deconstructedPinnedLeaves : dict[int, PinnedLeaves]
		Dictionary mapping from `leaf` pinned at `pile` to the dictionary with the `leaf` pinned at `pile`.
	"""
	if not pileIsOpen(pinnedLeaves, pile):
		deconstructedPinnedLeaves: dict[int, PinnedLeaves] = {pinnedLeaves[pile]: pinnedLeaves}
	else:
		pin: Callable[[int], PinnedLeaves] = atPilePinLeaf(pinnedLeaves, pile)
		leafCanBePinned: Callable[[int], bool] = leafIsNotPinned(pinnedLeaves)
		deconstructedPinnedLeaves = {leaf: pin(leaf) for leaf in filter(leafCanBePinned, leavesToPin)}
	return deconstructedPinnedLeaves

def deconstructPinnedLeavesByLeaf(pinnedLeaves: PinnedLeaves, leaf: int, listPiles: Iterable[int]) -> list[PinnedLeaves]:
	"""Pin `leaf` at each open `pile` is the domain of `leaf`.

	Return a `list` of `PinnedLeaves` with either `pinnedLeaves` because `leaf` is already pinned or one `PinnedLeaves` for each
	open `pile` in `listPiles` with leaf pinned at `pile` and `pinnedLeaves` if a `pile` is occupied.

	Parameters
	----------
	pinnedLeaves : PinnedLeaves
		Dictionary to divide and replace.
	pile : int
		`pile` at which to pin a `leaf`.
	leavesToPin : list[int]
		List of `leaves` to pin at `pile`.

	Returns
	-------
	deconstructedPinnedLeaves : list[PinnedLeaves]
		Dictionary mapping from `leaf` pinned at `pile` to the dictionary with the `leaf` pinned at `pile`.
	"""
	if leafIsNotPinned(pinnedLeaves, leaf):
		pinLeafTo: Callable[[int], PinnedLeaves] = atPilePinLeaf(pinnedLeaves, leaf=leaf)
		pileAvailable: Callable[[int], bool] = pileIsOpen(pinnedLeaves)
		pileOccupied: Callable[[int], bool] = complement(pileIsOpen(pinnedLeaves))
		deconstructedPinnedLeaves: list[PinnedLeaves] = [pinLeafTo(pile) for pile in filter(pileAvailable, listPiles)]
		if tuple(filter(pileOccupied, listPiles)):
			deconstructedPinnedLeaves.append(pinnedLeaves)
	else:
		deconstructedPinnedLeaves = [pinnedLeaves]
	return deconstructedPinnedLeaves

# ======= Fix one or more leaves in `pinnedLeaves` or `folding` =======================

@syntacticCurry
def atPilePinLeaf(pinnedLeaves: PinnedLeaves, pile: int, leaf: int) -> PinnedLeaves:
	"""Return a new dictionary with `leaf` pinned at `pile` based on `pinnedLeaves`.

	Parameters
	----------
	pinnedLeaves : PinnedLeaves
		Existing `pile` -> `leaf` mapping (partial folding).
	pile : int
		`pile` index at which to set the `leaf`.
	leaf : int
		`leaf` to pin.

	Returns
	-------
	dictionaryPinnedLeaves : PinnedLeaves
		New mapping including the assignment.

	See Also
	--------
	deconstructPinnedLeaves
	"""
	return associate(pinnedLeaves, pile, leaf)

def makeFolding(pinnedLeaves: PinnedLeaves, leavesToInsert: tuple[int, ...]) -> tuple[int, ...]:
	permutand: Iterator[int] = iter(leavesToInsert)
	return tuple([pinnedLeaves[pile] if pile in pinnedLeaves else next(permutand) for pile in range(len(pinnedLeaves) + len(leavesToInsert))])

# ======= Boolean filters =======================
# TODO make and use reusable filters.
@syntacticCurry
def isPinnedAtPile(pinnedLeaves: PinnedLeaves, leaf: int, pile: int) -> bool:
	"""Return True if `leaf` is presently pinned at `pile` in `pinnedLeaves`.

	Parameters
	----------
	pinnedLeaves : PinnedLeaves
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
	return leaf == pinnedLeaves.get(pile)

@syntacticCurry
def leafIsNotPinned(pinnedLeaves: PinnedLeaves, leaf: int) -> bool:
	"""Return True if `leaf` is not presently pinned in `pinnedLeaves`.

	Parameters
	----------
	pinnedLeaves : PinnedLeaves
		Partial folding mapping from pile -> leaf.
	leaf : int
		`leaf` index.

	Returns
	-------
	leafIsNotPinned : bool
		True if the mapping does not include `leaf`.
	"""
	return leaf not in pinnedLeaves.values()

@syntacticCurry
def pileIsOpen(pinnedLeaves: PinnedLeaves, pile: int) -> bool:
	"""Return True if `pile` is not presently pinned in `pinnedLeaves`.

	Parameters
	----------
	pinnedLeaves : PinnedLeaves
		Partial folding mapping from pile -> leaf.
	pile : int
		`pile` index.

	Returns
	-------
	pileIsOpen : bool
		True if `pile` is not a key in `pinnedLeaves`.
	"""
	return pile not in pinnedLeaves

# ======= Is this "group by", a "filter", "flow control", or something else? =======================

def _segregatePinnedAtPile(listPinnedLeaves: list[PinnedLeaves], leaf: int, pile: int) -> tuple[list[PinnedLeaves], list[PinnedLeaves]]:
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

# ======= Bulk modifications =======================

def deconstructListPinnedLeavesAtPile(listPinnedLeaves: list[PinnedLeaves], pile: int, leavesToPin: Iterable[int]) -> list[PinnedLeaves]:
	"""Expand every dictionary in `listPinnedLeaves` at `pile` into all pinning variants.

	Applies `deconstructPinnedLeaves` element-wise, then flattens the nested value collections (each a mapping leaf -> dictionary)
	into a single list of dictionaries, discarding the intermediate keyed structure. Centralizes expansion logic for downstream

	elimination predicates.

	Parameters
	----------
	listPinnedLeaves : list[PinnedLeaves]
		Partial folding dictionaries.
	pile : int
		`pile` index to expand.
	leavesTotal : int
		Total number of leaves.

	Returns
	-------
	listPinnedLeaves : list[PinnedLeaves]
		Flat list of expanded dictionaries covering all possible `leaf` assignments at `pile`.

	See Also
	--------
	deconstructPinnedLeaves
	"""
	return list(flatten(map(DOTvalues, map(deconstructPinnedLeavesAtPile, listPinnedLeaves, repeat(pile), repeat(leavesToPin)))))

def _excludeLeafRBeforeLeafK(state: EliminationState, k: int, r: int, pileK: int, listPinnedLeaves: list[PinnedLeaves]) -> list[PinnedLeaves]:
	"""Internal worker enforcing the ordering constraint: leaf `r` cannot precede leaf `k`.

	Mechanics
	---------
	1. Fully expand dictionaries at `pileK` (since we will inspect / segregate pinning status for leaf `k`).
	2. Scan piles from `pileK` upward, segregating dictionaries where `k` is pinned at each pile; accumulate those pinned sets.
	3. After gathering all dictionaries that pin `k` at or after `pileK`, exclude leaf `r` from immediately preceding pile (`pileK - 1`).

	This mirrors the structure required by Lunnon-style dimension product constraints (later applied in `theorem4` / `theorem2b`).

	Parameters
	----------
	state : EliminationState
		Mutable elimination state (provides `leavesTotal`, `pileLast`).
	k : int
		Reference leaf index derived from `productOfDimensions` for a dimension.
	r : int
		Leaf that must not appear before `k` (also dimension-derived).
	pileK : int
		Pile index currently under consideration for leaf `k`.
	listPinnedLeaves : list[PinnedLeaves]
		Working list of pinned dictionaries to transform.

	Returns
	-------
	listPinnedLeaves : list[PinnedLeaves]
		Updated list implementing the exclusion.

	See Also
	--------
	excludeLeafRBeforeLeafK, theorem4, theorem2b
	"""
	listPinnedLeaves = deconstructListPinnedLeavesAtPile(listPinnedLeaves, pileK, range(state.leavesTotal))
	listPinned: list[PinnedLeaves] = []
	for pile in range(pileK, state.pileLast + inclusive):
		(listPinnedLeaves, listPinnedAtPile) = _segregatePinnedAtPile(listPinnedLeaves, k, pile)
		listPinned.extend(listPinnedAtPile)
	leavesToPin: list[int] = list(range(state.leavesTotal))
	leavesToPin.remove(r)
	listPinnedLeaves.extend(excludeLeafAtPile(listPinned, r, pileK - 1, leavesToPin))
	return listPinnedLeaves

def excludeLeafRBeforeLeafK(state: EliminationState, k: int, r: int) -> EliminationState:
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
	for pileK in range(state.pileLast, 0, decreasing):
		state.listPinnedLeaves = _excludeLeafRBeforeLeafK(state, k, r, pileK, state.listPinnedLeaves)
	return state

def excludeLeafAtPile(listPinnedLeaves: list[PinnedLeaves], leaf: int, pile: int, leavesToPin: Iterable[int]) -> list[PinnedLeaves]:
	"""Return a new list of pinned-leaves dictionaries that forbid `leaf` at `pile`.

	Parameters
	----------
	listPinnedLeaves : list[PinnedLeaves]
		Collection of partial pinning dictionaries to transform.
	leaf : int
		`leaf` to exclude from `pile`.
	pile : int
		`pile` at which `leaf` must not be fixed.
	leavesToPin : list[int]
		List of leaves available for pinning at `pile`. Don't include `leaf`.

	Returns
	-------
	listPinnedLeaves : list[PinnedLeaves]
		Expanded / filtered list respecting the exclusion constraint.

	See Also
	--------
	deconstructPinnedLeaves : Performs the expansion for one dictionary.
	pinLeafAtPile : Complementary operation that forces a leaf at a pile.
	"""
	return deconstructListPinnedLeavesAtPile(_segregatePinnedAtPile(listPinnedLeaves, leaf, pile)[0], pile, leavesToPin)

def pinLeafAtPile(listPinnedLeaves: list[PinnedLeaves], leaf: int, pile: int) -> list[PinnedLeaves]:
	"""Return a new list of `pinnedLeaves` dictionaries with `leaf` pinned at `pile` and excluding every other `leaf` at `pile`.

	Inverse of `_excludeLeafAtPile`: for each partial dictionary, if the pile is free and does not already pin the leaf elsewhere,
	this expands the dictionary (via `deconstructPinnedLeaves`) and retains only the variant that pins the requested `leaf`.

	Parameters
	----------
	listPinnedLeaves : list[PinnedLeaves]
		Collection of partial pinning dictionaries to transform.
	leaf : int
		`leaf` to force at `pile`.
	pile : int
		`pile` index at which to pin the leaf.

	Returns
	-------
	sequencePinnedLeaves : list[PinnedLeaves]
		Filtered / expanded list where each dictionary pins `leaf` to `pile` (unless impossible, in which case the dictionary is dropped).

	See Also
	--------
	deconstructPinnedLeaves, _excludeLeafAtPile
	"""
	listNeedsPinning, sequencePinnedLeaves = _segregatePinnedAtPile(listPinnedLeaves, leaf, pile)
	sequencePinnedLeaves.extend(deconstructListPinnedLeavesAtPile(list(filter(pileIsOpen(pile=pile), listNeedsPinning)), pile, [leaf]))
	return sequencePinnedLeaves
