"""Generalized pinning functions in the "Elimination" algorithm for any `mapShape`.

Functions for 2^n-dimensional maps must go in other modules.

The development of this generalized module is severely hampered, however. Functions for 2^n-dimensional maps have a "beans and
cornbread" problem that was difficult for me to "solve"--due to my programming skills. If I were able to decouple the "beans and
cornbread" solution from the 2^n-dimensional functions, I would generalize more functions and move them here.
"""
from collections.abc import Callable, Iterable, Iterator
from cytoolz.dicttoolz import assoc as associate
from cytoolz.functoolz import compose, curry as syntacticCurry
from cytoolz.itertoolz import groupby as toolz_groupby
from functools import partial
from gmpy2 import bit_mask
from itertools import repeat
from mapFolding import inclusive
from mapFolding._e import (
	between, DOTgetPileIfLeaf, DOTgetPileIfPileRangeOfLeaves, DOTvalues, Folding, getLeafDomain, getPileRange, Leaf,
	leafIsInPileRange, leafIsNotPinned, oopsAllLeaves, PermutationSpace, Pile, pileIsOpen, thisIsALeaf)
from mapFolding._e.dataBaskets import EliminationState
from more_itertools import flatten, ilen
from operator import getitem

#======== Boolean filters =======================

@syntacticCurry
def atPilePinLeafSafetyFilter(permutationSpace: PermutationSpace, pile: Pile, leaf: Leaf) -> bool:
	"""Return `True` if it is safe to call `atPilePinLeafSafetyFilter(permutationSpace, pile, leaf)`.

	For performance, you probably can and probably *should* create a set of filters for your circumstances.

	Parameters
	----------
	permutationSpace : PermutationSpace
		A mapping of each `pile` with a pinned `leaf`.
	pile : int
		`pile` at which to pin.
	leaf : int
		`leaf` to pin.

	Returns
	-------
	isSafeToPin : bool
		True if it is safe to pin `leaf` at `pile` in `permutationSpace`.
	"""
	return isPinnedAtPile(permutationSpace, leaf, pile) or (pileIsOpen(permutationSpace, pile) and leafIsNotPinned(permutationSpace, leaf))

@syntacticCurry
def isPinnedAtPile(permutationSpace: PermutationSpace, leaf: Leaf, pile: Pile) -> bool:
	"""Return `True` if `leaf` is presently pinned at `pile` in `permutationSpace`.

	Parameters
	----------
	permutationSpace : PermutationSpace
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
	return leaf == permutationSpace.get(pile)

#======== Group by =======================

def _segregateLeafPinnedAtPile(listPermutationSpace: list[PermutationSpace], leaf: Leaf, pile: Pile) -> tuple[list[PermutationSpace], list[PermutationSpace]]:
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

#======== Pin a `leaf` in a `PermutationSpace` or `Folding` =======================
# NOTE This section ought to contain all functions based on the "Elimination" algorithm that pin a `leaf` in a `PermutationSpace` or `Folding`.

@syntacticCurry
def atPilePinLeaf(permutationSpace: PermutationSpace, pile: Pile, leaf: Leaf) -> PermutationSpace:
	"""Return a new `PermutationSpace` with `leaf` pinned at `pile` without modifying `permutationSpace`.

	Warning
	-------
	This function assumes either 1. `leaf` is not pinned and `pile` is open or 2. `leaf` is pinned at `pile`. Overwriting a
	different `leaf` pinned at `pile` corrupts the permutation space.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary of `pile` with pinned `leaf` or pile-range of leaves, if a `leaf` is pinned at `pile` or the pile-range of
		leaves is defined.
	pile : int
		`pile` at which to pin `leaf`.
	leaf : int
		`leaf` to pin.

	Returns
	-------
	dictionaryPermutationSpace : PermutationSpace
		New dictionary with `pile` mapped to `leaf`.

	See Also
	--------
	deconstructPermutationSpace
	"""
	return associate(permutationSpace, pile, leaf)

# TODO more flexible.
def makeFolding(permutationSpace: PermutationSpace, leavesToInsert: tuple[Leaf, ...]) -> Folding:
	if leavesToInsert:
		permutand: Iterator[Leaf] = iter(leavesToInsert)
		pilesTotal: int = ilen(filter(thisIsALeaf, DOTvalues(permutationSpace))) + len(leavesToInsert)
		# pilesTotal: int = len(oopsAllLeaves(permutationSpace)) + len(leavesToInsert)  # noqa: ERA001
		folding: Folding = tuple([
			leafOrLeafRange if (leafOrLeafRange := DOTgetPileIfLeaf(permutationSpace, pile)) else next(permutand)
			for pile in range(pilesTotal)
		])
	else:
		folding = tuple(DOTvalues(oopsAllLeaves(permutationSpace)))
	return folding

#======== Deconstruct a `PermutationSpace` dictionary =======
def deconstructPermutationSpaceAtPile(permutationSpace: PermutationSpace, pile: Pile, leavesToPin: Iterable[Leaf]) -> dict[Leaf, PermutationSpace]:
	"""Deconstruct an open `pile` to the `leaf` range of `pile`.

	Return a dictionary of `PermutationSpace` with either `permutationSpace` because it already has a `leaf` pinned at `pile` or one
	`PermutationSpace` for each `leaf` in `leavesToPin` pinned at `pile`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary to divide and replace.
	pile : int
		`pile` at which to pin a `leaf`.
	leavesToPin : list[int]
		List of `leaves` to pin at `pile`.

	Returns
	-------
	deconstructedPermutationSpace : dict[int, PermutationSpace]
		Dictionary mapping from `leaf` pinned at `pile` to the `PermutationSpace` dictionary with the `leaf` pinned at `pile`.
	"""
	if thisIsALeaf(leaf := permutationSpace.get(pile)):
		deconstructedPermutationSpace: dict[Leaf, PermutationSpace] = {leaf: permutationSpace}
	else:
		pin: Callable[[Leaf], PermutationSpace] = atPilePinLeaf(permutationSpace, pile)
		leafCanBePinned: Callable[[Leaf], bool] = leafIsNotPinned(permutationSpace)
		deconstructedPermutationSpace = {leaf: pin(leaf) for leaf in filter(leafCanBePinned, leavesToPin)}
	return deconstructedPermutationSpace

def deconstructPermutationSpaceByDomainOfLeaf(permutationSpace: PermutationSpace, leaf: Leaf, leafDomain: Iterable[Pile]) -> list[PermutationSpace]:
	"""Pin `leaf` at each open `pile` in the domain of `leaf`.

	Return a `list` of `PermutationSpace` with either `permutationSpace` because `leaf` is already pinned or one `PermutationSpace` for each
	open `pile` in `leafDomain` with `leaf` pinned at `pile`.

	Parameters
	----------
	permutationSpace : PermutationSpace
		Dictionary to divide and replace.
	leaf : int
		`leaf` to pin.
	leafDomain : Iterable[int]
		Domain of `pile` indices for `leaf`.

	Returns
	-------
	deconstructedPermutationSpace : list[PermutationSpace]
		List of `PermutationSpace` dictionaries with `leaf` pinned at each open `pile` in `leafDomain`.
	"""
	if leafIsNotPinned(permutationSpace, leaf):
		pileOpen: Callable[[int], bool] = pileIsOpen(permutationSpace)
		leafInPileRange: Callable[[int], bool] = compose(leafIsInPileRange(leaf), partial(DOTgetPileIfPileRangeOfLeaves, permutationSpace, default=bit_mask(len(permutationSpace))))
		pinLeafAt: Callable[[int], PermutationSpace] = atPilePinLeaf(permutationSpace, leaf=leaf)
		deconstructedPermutationSpace: list[PermutationSpace] = list(map(pinLeafAt, filter(leafInPileRange, filter(pileOpen, leafDomain))))
	else:
		deconstructedPermutationSpace = [permutationSpace]
	return deconstructedPermutationSpace

def deconstructPermutationSpaceByDomainsCombined(permutationSpace: PermutationSpace, leaves: tuple[Leaf, ...], leavesDomain: Iterable[tuple[Pile, ...]]) -> list[PermutationSpace]:
	"""Prototype."""
	deconstructedPermutationSpace: list[PermutationSpace] = []

	def pileOpenByIndex(index: int) -> Callable[[tuple[Pile, ...]], bool]:
		def workhorse(domain: tuple[Pile, ...]) -> bool:
			return pileIsOpen(permutationSpace, domain[index])
		return workhorse

	def leafInPileRangeByIndex(index: int) -> Callable[[tuple[Pile, ...]], bool]:
		def workhorse(domain: tuple[Pile, ...]) -> bool:
			pileRangeOfLeaves = DOTgetPileIfPileRangeOfLeaves(permutationSpace, domain[index], default=bit_mask(len(permutationSpace)))
			return leafIsInPileRange(leaves[index], pileRangeOfLeaves)
		return workhorse

	def isPinnedAtPileByIndex(leaf: Leaf, index: int) -> Callable[[tuple[Pile, ...]], bool]:
		def workhorse(domain: tuple[Pile, ...]) -> bool:
			return isPinnedAtPile(permutationSpace, leaf, domain[index])
		return workhorse

	if any(map(leafIsNotPinned(permutationSpace), leaves)):
		for index in range(len(leaves)):
			"""Redefine leavesDomain by filtering out domains that are not possible with the current `PermutationSpace`."""
			if leafIsNotPinned(permutationSpace, leaves[index]):
				"""`leaves[index]` is not pinned, so it needs a pile.
				In each iteration of `leavesDomain`, `listOfPiles`, the pile it needs is `listOfPiles[index]`.
				Therefore, if `listOfPiles[index]` is open, filter in the iteration. If `listOfPiles[index]` is occupied, filter out the iteration."""
				leavesDomain = filter(pileOpenByIndex(index), leavesDomain)
				"""`leaves[index]` is not pinned, it wants `listOfPiles[index]`, and `listOfPiles[index]` is open.
				Is `leaves[index]` in the pile-range of `listOfPiles[index]`?"""
				leavesDomain = filter(leafInPileRangeByIndex(index), leavesDomain)
			else:
				"""`leaves[index]` is pinned.
				In each iteration of `leavesDomain`, `listOfPiles`, the pile in which `leaves[index]` is pinned must match `listOfPiles[index]`.
				Therefore, if the pile in which `leaves[index]` is pinned matches `listOfPiles[index]`, filter in the iteration. Otherwise, filter out the iteration."""
				leavesDomain = filter(isPinnedAtPileByIndex(leaves[index], index), leavesDomain)

		for listOfPiles in leavesDomain:
			"""Properly and safely deconstruct `permutationSpace` by the combined domain of leaves.
			The parameter `leavesDomain` is the full domain of the leaves, so deconstructing with `leavesDomain` preserves the permutation space.
			For each leaf in leaves, I filter out occupied piles, so I will not overwrite any pinned leaves--that would invalidate the permutation space.
			I apply filters that prevent pinning the same leaf twice.
			Therefore, for each domain in `leavesDomain`, I can safely pin `leaves[index]` at `listOfPiles[index]` without corrupting the permutation space."""
			permutationSpaceForListOfPiles: PermutationSpace = permutationSpace.copy()
			for index in range(len(leaves)):
				permutationSpaceForListOfPiles = atPilePinLeaf(permutationSpaceForListOfPiles, listOfPiles[index], leaves[index])
			deconstructedPermutationSpace.append(permutationSpaceForListOfPiles)
	else:
		deconstructedPermutationSpace = [permutationSpace]

	return deconstructedPermutationSpace

#======== Bulk modifications =======================

def deconstructListPermutationSpaceAtPile(listPermutationSpace: Iterable[PermutationSpace], pile: Pile, leavesToPin: Iterable[Leaf]) -> Iterator[PermutationSpace]:
	"""Expand every dictionary in `listPermutationSpace` at `pile` into all pinning variants.

	Applies `deconstructPermutationSpace` element-wise, then flattens the nested value collections (each a mapping leaf -> dictionary)
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
	deconstructPermutationSpace
	"""
	return  flatten(map(DOTvalues, map(deconstructPermutationSpaceAtPile, listPermutationSpace, repeat(pile), repeat(leavesToPin))))

def excludeLeaf_rBeforeLeaf_kAtPile_k(state: EliminationState, k: Leaf, r: Leaf, pile_k: Pile, domain_r: Iterable[Pile] | None = None, rangePile_k: Iterable[Leaf] | None = None) -> EliminationState:
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
	def notLeaf_r(comparand: Leaf, r: Leaf = r) -> bool:
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
		for permutationSpace_kPinnedAt_pile_k, iterable_pile_kOccupied in segregateLeafByDeconstructingListPermutationSpaceAtPile(listPermutationSpace, k, pile_k, rangePile_k):
			listPermutationSpaceCompleted.extend(iterable_pile_kOccupied)
			iterator_kPinnedAt_pile_k.append(permutationSpace_kPinnedAt_pile_k)
	else:
		listPermutationSpaceCompleted.extend(listPermutationSpace)

	for pile_r in domain_r:
		iterator_kPinnedAt_pile_k = excludeLeafAtPile(iterator_kPinnedAt_pile_k, r, pile_r, tuple(filter(notLeaf_r, getPileRange(state, pile_r))))

	state.listPermutationSpace.extend(listPermutationSpaceCompleted)
	state.listPermutationSpace.extend(iterator_kPinnedAt_pile_k)

	return state

def excludeLeaf_rBeforeLeaf_k(state: EliminationState, k: Leaf, r: Leaf, domain_k: Iterable[Pile] | None = None, domain_r: Iterable[Pile] | None = None) -> EliminationState:
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

	Returns
	-------
	listPermutationSpace : Iterable[PermutationSpace]
		Expanded / filtered list respecting the exclusion constraint.

	See Also
	--------
	deconstructPermutationSpace : Performs the expansion for one dictionary.
	pinLeafAtPile : Complementary operation that forces a leaf at a pile.
	"""
	return deconstructListPermutationSpaceAtPile(getitem(_segregateLeafPinnedAtPile(list(listPermutationSpace), leaf, pile), 0), pile, leavesToPin)

def requireLeafPinnedAtPile(listPermutationSpace: list[PermutationSpace], leaf: Leaf, pile: Pile) -> list[PermutationSpace]:
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

def segregateLeafByDeconstructingListPermutationSpaceAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile, leavesToPin: Iterable[Leaf]) -> Iterator[tuple[PermutationSpace, tuple[PermutationSpace, ...]]]:
	for permutationSpace in listPermutationSpace:
		deconstructedPermutationSpaceAtPile: dict[Leaf, PermutationSpace] = deconstructPermutationSpaceAtPile(permutationSpace, pile, leavesToPin)
		leafPinnedAtPile: PermutationSpace = deconstructedPermutationSpaceAtPile.pop(leaf)
		yield (leafPinnedAtPile, tuple(deconstructedPermutationSpaceAtPile.values()))

