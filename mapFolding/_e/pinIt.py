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
	assoc as associate, compose, concat, curry as syntacticCurry, groupby as toolz_groupby, itemfilter, keyfilter as filterPile, merge, unique,
	valfilter as filterLeaf, valfilter as filterLeafOptions, valfilter as filterValue, valmap as mapLeaf)
# TODO One or more things is messed up with humpy_*toolz.*.map
from hunterMakesPy import errorL33T, inclusive, raiseIfNone
from itertools import chain, combinations, product as CartesianProduct, repeat
from mapFolding._e import (
	bifurcatePermutationSpace, dimensionNearest首, DOTgetPileIfLeaf, DOTgetPileIfLeafOptions, getDictionaryLeafOptions, getIteratorOfLeaves,
	getLeafDomain, getLeafOptions, howManyLeavesInLeafOptions, JeanValjean, leafOptionsAND, makeLeafAntiOptions)
from mapFolding._e.algorithms.iff import creaseViolation吗, oddLeaf吗
from mapFolding._e.dataBaskets import PermutationSpace
from mapFolding._e.filters import (
	extractPinnedLeaves, extractUndeterminedPiles, leafInLeafOptions吗, leafNotPinned吗, leafPinnedAtPile吗, leafPinned吗, pileNotOpen吗, pileOpen吗)
from mapFolding.genericNeedsNewHome import between吗, DOTitems, DOTkeys, DOTvalues, reverseLookup, thisHasThat吗, thisNotHaveThat吗
from more_itertools import flatten, one
from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable, Iterator, Sequence
	from hunterMakesPy import CallableFunction
	from mapFolding._e.dataBaskets import EliminationState
	from mapFolding._e.theTypes import DimensionIndex, Folding, Leaf, LeafOptions, LeafSpace, Pile, PinnedLeaves, UndeterminedPiles

#======== Boolean filters =======================

@syntacticCurry
def atPilePinLeafSafetyFilter(permutationSpace: PermutationSpace, pile: Pile, leaf: Leaf) -> bool:
	"""Return `True` if it is safe to call `atPilePinLeaf(permutationSpace, pile, leaf)`.

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
	return leafPinnedAtPile吗(permutationSpace, leaf, pile) or (pileOpen吗(permutationSpace, pile) and leafNotPinned吗(permutationSpace, leaf))

@syntacticCurry
def disqualifyPinningLeafAtPile(state: EliminationState, leaf: Leaf) -> bool:
	return any((
		leafPinned吗(state.permutationSpace, leaf)
		, pileNotOpen吗(state.permutationSpace, state.pile)
		, state.pile not in getLeafDomain(state, leaf),
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
	isPinned: Callable[[PermutationSpace], bool] = leafPinnedAtPile吗(leaf=leaf, pile=pile)
	grouped: dict[bool, list[PermutationSpace]] = toolz_groupby(isPinned, listPermutationSpace)
	return (grouped.get(False, []), grouped.get(True, []))

def moveFoldingToListFolding(state: EliminationState) -> EliminationState:
	listPermutationSpace: deque[PermutationSpace] = state.listPermutationSpace.copy()
	state.listPermutationSpace = deque()
	for permutationSpace in listPermutationSpace:
		if any(map(leafNotPinned吗(permutationSpace), range(state.leavesTotal))):
			state.listPermutationSpace.append(permutationSpace)
		else:
			folding: Folding = makeFolding(permutationSpace, ())
			state.listFolding.append(folding)
	return state

#======== Pin a `Leaf` in a `PermutationSpace` or `Folding` =======================
# NOTE This section ought to contain all functions based on the "Elimination" algorithm that pin a `Leaf` in a `PermutationSpace` or `Folding`.

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
	deconstructPermutationSpaceAtPile
	"""
	return PermutationSpace(associate(permutationSpace, pile, leaf))

def makeFolding(permutationSpace: PermutationSpace, leavesToInsert: Sequence[Leaf]) -> Folding:
	pilesToInsert: Iterator[Pile] = DOTkeys(extractUndeterminedPiles(permutationSpace))
	# NOTE `cast` because the type checkers cannot possible know that the prior logic leads to all int.
	return tuple(DOTvalues(dict(sorted(DOTitems(cast("PinnedLeaves", merge(permutationSpace, dict(zip(pilesToInsert, leavesToInsert, strict=True)))))))))

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
	if (leaf := DOTgetPileIfLeaf(permutationSpace, pile)) is not None:
		deconstructedPermutationSpace: dict[Leaf, PermutationSpace] = {leaf: permutationSpace}
	else:
		pin: Callable[[Leaf], PermutationSpace] = atPilePinLeaf(permutationSpace, pile)
		leafCanBePinned: Callable[[Leaf], bool] = leafNotPinned吗(permutationSpace)
		deconstructedPermutationSpace = {leaf: pin(leaf) for leaf in filter(leafCanBePinned, leavesToPin)}
	return deconstructedPermutationSpace

def deconstructPermutationSpaceByDomainOfLeaf(permutationSpace: PermutationSpace, leaf: Leaf, leafDomain: Iterable[Pile]) -> deque[PermutationSpace]:
	"""Pin `leaf` at each open `pile` in the domain of `leaf`.

	Return a `deque` of `PermutationSpace` with either `permutationSpace` because `leaf` is already pinned or one `PermutationSpace` for each
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
	deconstructedPermutationSpace : deque[PermutationSpace]
		Deque of `PermutationSpace` dictionaries with `leaf` pinned at each open `pile` in `leafDomain`.
	"""
	if leafNotPinned吗(permutationSpace, leaf):
		pileOpen: Callable[[int], bool] = pileOpen吗(permutationSpace)
		leafInPileRange: Callable[[int], bool] = compose(leafInLeafOptions吗(leaf), partial(DOTgetPileIfLeafOptions, permutationSpace, default=bit_mask(len(permutationSpace))))
		pinLeafAt: Callable[[int], PermutationSpace] = atPilePinLeaf(permutationSpace, leaf=leaf)
		deconstructedPermutationSpace: deque[PermutationSpace] = deque(map(pinLeafAt, filter(leafInPileRange, filter(pileOpen, leafDomain))))
	else:
		deconstructedPermutationSpace = deque([permutationSpace])
	return deconstructedPermutationSpace

def deconstructPermutationSpaceByDomainsCombined(permutationSpace: PermutationSpace, leaves: Sequence[Leaf], leavesDomain: Iterable[Sequence[Pile]]) -> deque[PermutationSpace]:
	"""Prototype."""  # noqa: DOC201
	deconstructedPermutationSpace: deque[PermutationSpace] = deque()

	def pileOpenByIndex(index: int) -> CallableFunction[[Sequence[Pile]], bool]:
		def workhorse(domain: Sequence[Pile]) -> bool:
			return pileOpen吗(permutationSpace, domain[index])
		return workhorse

	def leafInPileRangeByIndex(index: int) -> CallableFunction[[Sequence[Pile]], bool]:
		def workhorse(domain: Sequence[Pile]) -> bool:
			leafOptions: LeafOptions = raiseIfNone(DOTgetPileIfLeafOptions(permutationSpace, domain[index], default=bit_mask(len(permutationSpace))))
			return leafInLeafOptions吗(leaves[index], leafOptions)
		return workhorse

	def isPinnedAtPileByIndex(leaf: Leaf, index: int) -> CallableFunction[[Sequence[Pile]], bool]:
		def workhorse(domain: Sequence[Pile]) -> bool:
			return leafPinnedAtPile吗(permutationSpace, leaf, domain[index])
		return workhorse

	if any(map(leafNotPinned吗(permutationSpace), leaves)):
		for index in range(len(leaves)):
			"""Redefine leavesDomain by filtering out domains that are not possible with the current `PermutationSpace`."""
			if leafNotPinned吗(permutationSpace, leaves[index]):
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
		deconstructedPermutationSpace = deque([permutationSpace])

	return deconstructedPermutationSpace

#======== Bulk modifications =======================

def deconstructListPermutationSpaceAtPile(listPermutationSpace: Iterable[PermutationSpace], pile: Pile, leavesToPin: Iterable[Leaf]) -> Iterator[PermutationSpace]:
	"""Expand every dictionary in `listPermutationSpace` at `pile` into all pinning variants.

	Applies `deconstructPermutationSpaceAtPile` element-wise, then flattens the nested value collections (each a mapping leaf -> dictionary)
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
	deconstructPermutationSpaceAtPile
	"""
	return flatten(map(DOTvalues, map(deconstructPermutationSpaceAtPile, listPermutationSpace, repeat(pile), repeat(leavesToPin))))

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

		if leafPinnedAtPile吗(permutationSpace, leaf_k, pile_k):
			listPermutationSpace_kPinnedAt_pile_k.append(permutationSpace)
		elif leafPinned吗(permutationSpace, leaf_k) or pileNotOpen吗(permutationSpace, pile_k) or leaf_k not in rangePile_k:
			listPermutationSpaceCompleted.append(permutationSpace)
		else:
			leafOptionsAt_pile_k: LeafOptions = raiseIfNone(DOTgetPileIfLeafOptions(permutationSpace, pile_k, default=bit_mask(len(permutationSpace))))
			if leafInLeafOptions吗(leaf_k, leafOptionsAt_pile_k):
				listPermutationSpace_kPinnedAt_pile_k.append(atPilePinLeaf(permutationSpace, pile_k, leaf_k))
				leafSpaceWithoutLeaf_k = JeanValjean(bit_clear(leafOptionsAt_pile_k, leaf_k))
				if leafSpaceWithoutLeaf_k is not None:
					listPermutationSpaceCompleted.append(PermutationSpace(associate(permutationSpace, pile_k, leafSpaceWithoutLeaf_k)))
			else:
				listPermutationSpaceCompleted.append(permutationSpace)

		iterator_kPinnedAt_pile_k: Iterable[PermutationSpace] = listPermutationSpace_kPinnedAt_pile_k
		for pile_r in domain_r:
			iterator_kPinnedAt_pile_k = excludeLeafAtPile(iterator_kPinnedAt_pile_k, leaf_r, pile_r, ())

		listPermutationSpace.extend(listPermutationSpaceCompleted)
		listPermutationSpace.extend(iterator_kPinnedAt_pile_k)

	state.listPermutationSpace = listPermutationSpace

	return reduceAllPermutationSpace(state)

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
	deconstructPermutationSpaceAtPile : Performs the expansion for one dictionary.
	requireLeafPinnedAtPile : Complementary operation that forces a `leaf` at a `pile`.
	"""
	del leavesToPin

	for permutationSpace in listPermutationSpace:
		if leafPinnedAtPile吗(permutationSpace, leaf, pile):
			continue

		if (leafOptionsAtPile := DOTgetPileIfLeafOptions(permutationSpace, pile)) is None:
			yield permutationSpace
			continue

		if leafInLeafOptions吗(leaf, leafOptionsAtPile):
			leafSpaceWithoutLeaf = JeanValjean(bit_clear(leafOptionsAtPile, leaf))
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
	deconstructPermutationSpaceAtPile, excludeLeafAtPile
	"""
	listLeafAtPile: deque[PermutationSpace] = deque()

	for permutationSpace in listPermutationSpace:
		if leafPinnedAtPile吗(permutationSpace, leaf, pile):
			listLeafAtPile.append(permutationSpace)
		elif leafPinned吗(permutationSpace, leaf) or pileNotOpen吗(permutationSpace, pile):
			continue
		else:
			leafOptionsAtPile: LeafOptions = raiseIfNone(DOTgetPileIfLeafOptions(permutationSpace, pile, default=bit_mask(len(permutationSpace))))
			if leafInLeafOptions吗(leaf, leafOptionsAtPile):
				listLeafAtPile.append(atPilePinLeaf(permutationSpace, pile, leaf))

	return listLeafAtPile

def segregateLeafByDeconstructingListPermutationSpaceAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile, leavesToPin: Iterable[Leaf]) -> Iterator[tuple[PermutationSpace, tuple[PermutationSpace, ...]]]:
	for permutationSpace in listPermutationSpace:
		deconstructedPermutationSpaceAtPile: dict[Leaf, PermutationSpace] = deconstructPermutationSpaceAtPile(permutationSpace, pile, leavesToPin)
		leafPinnedAtPile: PermutationSpace = deconstructedPermutationSpaceAtPile.pop(leaf)
		yield (leafPinnedAtPile, tuple(deconstructedPermutationSpaceAtPile.values()))

#======== Reducing `LeafOptions` ===============================

def reduceAllPermutationSpace(state: EliminationState, listFunctionsReduction: Sequence[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = ()) -> EliminationState:
	"""Reduce permutation space by iteratively applying constraint propagation.

	You can use this function to shrink the search space for map-folding computations by applying
	multiple constraint-propagation strategies in a loop until the permutation space stabilizes. The
	function orchestrates the unified constraint-satisfaction algorithm implemented across the
	specialized `_reducePermutationSpace_*` functions in this module. Each iteration applies each
	constraint type in sequence. The function continues iterating until the total permutation space
	size stops decreasing.

	The function is the orchestrator for the constraint-propagation system. The function treats the
	specialized reduction functions as interdependent components of a single large algorithm, not as
	independent transformations. Each function assumes other functions will run afterward to propagate
	newly discovered constraints.

	Parameters
	----------
	state : EliminationState
		A data basket containing `listPermutationSpace` to reduce and supporting computed properties.

	Returns
	-------
	updatedState : EliminationState
		The `state` with `state.listPermutationSpace` reduced by constraint propagation.
	"""
	# ------------ Initialize `listPermutationSpace` ------------------------------
	if not listFunctionsReduction:
		listFunctionsReduction = (
			reducePermutationSpace_LeafIsPinned,
			reducePermutationSpace_leafDomainOf1,
			reducePermutationSpace_nakedSubset,
			# TODO I cannot think of a reason why this ought not to be a general function. So, the
			# test failures suggest an implementation error or bug. A001415(3) and A001416(2), which
			# happen to be the same mapShape, (2, 3).
			# reducePermutationSpace_CrossedCreases,
		)

	listPermutationSpace: deque[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = deque()
	listPermutationSpaceIrreducible: deque[PermutationSpace] = deque()

	# TODO (dissatisfied) I'm not satisfied with this function, and I _think_ the fundamental issue is the lack of no-op.

	while listPermutationSpace:
		# ------------ Initialize `permutationSpace` ------------------------------
		# TODO (dissatisfied) `... | None` is not natural.
		permutationSpace: PermutationSpace | None = listPermutationSpace.pop()
		# NOTE `sumPermutationSpace` detects _any_ change in the permutation space; not to be confused with `sum首`.
		# NOTE reminder: _all_ changes in a permutation space are reductions in the probability space.
		sumPermutationSpace: Leaf | LeafOptions = sum(permutationSpace.values())
		# TODO I feel like this could be a dynamic self-ordering queue based on how often a
		# `reducePermutationSpace` function returns `None` or an altered `PermutationSpace`. Even
		# better would be if the self-ordering persisted across sessions so that previous computations
		# inform the current computation. At the very least, I think it would be computationally
		# inexpensive to order the queue based on the functions that most recently returned `None` or
		# an altered `PermutationSpace`. That would be a MRU, most recently used, queue, right? If LRU
		# is cheap, then is MRU cheap?
		functionsReduction: deque[Callable[[EliminationState, PermutationSpace], PermutationSpace | None]] = deque(listFunctionsReduction)
		keepGoing: bool = True

		while keepGoing:
			reducePermutationSpace: Callable[[EliminationState, PermutationSpace], PermutationSpace | None] = functionsReduction.popleft()
			# TODO (dissatisfied) `raiseIfNone` is _only_ necessary because of `... | None`.
			permutationSpace = reducePermutationSpace(state, raiseIfNone(permutationSpace))

			if not permutationSpace:
				# TODO (dissatisfied) In this previous version of this function, this check was
				# handled by `more_itertools.filter_map`. In this version, it is necessary because the
				# signatures of `_reducePermutationSpace_*` are `reducePermutationSpace:
				# Callable[[EliminationState, PermutationSpace], PermutationSpace | None]` passing
				# `PermutationSpace | None` will raise an exception instead of no-op. I will NOT add a
				# guard to every `_reducePermutationSpace_*` function to handle `None`.
				keepGoing = False
			elif sumPermutationSpace != sum(permutationSpace.values()):
				# TODO I suspect there are faster ways to check if an object has been altered.
				# NOTE Reset the `functionsReduction` queue.
				functionsReduction = deque(listFunctionsReduction)
				sumPermutationSpace = sum(permutationSpace.values())
			elif not functionsReduction:
				# NOTE due to the previous `elif`, `... and (sumPermutationSpace == sum(permutationSpace.values()))` is implied.
				listPermutationSpaceIrreducible.append(permutationSpace)
				keepGoing = False

	state.listPermutationSpace.extend(listPermutationSpaceIrreducible)

	return state

#-------- Shared logic -----------------------------------------

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
		Dictionary mapping pile indices to leaf indices or `LeafOptions` bitsets. The function
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
			# NOTE quick return
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
	pileOf_kCrease: Pile = errorL33T
	pileOf_rCrease: Pile = errorL33T
	pilesForbidden: Iterable[Pile] = []
	permutationSpaceHasNewLeaf: bool = True

	generators: deque[CartesianProduct[tuple[DimensionIndex, PinnedLeaves, tuple[tuple[Pile, Leaf], tuple[Pile, Leaf]]]]] = deque()
	for dimension in range(state.dimensionsTotal):
		parityEven: PinnedLeaves = {}
		parityOdd: PinnedLeaves = {}
		for pileLeaf in DOTitems(extractPinnedLeaves(permutationSpace)):
			if oddLeaf吗(state.mapShape, pileLeaf[1], dimension):
				parityOdd.update((pileLeaf,))
			else:
				parityEven.update((pileLeaf,))
		generators.append(CartesianProduct((dimension,), (parityOdd,), combinations(parityEven.items(), 2)))
		generators.append(CartesianProduct((dimension,), (parityEven,), combinations(parityOdd.items(), 2)))

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		sum首: int = sum(map(dimensionNearest首, permutationSpace.values()))

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
					return None
				continue

			else:  # elif not leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
				continue

			if not (permutationSpace := reduceLeafSpace(state, permutationSpace
					, DOTitems(filterPile(thisHasThat吗(pilesForbidden), extractUndeterminedPiles(permutationSpace)))
					, leafAntiOptions
			)):
				return None

		if sum(map(dimensionNearest首, permutationSpace.values())) < sum首:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

def reducePermutationSpace_LeafIsPinned(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to propagate leaf pinning constraints.

	I use this constraint encoder to enforce that every pinned leaf can appear at only one pile.
	For every leaf pinned at a pile, I remove that leaf from `LeafOptions` at all other piles.
	When `LeafOptions` at a pile reduces to a single leaf, I convert `pile: leafOptions` to
	`pile: leaf` (pinning the leaf).

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
		leavesPinned, pilesUndetermined = bifurcatePermutationSpace(permutationSpace)
		sum首: int = sum(map(dimensionNearest首, permutationSpace.values()))
		# NOTE using the walrus operator here `if not (permutationSpace := _reduceLeafSpace...` means
		# that type checkers are ok with `permutationSpace: PermutationSpace`. If I assigned without
		# the `if` check, `permutationSpace = _reduceLeafSpace...`, then the annotation would need to
		# be `permutationSpace: PermutationSpace | None` because `_reduceLeafSpace` can return `None`.
		# Furthermore, not creating an intermediate variable is more efficient.
		if not (permutationSpace := reduceLeafSpace(
				state, permutationSpace, DOTitems(pilesUndetermined), makeLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned))
		)):
			return None
		if sum(map(dimensionNearest首, permutationSpace.values())) < sum首:
			# NOTE 2026 July 7 Does this produces false positives?
			# 1. If the value is a `Leaf`, then `dimensionNearest首(leaf)` cannot possibly change.
			# 2. If the value starts as `LeafOptions`, and if the value remains `LeafOptions`, then
			#    `dimensionNearest首(leafOptions)` will stay the same (e.g., `== leavesTotal`) even if
			#    the size of `LeafOptions` domain is reduced.
			# 3. If the value starts as `LeafOptions`, but the value becomes a `Leaf`, then
			#    `dimensionNearest首(leafOptions) = leavesTotal`, but `dimensionNearest首(leaf) <
			#    dimensionsTotal = log2(leavesTotal)`
			# Therefore, it is a precise measurement of whether a new leaf has been pinned.
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
		sum首: int = sum(map(dimensionNearest首, permutationSpace.values()))

		pilesUndetermined: UndeterminedPiles = extractUndeterminedPiles(permutationSpace)

		groupByLeafOptions: dict[LeafOptions, set[Pile]] = {}
		for pile, leafOptions in filterLeafOptions(thisNotHaveThat吗(unique(pilesUndetermined.values())), pilesUndetermined).items():
			groupByLeafOptions.setdefault(leafOptions, set()).add(pile)

		for leafOptions, setPiles in DOTitems(itemfilter(lambda groupBy: (howManyLeavesInLeafOptions(groupBy[leafOptionsKey])) == len(groupBy[piles]), groupByLeafOptions)):

			if not (
				permutationSpace := reduceLeafSpace(
					state
					, permutationSpace
					, DOTitems(filterPile(thisNotHaveThat吗(setPiles), pilesUndetermined))
					, makeLeafAntiOptions(state.leavesTotal, getIteratorOfLeaves(leafOptions))
				)
			):
				return None

		if sum(map(dimensionNearest首, permutationSpace.values())) < sum首:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

#-------- Functions that do NOT use the shared logic -----------------------------------------

def reducePermutationSpace_leafDomainOf1(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to detect and pin leaves with domain size one.

	I use this constraint encoder to detect leaves that can appear at only one pile (domain size
	one) and pin those leaves. I compute the domain size for each leaf by counting how many piles
	contain that leaf (either pinned or in `LeafOptions`). When a leaf appears at exactly one
	pile, I pin that leaf at that pile using `atPilePinLeaf` [1] and propagate the pinning using
	`reducePermutationSpace_leafDomainOf1`.

	The function also validates that every leaf has nonzero domain size. When any leaf has zero
	domain (cannot appear anywhere), I invalidate `permutationSpace` by returning `None`.

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
	[1] mapFolding._e.pinIt.atPilePinLeaf
	"""
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		leavesPinned, pilesUndetermined = bifurcatePermutationSpace(permutationSpace)

		counterLeafDomainSize: Counter[Leaf] = Counter(chain(chain.from_iterable(map(getIteratorOfLeaves, DOTvalues(pilesUndetermined))), DOTvalues(leavesPinned)))

		if set(range(state.leavesTotal)).difference(counterLeafDomainSize.keys()):
			return None

		leavesWithDomainOf1: set[Leaf] = set(DOTkeys(filterValue((1).__eq__, counterLeafDomainSize))).difference(leavesPinned.values()).difference([state.leavesTotal])
		if leavesWithDomainOf1:
			leaf: Leaf = leavesWithDomainOf1.pop()
			sherpa: PermutationSpace | None = reducePermutationSpace_LeafIsPinned(state, atPilePinLeaf(permutationSpace, one(DOTkeys(filterLeaf(leafInLeafOptions吗(leaf), pilesUndetermined))), leaf))
			if (sherpa is None) or (not sherpa):
				return None
			else:
				permutationSpace = sherpa
			permutationSpaceHasNewLeaf = True
	return permutationSpace

#======== Initialization =====================

def addMissingLeafOptionsToPermutationSpace(state: EliminationState) -> EliminationState:
	state.permutationSpace = PermutationSpace(merge(mapLeaf(compose(raiseIfNone, JeanValjean), getDictionaryLeafOptions(state)), state.permutationSpace))
	return state
