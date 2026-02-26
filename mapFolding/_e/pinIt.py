"""Generalized pinning functions in the "Elimination" algorithm for any `mapShape`.

Functions for 2^n-dimensional maps must go in other modules.

The development of this generalized module is severely hampered, however. Functions for 2^n-dimensional maps have a "beans and
cornbread" problem that was difficult for me to "solve"--due to my programming skills. If I were able to decouple the "beans and
cornbread" solution from the 2^n-dimensional functions, I would generalize more functions and move them here.
"""
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from gmpy2 import bit_clear, bit_mask
from hunterMakesPy import inclusive, raiseIfNone
from itertools import chain, repeat
from mapFolding._e import (
	bifurcatePermutationSpace, dimensionNearest首, DOTgetPileIfLeaf, DOTgetPileIfLeafOptions, DOTitems, DOTkeys, DOTvalues,
	Folding, getIteratorOfLeaves, getLeafDomain, getLeafOptions, howManyLeavesInLeafOptions, JeanValjean, Leaf,
	LeafOptions, leafOptionsAND, LeafSpace, makeLeafAntiOptions, PermutationSpace, Pile, UndeterminedPiles)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import (
	between吗, extractPinnedLeaves, extractUndeterminedPiles, leafIsInPileRange, leafIsNotPinned, leafIsPinned,
	leafIsPinnedAtPile, pileIsNotOpen, pileIsOpen, thisIsALeaf, thisNotHaveThat)
from math import prod
from more_itertools import filter_map, flatten, ilen, one
from tlz.curried import map as toolz_map  # pyright: ignore[reportMissingModuleSource]
from tlz.dicttoolz import (  # pyright: ignore[reportMissingModuleSource]
	assoc as associate, itemfilter, keyfilter, valfilter)
from tlz.functoolz import compose, curry as syntacticCurry  # pyright: ignore[reportMissingModuleSource]
from tlz.itertoolz import groupby as toolz_groupby, unique  # pyright: ignore[reportMissingModuleSource]

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
	return leafIsPinnedAtPile(permutationSpace, leaf, pile) or (pileIsOpen(permutationSpace, pile) and leafIsNotPinned(permutationSpace, leaf))

@syntacticCurry
def disqualifyPinningLeafAtPile(state: EliminationState, leaf: Leaf) -> bool:
	return any((
		leafIsPinned(state.permutationSpace, leaf),
		pileIsNotOpen(state.permutationSpace, state.pile),
		state.pile not in getLeafDomain(state, leaf),
	))

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
	isPinned: Callable[[PermutationSpace], bool] = leafIsPinnedAtPile(leaf=leaf, pile=pile)
	grouped: dict[bool, list[PermutationSpace]] = toolz_groupby(isPinned, listPermutationSpace)
	return (grouped.get(False, []), grouped.get(True, []))

def moveFoldingToListFolding(state: EliminationState) -> EliminationState:
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace.copy()
	state.listPermutationSpace = []
	for permutationSpace in listPermutationSpace:
		if any(map(leafIsNotPinned(permutationSpace), range(state.leavesTotal))):
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
	return associate(permutationSpace, pile, leaf)

# TODO more flexible.
def makeFolding(permutationSpace: PermutationSpace, leavesToInsert: tuple[Leaf, ...]) -> Folding:
	if leavesToInsert:
		permutand: Iterator[Leaf] = iter(leavesToInsert)
		pilesTotal: int = ilen(filter(thisIsALeaf, DOTvalues(permutationSpace))) + len(leavesToInsert)
		# pilesTotal: int = len(extractPinnedLeaves(permutationSpace)) + len(leavesToInsert)  # noqa: ERA001
		folding: Folding = tuple([
			leafOrLeafRange if (leafOrLeafRange := DOTgetPileIfLeaf(permutationSpace, pile)) is not None else next(permutand)
			for pile in range(pilesTotal)
		])
	else:
		folding = tuple(DOTvalues(extractPinnedLeaves(permutationSpace)))
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
	if (leaf := DOTgetPileIfLeaf(permutationSpace, pile)) is not None:
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
		leafInPileRange: Callable[[int], bool] = compose(leafIsInPileRange(leaf), partial(DOTgetPileIfLeafOptions, permutationSpace, default=bit_mask(len(permutationSpace))))
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
			leafOptions: LeafOptions = raiseIfNone(DOTgetPileIfLeafOptions(permutationSpace, domain[index], default=bit_mask(len(permutationSpace))))
			return leafIsInPileRange(leaves[index], leafOptions)
		return workhorse

	def isPinnedAtPileByIndex(leaf: Leaf, index: int) -> Callable[[tuple[Pile, ...]], bool]:
		def workhorse(domain: tuple[Pile, ...]) -> bool:
			return leafIsPinnedAtPile(permutationSpace, leaf, domain[index])
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
	return  flatten(map(DOTvalues, map(deconstructPermutationSpaceAtPile, listPermutationSpace, repeat(pile), repeat(leavesToPin))))

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
	listPermutationSpace: list[PermutationSpace] = []

	if domain_r is None:
		domain_r = getLeafDomain(state, leaf_r)
	domain_r = tuple(filter(between吗(0, pile_k - inclusive), domain_r))

	if rangePile_k is None:
		rangePile_k = getIteratorOfLeaves(getLeafOptions(state, pile_k))
	rangePile_k = frozenset(rangePile_k)

	for permutationSpace in state.listPermutationSpace:
		listPermutationSpace_kPinnedAt_pile_k: list[PermutationSpace] = []
		listPermutationSpaceCompleted: list[PermutationSpace] = []

		if leafIsPinnedAtPile(permutationSpace, leaf_k, pile_k):
			listPermutationSpace_kPinnedAt_pile_k.append(permutationSpace)
		elif leafIsPinned(permutationSpace, leaf_k) or pileIsNotOpen(permutationSpace, pile_k) or leaf_k not in rangePile_k:
			listPermutationSpaceCompleted.append(permutationSpace)
		else:
			leafOptionsAt_pile_k: LeafOptions = raiseIfNone(DOTgetPileIfLeafOptions(permutationSpace, pile_k, default=bit_mask(len(permutationSpace))))
			if leafIsInPileRange(leaf_k, leafOptionsAt_pile_k):
				listPermutationSpace_kPinnedAt_pile_k.append(atPilePinLeaf(permutationSpace, pile_k, leaf_k))
				leafSpaceWithoutLeaf_k = JeanValjean(bit_clear(leafOptionsAt_pile_k, leaf_k))
				if leafSpaceWithoutLeaf_k is not None:
					listPermutationSpaceCompleted.append(associate(permutationSpace, pile_k, leafSpaceWithoutLeaf_k))
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

	Returns
	-------
	listPermutationSpace : Iterable[PermutationSpace]
		Expanded / filtered list respecting the exclusion constraint.

	See Also
	--------
	deconstructPermutationSpaceAtPile : Performs the expansion for one dictionary.
	requireLeafPinnedAtPile : Complementary operation that forces a `leaf` at a `pile`.
	"""
	del leavesToPin

	for permutationSpace in listPermutationSpace:
		if leafIsPinnedAtPile(permutationSpace, leaf, pile):
			continue

		if (leafOptionsAtPile := DOTgetPileIfLeafOptions(permutationSpace, pile)) is None:
			yield permutationSpace
			continue

		if leafIsInPileRange(leaf, leafOptionsAtPile):
			leafSpaceWithoutLeaf = JeanValjean(bit_clear(leafOptionsAtPile, leaf))
			if leafSpaceWithoutLeaf is not None:
				yield associate(permutationSpace, pile, leafSpaceWithoutLeaf)
		else:
			yield permutationSpace

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
	deconstructPermutationSpaceAtPile, excludeLeafAtPile
	"""
	listLeafAtPile: list[PermutationSpace] = []

	for permutationSpace in listPermutationSpace:
		if leafIsPinnedAtPile(permutationSpace, leaf, pile):
			listLeafAtPile.append(permutationSpace)
		elif leafIsPinned(permutationSpace, leaf) or pileIsNotOpen(permutationSpace, pile):
			continue
		else:
			leafOptionsAtPile: LeafOptions = raiseIfNone(DOTgetPileIfLeafOptions(permutationSpace, pile, default=bit_mask(len(permutationSpace))))
			if leafIsInPileRange(leaf, leafOptionsAtPile):
				listLeafAtPile.append(atPilePinLeaf(permutationSpace, pile, leaf))

	return listLeafAtPile

def segregateLeafByDeconstructingListPermutationSpaceAtPile(listPermutationSpace: Iterable[PermutationSpace], leaf: Leaf, pile: Pile, leavesToPin: Iterable[Leaf]) -> Iterator[tuple[PermutationSpace, tuple[PermutationSpace, ...]]]:
	for permutationSpace in listPermutationSpace:
		deconstructedPermutationSpaceAtPile: dict[Leaf, PermutationSpace] = deconstructPermutationSpaceAtPile(permutationSpace, pile, leavesToPin)
		leafPinnedAtPile: PermutationSpace = deconstructedPermutationSpaceAtPile.pop(leaf)
		yield (leafPinnedAtPile, tuple(deconstructedPermutationSpaceAtPile.values()))

#======== Reducing `LeafOptions` ===============================

def reduceAllPermutationSpace(state: EliminationState) -> EliminationState:
	"""Reduce permutation space by iteratively applying constraint propagation.

	You can use this function to shrink the search space for map-folding computations by applying
	multiple constraint-propagation strategies in a loop until the permutation space stabilizes.
	The function orchestrates the unified constraint-satisfaction algorithm implemented across
	the specialized `_reducePermutationSpace_*` functions in this module. Each iteration applies
	each constraint type in sequence. The function continues iterating until the total permutation
	space size stops decreasing.

	The function is the orchestrator for the constraint-propagation system. The function treats
	the specialized reduction functions as interdependent components of a single large algorithm,
	not as independent transformations. Each function assumes other functions will run afterward
	to propagate newly discovered constraints.

	Algorithm Details
	-----------------
	The function applies these constraint types in sequence:

	1. Crease adjacency (via `_reducePermutationSpace_byCrease`)
	2. Pinned leaf propagation (via `_reducePermutationSpace_LeafIsPinned`)
	3. Head-before-tail ordering (via `_reducePermutationSpace_HeadsBeforeTails`)
	4. Conditional predecessors (via `_reducePermutationSpace_ConditionalPredecessors`)
	5. Crossed crease detection (via `_reducePermutationSpace_CrossedCreases`)
	6. Non-consecutive dimensions (via `_reducePermutationSpace_noConsecutiveDimensions`)
	7. Domain size one detection (via `_reducePermutationSpace_leafDomainOf1`)
	8. Naked subset elimination (via `_reducePermutationSpace_nakedSubset`)

	The function measures the total permutation space size before and after each full iteration.
	When the size stops decreasing, the function terminates and returns `state` with the reduced
	`state.listPermutationSpace`.

	The function uses `filter_map` [1] to apply each reduction function, automatically filtering
	out invalidated permutation spaces (those that return `None`).

	Parameters
	----------
	state : EliminationState
		A data basket containing `listPermutationSpace` to reduce and supporting computed
		properties.

	Returns
	-------
	updatedState : EliminationState
		The `state` with `state.listPermutationSpace` reduced by constraint propagation.

	Examples
	--------
	>>> from mapFolding._e.algorithms.eliminationCrease import doTheNeedful
	>>> sherpa = moveFoldingToListFolding(
	...     removeIFFViolationsFromEliminationState(
	...         reduceAllPermutationSpaceInEliminationState(sherpa)))

	References
	----------
	[1] more_itertools.filter_map
		https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.filter_map

	"""
	def prodOfDOTvalues(listLeafOptions: Iterable[LeafOptions]) -> int:
		return prod(map(howManyLeavesInLeafOptions, listLeafOptions))

	permutationsPermutationSpaceTotal: Callable[[list[PermutationSpace]], int] = compose(sum, toolz_map(compose(prodOfDOTvalues, DOTvalues, extractUndeterminedPiles)))
	permutationSpaceTotal: int = permutationsPermutationSpaceTotal(state.listPermutationSpace)
	continueReduction: bool = True

	while continueReduction:
		continueReduction = False

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		state.listPermutationSpace = []
		state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_LeafIsPinned(state), listPermutationSpace))

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		state.listPermutationSpace = []
		state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_leafDomainOf1(state), listPermutationSpace))

		listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
		state.listPermutationSpace = []
		state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_nakedSubset(state), listPermutationSpace))

		permutationSpaceTotalReduced: int = permutationsPermutationSpaceTotal(state.listPermutationSpace)

		if permutationSpaceTotalReduced < permutationSpaceTotal:
			continueReduction = True
			permutationSpaceTotal = permutationSpaceTotalReduced

	return state

#-------- Shared logic -----------------------------------------

def _reduceLeafSpace(state: EliminationState, permutationSpace: PermutationSpace, pilesToUpdate: deque[tuple[Pile, LeafOptions]], leafAntiOptions: LeafOptions) -> PermutationSpace:
	"""I use this to update permutation space by removing forbidden leaves from piles.

	I use this shared subroutine to handle the mechanical work of updating `LeafOptions` at
	specified piles by removing forbidden leaves. All constraint encoders (`_reducePermutationSpace_*`)
	call this function to perform the actual updates. I process each pile in `pilesToUpdate`,
	remove leaves specified by `leafAntiOptions`, and propagate newly pinned leaves. I detect
	beans-without-cornbread configurations and pin the complementary cornbread leaf when
	appropriate.

	I do not return a `bool` for `permutationSpaceHasNewLeaf`. Calling functions compare
	`permutationSpace` properties before and after calling this function to detect whether new
	leaves were pinned.

	Algorithm Details
	-----------------
	For each pile in `pilesToUpdate`:

	1. Remove forbidden leaves by computing `leafOptionsAND(leafAntiOptions, leafOptions)`.
	2. Use `JeanValjean` [1] to convert the result to `LeafSpace` (either `Leaf` or
		`LeafOptions`).
	3. If the result is `None` (empty domain), invalidate `permutationSpace` by setting to `{}`.
	4. If the result is a `Leaf`, check for beans-without-cornbread configurations:
		- Beans-without-cornbread occurs when one member of a crease pair (beans/cornbread) is pinned but the adjacent crease neighbor (cornbread/beans) is not.
		- Pin the complementary cornbread leaf at the appropriate adjacent pile.
		- Set `permutationSpaceHasNewLeaf = True` to signal the calling function.

	When `permutationSpaceHasNewLeaf` becomes `True`, I call `_reducePermutationSpace_LeafIsPinned`
	to propagate the newly pinned leaf before returning.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.
	pilesToUpdate : deque[tuple[Pile, LeafOptions]]
		Piles to update with `pile` and existing `leafOptions`.
	leafAntiOptions : LeafOptions
		A bitset of leaves to remove from `LeafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace
		The updated `permutationSpace` if valid; otherwise an empty dictionary (invalid).

	Examples
	--------
	Calling functions detect `permutationSpaceHasNewLeaf` by comparing properties before and
	after:

	>>> sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
	>>> permutationSpace = _reduceLeafSpace(state, permutationSpace, pilesToUpdate, leafAntiOptions)
	>>> if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
	...     permutationSpaceHasNewLeaf = True

	References
	----------
	[1] mapFolding._e.JeanValjean
		Internal package reference

	"""
	permutationSpaceHasNewLeaf: bool = False
	while permutationSpace and pilesToUpdate and not permutationSpaceHasNewLeaf:
		pile, leafOptions = pilesToUpdate.pop()

		leafSpace: LeafSpace | None = JeanValjean(leafOptionsAND(leafAntiOptions, leafOptions))
		if leafSpace is not None:

			permutationSpace[pile] = leafSpace
			if thisIsALeaf(permutationSpace[pile]):
				permutationSpaceHasNewLeaf = True
		else:
			permutationSpace = {}

	if permutationSpace and permutationSpaceHasNewLeaf:
		sherpa: PermutationSpace | None = _reducePermutationSpace_LeafIsPinned(state, permutationSpace)
		if not sherpa:
			permutationSpace = {}
		else:
			permutationSpace = sherpa
	return permutationSpace

#-------- Functions that use the shared logic -----------------------------------------

@syntacticCurry
def _reducePermutationSpace_LeafIsPinned(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to propagate leaf pinning constraints.

	I use this constraint encoder to enforce that every pinned leaf can appear at only one pile.
	For every leaf pinned at a pile, I remove that leaf from `LeafOptions` at all other piles.
	When `LeafOptions` at a pile reduces to a single leaf, I convert `pile: leafOptions` to
	`pile: leaf` (pinning the leaf). When that creates a beans-without-cornbread configuration,
	I pin the complementary cornbread leaf at the appropriate adjacent pile.

	This function is the primary propagator for newly pinned leaves. All other constraint encoders
	call `_reduceLeafSpace`, which calls this function when new leaves are pinned. This function
	iteratively applies pinning until no new leaves are discovered.

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

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		leavesPinned, pilesUndetermined = bifurcatePermutationSpace(permutationSpace)
		sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
		if not (permutationSpace := _reduceLeafSpace(state, permutationSpace, deque(pilesUndetermined.items()), makeLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned)))):
			return None
		if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_nakedSubset(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
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

		pilesUndetermined: UndeterminedPiles = extractUndeterminedPiles(permutationSpace)

		groupByLeafOptions: dict[LeafOptions, set[Pile]] = {}
		for pile, leafOptions in valfilter(thisNotHaveThat(unique(pilesUndetermined.values())), pilesUndetermined).items():
			groupByLeafOptions.setdefault(leafOptions, set()).add(pile)

		dequeLeafOptionsAndPiles: deque[tuple[LeafOptions, set[Pile]]] = deque(DOTitems(
			itemfilter(lambda groupBy: (howManyLeavesInLeafOptions(groupBy[leafOptionsKey])) == len(groupBy[piles]), groupByLeafOptions)))

		while dequeLeafOptionsAndPiles and not permutationSpaceHasNewLeaf:
			leafOptions, setPiles = dequeLeafOptionsAndPiles.pop()

			sumBeforeReduction: int = sum(map(dimensionNearest首, permutationSpace.values()))
			if not (permutationSpace := _reduceLeafSpace(state, permutationSpace
					, pilesToUpdate = deque(DOTitems(keyfilter(thisNotHaveThat(setPiles), pilesUndetermined)))
					, leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, getIteratorOfLeaves(leafOptions))
				)):
				return None
			if sum(map(dimensionNearest首, permutationSpace.values())) < sumBeforeReduction:
				permutationSpaceHasNewLeaf = True

	return permutationSpace

#-------- Functions that do NOT use the shared logic -----------------------------------------

@syntacticCurry
def _reducePermutationSpace_leafDomainOf1(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""I use this to detect and pin leaves with domain size one.

	I use this constraint encoder to detect leaves that can appear at only one pile (domain size
	one) and pin those leaves. I compute the domain size for each leaf by counting how many piles
	contain that leaf (either pinned or in `LeafOptions`). When a leaf appears at exactly one
	pile, I pin that leaf at that pile using `atPilePinLeaf` [1] and propagate the pinning using
	`_reducePermutationSpace_LeafIsPinned`.

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
		Internal package reference

	"""
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		leavesPinned, pilesUndetermined = bifurcatePermutationSpace(permutationSpace)

		counterLeafDomainSize: Counter[Leaf] = Counter(chain(chain.from_iterable(map(getIteratorOfLeaves, DOTvalues(pilesUndetermined))), DOTvalues(leavesPinned)))

		if set(range(state.leavesTotal)).difference(counterLeafDomainSize.keys()):
			return None

		leavesWithDomainOf1: set[Leaf] = set(DOTkeys(valfilter((1).__eq__, counterLeafDomainSize))).difference(leavesPinned.values()).difference([state.leavesTotal])
		if leavesWithDomainOf1:
			leaf: Leaf = leavesWithDomainOf1.pop()
			sherpa: PermutationSpace | None = _reducePermutationSpace_LeafIsPinned(state, atPilePinLeaf(permutationSpace, one(DOTkeys(valfilter(leafIsInPileRange(leaf), pilesUndetermined))), leaf))
			if (sherpa is None) or (not sherpa):
				return None
			else:
				permutationSpace = sherpa
			permutationSpaceHasNewLeaf = True
	return permutationSpace
