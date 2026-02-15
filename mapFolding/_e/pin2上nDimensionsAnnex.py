# ruff: noqa: ERA001
from collections import Counter, deque
from cytoolz.dicttoolz import keyfilter, valfilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from gmpy2 import bit_flip, bit_test as isBit1吗
from itertools import chain, combinations, product as CartesianProduct
from mapFolding import inclusive
from mapFolding._e import (
	bifurcatePermutationSpace, dimensionNearestTail, dimensionNearest首, DOTitems, DOTkeys, DOTvalues,
	getDictionaryConditionalLeafPredecessors, getIteratorOfLeaves, getLeafAntiOptions, getLeavesCreaseAnte,
	getLeavesCreasePost, JeanValjean, Leaf, LeafOptions, leafOptionsAND, LeafSpace, mapShapeIs2上nDimensions,
	PermutationSpace, Pile, PinnedLeaves, UndeterminedPiles, 一, 零, 首一, 首零一)
from mapFolding._e.algorithms.iff import thisIsAViolation
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import (
	between吗, extractPinnedLeaves, extractUndeterminedPiles, leafIsInLeafOptions, leafIsPinned, mappingHasKey,
	notLeafOriginOrLeaf零, notPileLast, thisHasThat, thisIsALeaf, thisIsLeafOptions)
from mapFolding._e.pinIt import atPilePinLeaf, disqualifyPinningLeafAtPile
from more_itertools import filter_map, one, pairwise, split_when, triplewise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator

#======== Boolean filters ======================================

# SEMIOTICS `leafIsOddInDimension` clarify and converge with the generalized version.
@syntacticCurry
def leafIsOddInDimension(leaf: Leaf, dimension: int) -> bool:
	"""A specialized version of parity checking for 2^n-dimensional maps."""
	return isBit1吗(leaf, dimension)

#======== Reducing `LeafOptions` =======

# TODO implement - The signs of the magnitudes alternate: if the difference between two leaves is +2, for example, then before there can be another difference of +2, there must be a difference of -2.
# TODO implement - Because `state.leavesTotal // 2` always has one more than `- state.leavesTotal // 2`, the first and last differences with magnitude `state.leavesTotal // 2` are positive.
# TODO implement The running total of the differences does not repeat in a Folding. Reminder: you can compute the total from both ends: we know the final total is `state.leavesTotal // 2`.
def reduceAllPermutationSpaceInEliminationState(state: EliminationState) -> EliminationState:
	"""Flow control to apply per-`PermutationSpace` functions to all of `state.listPermutationSpace`."""
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_byCrease(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_LeafIsPinned(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_HeadsBeforeTails(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_ConditionalPredecessors(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_CrossedCreases(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_noConsecutiveDimensions(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_leafDomainIs1(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_nakedSubset(state), listPermutationSpace))

	return state

#-------- Shared logic -----------------------------------------


def _reduceLeafOptionsOfPiles(state: EliminationState, permutationSpace: PermutationSpace, pilesToUpdate: deque[tuple[Pile, LeafOptions]], leafAntiOptions: LeafOptions) -> PermutationSpace:
	permutationSpaceHasNewLeaf: bool = False
	while permutationSpace and pilesToUpdate and not permutationSpaceHasNewLeaf:
		pile, leafOptions = pilesToUpdate.pop()

		leafSpace: LeafSpace | None = JeanValjean(leafOptionsAND(leafAntiOptions, leafOptions))
		if leafSpace is not None:

			permutationSpace[pile] = leafSpace
			if thisIsALeaf(permutationSpace[pile]):
				leafBeans: Leaf | None = None
				for beans, cornbread in ((一+零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))):
					beansPinned: bool = leafIsPinned(permutationSpace, beans)
					cornbreadPinned: bool = leafIsPinned(permutationSpace, cornbread)
					if beansPinned ^ cornbreadPinned:
						leafBeans = beans if beansPinned else cornbread
						break

				if leafBeans is not None:

					pileCornbread = pile
					if leafBeans in [一+零, 首一(state.dimensionsTotal)]:
						pileCornbread += 1
						leafCornbread: int = one(getLeavesCreasePost(state, leafBeans))
					else:
						pileCornbread -= 1
						leafCornbread = one(getLeavesCreaseAnte(state, leafBeans))

					if disqualifyPinningLeafAtPile(EliminationState(state.mapShape, pile=pileCornbread, permutationSpace=permutationSpace), leafCornbread):
						permutationSpace = {}
					else:
						permutationSpace = atPilePinLeaf(permutationSpace, pileCornbread, leafCornbread)
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
def _reducePermutationSpace_byCrease(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	permutationSpaceHasNewLeaf = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		dequePileLeafSpacePileLeafSpace: deque[tuple[tuple[Pile, LeafSpace], tuple[Pile, LeafSpace]]] = deque(pairwise(sorted(permutationSpace.items())))
		while dequePileLeafSpacePileLeafSpace and not permutationSpaceHasNewLeaf:
			(pile_k, leafSpace_k), (pile_r, leafSpace_r) = dequePileLeafSpacePileLeafSpace.pop()

			leafAntiOptions: LeafOptions = getLeafAntiOptions(state.leavesTotal, frozenset())
			leavesCrease: Iterator[Leaf] = iter(())
			pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque()

			if thisIsALeaf(leafSpace_k) and thisIsLeafOptions(leafSpace_r):
				pilesToUpdate = deque([(pile_r, leafSpace_r)])
				leavesCrease = getLeavesCreasePost(state, leafSpace_k)
			elif thisIsLeafOptions(leafSpace_k) and thisIsALeaf(leafSpace_r):
				pilesToUpdate = deque([(pile_k, leafSpace_k)])
				leavesCrease = getLeavesCreaseAnte(state, leafSpace_r)
			else:
				continue

			leafAntiOptions = getLeafAntiOptions(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease))

			sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
			if not (permutationSpace := _reduceLeafOptionsOfPiles(state, permutationSpace, pilesToUpdate, leafAntiOptions)):
				return None
			if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaves:
				permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_ConditionalPredecessors(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
		return permutationSpace

	dictionaryConditionalLeafPredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = getDictionaryConditionalLeafPredecessors(state)

	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		dequePileLeaf: deque[tuple[Pile, Leaf]] = deque(sorted(DOTitems(valfilter(mappingHasKey(dictionaryConditionalLeafPredecessors),
			keyfilter(notPileLast(state.pileLast), valfilter(notLeafOriginOrLeaf零, extractPinnedLeaves(permutationSpace)))))))

		while dequePileLeaf and not permutationSpaceHasNewLeaf:
			pile, leaf = dequePileLeaf.pop()

			if mappingHasKey(dictionaryConditionalLeafPredecessors[leaf], pile):
				# For this `pile:leaf` in `permutationSpace`, `dictionaryConditionalLeafPredecessors` has a `list` of at least one
				# `leaf` that must precede this `pile:leaf`, so the `list` cannot follow this `pile:leaf`, so remove the `list`
				# from the `LeafOptions` at piles after `pile`.
				leafAntiOptions: LeafOptions = getLeafAntiOptions(state.leavesTotal, dictionaryConditionalLeafPredecessors[leaf][pile])

				pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque(DOTitems(extractUndeterminedPiles(keyfilter(between吗(pile + inclusive, state.pileLast), permutationSpace))))

				sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafOptionsOfPiles(state, permutationSpace, pilesToUpdate, leafAntiOptions)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaves:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_CrossedCreases(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	permutationSpaceHasNewLeaf = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		for dimension in range(state.dimensionsTotal):

			dictionaryLeafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in extractPinnedLeaves(permutationSpace).items()}

			# For efficiency, I wish I could create the two dictionaries with one operation and without the intermediate `leavesPinned`.
			leavesPinned: PinnedLeaves = extractPinnedLeaves(permutationSpace)
			leavesPinnedEvenInDimension: PinnedLeaves = valfilter(complement(leafIsOddInDimension(dimension=dimension)), leavesPinned)
			leavesPinnedOddInDimension: PinnedLeaves = valfilter(leafIsOddInDimension(dimension=dimension), leavesPinned)

			dequePileLeafPileLeaf: deque[tuple[PinnedLeaves, tuple[tuple[Pile, Leaf], tuple[Pile, Leaf]]]] = deque(
												CartesianProduct((leavesPinnedOddInDimension,), combinations(leavesPinnedEvenInDimension.items(), 2)))
			dequePileLeafPileLeaf.extend(deque(CartesianProduct((leavesPinnedEvenInDimension,), combinations(leavesPinnedOddInDimension.items(), 2))))

			while dequePileLeafPileLeaf and not permutationSpaceHasNewLeaf:
				leavesPinnedParityOpposite, ((pileOf_k, leaf_k), (pileOf_r, leaf_r)) = dequePileLeafPileLeaf.pop()
				leaf_kCrease: Leaf = int(bit_flip(leaf_k, dimension))
				leaf_rCrease: Leaf = int(bit_flip(leaf_r, dimension))
				pileOf_kCrease: Pile = 31212012
				pileOf_rCrease: Pile = 31212012

				if (leaf_kCreaseIsPinned := leafIsPinned(leavesPinnedParityOpposite, leaf_kCrease)):
					pileOf_kCrease = dictionaryLeafToPile[leaf_kCrease]
				if (leaf_rCreaseIsPinned := leafIsPinned(leavesPinnedParityOpposite, leaf_rCrease)):
					pileOf_rCrease = dictionaryLeafToPile[leaf_rCrease]

				pilesForbidden: Iterable[Pile] = []
				leafAntiOptions: LeafOptions = getLeafAntiOptions(state.leavesTotal, frozenset())

				if leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
					leafAntiOptions = getLeafAntiOptions(state.leavesTotal, (leaf_rCrease,))

					if pileOf_k < pileOf_r < pileOf_kCrease:
						pilesForbidden = frozenset([*range(pileOf_k), *range(pileOf_kCrease + 1, state.pileLast + inclusive)])
					elif pileOf_kCrease < pileOf_r < pileOf_k:
						pilesForbidden = frozenset([*range(pileOf_kCrease), *range(pileOf_k + 1, state.pileLast + inclusive)])
					elif (pileOf_r < pileOf_kCrease < pileOf_k) or (pileOf_kCrease < pileOf_k < pileOf_r):
						pilesForbidden = frozenset(range(pileOf_kCrease + 1, pileOf_k))
					elif (pileOf_r < pileOf_k < pileOf_kCrease) or (pileOf_k < pileOf_kCrease < pileOf_r):
						pilesForbidden = frozenset(range(pileOf_k + 1, pileOf_kCrease))

				elif not leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
					leafAntiOptions = getLeafAntiOptions(state.leavesTotal, (leaf_kCrease,))

					if pileOf_rCrease < pileOf_k < pileOf_r:
						pilesForbidden = frozenset([*range(pileOf_rCrease), *range(pileOf_r + 1, state.pileLast + inclusive)])
					elif pileOf_r < pileOf_k < pileOf_rCrease:
						pilesForbidden = frozenset([*range(pileOf_r), *range(pileOf_rCrease + 1, state.pileLast + inclusive)])
					elif (pileOf_k < pileOf_r < pileOf_rCrease) or (pileOf_r < pileOf_rCrease < pileOf_k):
						pilesForbidden = frozenset(range(pileOf_r + 1, pileOf_rCrease))
					elif (pileOf_k < pileOf_rCrease < pileOf_r) or (pileOf_rCrease < pileOf_r < pileOf_k):
						pilesForbidden = frozenset(range(pileOf_rCrease + 1, pileOf_r))

				elif leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
					if thisIsAViolation(pileOf_k, pileOf_r, pileOf_kCrease, pileOf_rCrease):
						return None

				elif not leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
				# case leaves: leaf_k, leaf_r
				# I don't think I have enough information to do anything.
					pass

				pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque(DOTitems(keyfilter(thisHasThat(pilesForbidden), extractUndeterminedPiles(permutationSpace))))

				sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafOptionsOfPiles(state, permutationSpace, pilesToUpdate, leafAntiOptions)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaves:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_HeadsBeforeTails(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		dequePileLeaf: deque[tuple[Pile, Leaf]] = deque(sorted(DOTitems(keyfilter(notPileLast(state.pileLast), valfilter(notLeafOriginOrLeaf零, extractPinnedLeaves(permutationSpace))))))
		while dequePileLeaf and not permutationSpaceHasNewLeaf:
			pile, leaf = dequePileLeaf.pop()
			dimensionTail: int = dimensionNearestTail(leaf)
			dimensionHead: int = dimensionNearest首(leaf)
			if 0 < dimensionHead:
				leavesForbidden = range(state.productsOfDimensions[dimensionHead], state.leavesTotal, state.productsOfDimensions[dimensionHead])
				floor: Pile = 2
				ceiling: Pile = pile - inclusive

				leafAntiOptions: LeafOptions = getLeafAntiOptions(state.leavesTotal, leavesForbidden)
				pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque(extractUndeterminedPiles(keyfilter(between吗(floor, ceiling), permutationSpace)).items())

				sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafOptionsOfPiles(state, permutationSpace, pilesToUpdate, leafAntiOptions)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaves:
					permutationSpaceHasNewLeaf = True

			if 0 < dimensionTail:
				leavesForbidden = range(0, state.sumsOfProductsOfDimensions[dimensionTail], 1)
				floor: Pile = pile + inclusive
				ceiling: Pile = state.pileLast

				leafAntiOptions: LeafOptions = getLeafAntiOptions(state.leavesTotal, leavesForbidden)
				pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque(extractUndeterminedPiles(keyfilter(between吗(floor, ceiling), permutationSpace)).items())

				sumChecksForNewLeaves = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafOptionsOfPiles(state, permutationSpace, pilesToUpdate, leafAntiOptions)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaves:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_LeafIsPinned(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""Update or invalidate `permutationSpace`: for every `leaf` pinned at a `pile`, remove `leaf` from `LeafOptions` from every other `pile`; or return `None` if the updated `permutationSpace` is invalid.

	If the `LeafOptions` for a `pile` is reduced to one `leaf`, then convert from `pile: leafOptions` to `pile: leaf`.
	If that results in "beans without cornbread", then pin the complementary "cornbread" `leaf` at the appropriate adjacent
	`pile`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: leafOptions`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		An updated `permutationSpace` if valid; otherwise `None`.

	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		leavesPinned, pilesUndetermined = bifurcatePermutationSpace(permutationSpace)

		leafAntiOptions: LeafOptions = getLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned))

		pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque(pilesUndetermined.items())

		sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
		if not (permutationSpace := _reduceLeafOptionsOfPiles(state, permutationSpace, pilesToUpdate, leafAntiOptions)):
			return None
		if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaves:
			permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_nakedSubset(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""New.

	- extract UndeterminedPiles
	- sort by LeafOptions
	- more_itertools.split_when LeafOptionsX != LeafOptionsY
	- filter: len of list == list[0].bit_count - 1
	- if a list comes out, it's a naked subset
	- the leaves of the naked subset become the leafAntiOptions for all other LeafOptions: invoke _reduceLeafOptionsOfPiles

	This is not supposed to be a comprehensive analysis of exact coverage.
		- High-throughput for a strong ROI
		- VERY STRONGLY mirror the structure of other functions because it is a near certainty some existing functions and some yet-to-exist functions will merge in the future.

	"""
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		pilesUndetermined: UndeterminedPiles = extractUndeterminedPiles(permutationSpace)

		listPileAndLeafOptionsSorted: list[tuple[Pile, LeafOptions]] = sorted(
			pilesUndetermined.items()
			, key=lambda pileAndLeafOptions: pileAndLeafOptions[1]
		)

		iteratorPileAndLeafOptionsGrouped: Iterator[list[tuple[Pile, LeafOptions]]] = map(
			list
			, split_when(
				listPileAndLeafOptionsSorted
				, lambda pileAndLeafOptionsLeft, pileAndLeafOptionsRight: (
					pileAndLeafOptionsLeft[1] != pileAndLeafOptionsRight[1]
				)
			)
		)

		iteratorNakedSubsets: Iterable[list[tuple[Pile, LeafOptions]]] = filter(
			lambda listPileAndLeafOptionsGrouped: (
				len(listPileAndLeafOptionsGrouped)
				== (listPileAndLeafOptionsGrouped[0][1].bit_count() - 1)
			)
			, iteratorPileAndLeafOptionsGrouped
		)

		dequePileAndLeafOptionsGrouped: deque[list[tuple[Pile, LeafOptions]]] = deque(iteratorNakedSubsets)

		while dequePileAndLeafOptionsGrouped and not permutationSpaceHasNewLeaf:
			listPileAndLeafOptionsGrouped: list[tuple[Pile, LeafOptions]] = dequePileAndLeafOptionsGrouped.pop()
			leafOptionsNakedSubset: LeafOptions = listPileAndLeafOptionsGrouped[0][1]

			pilesNakedSubset: set[Pile] = {pileAndLeafOptions[0] for pileAndLeafOptions in listPileAndLeafOptionsGrouped}

			pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque(filter(
				lambda pileAndLeafOptions: pileAndLeafOptions[0] not in pilesNakedSubset
				, pilesUndetermined.items()
			))

			leafAntiOptions: LeafOptions = getLeafAntiOptions(
				state.leavesTotal
				, getIteratorOfLeaves(leafOptionsNakedSubset)
			)
			sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
			if not (permutationSpace := _reduceLeafOptionsOfPiles(state, permutationSpace, pilesToUpdate, leafAntiOptions)):
				return None
			if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaves:
				permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_noConsecutiveDimensions(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
# TODO Figure out a way to measure how often this function (or the other functions) actually reduces `permutationSpace`.
	permutationSpaceHasNewLeaf = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		piles3consecutive: deque[tuple[tuple[Pile, LeafSpace], tuple[Pile, LeafSpace], tuple[Pile, LeafSpace]]
						] = deque(triplewise(sorted(DOTitems(permutationSpace))))

		while piles3consecutive and not permutationSpaceHasNewLeaf:
			(pile_k, leafSpace_k), (pile, leafSpace), (pile_r, leafSpace_r) = piles3consecutive.pop()

			leafAntiOptions: LeafOptions = getLeafAntiOptions(state.leavesTotal, frozenset())
			leafForbidden: Leaf = 0
			pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque()

			if thisIsALeaf(leafSpace_k) and thisIsALeaf(leafSpace) and thisIsLeafOptions(leafSpace_r):
				pilesToUpdate = deque([(pile_r, leafSpace_r)])
				differenceOfLeaves: int = leafSpace_k - leafSpace
				leafForbidden = leafSpace + differenceOfLeaves
			elif thisIsALeaf(leafSpace_k) and thisIsLeafOptions(leafSpace) and thisIsALeaf(leafSpace_r):
				pilesToUpdate = deque([(pile, leafSpace)])
				leafForbidden = (leafSpace_k + leafSpace_r) // 2
			elif thisIsLeafOptions(leafSpace_k) and thisIsALeaf(leafSpace) and thisIsALeaf(leafSpace_r):
				pilesToUpdate = deque([(pile_k, leafSpace_k)])
				differenceOfLeaves: int = leafSpace - leafSpace_r
				leafForbidden = leafSpace - differenceOfLeaves
			else:
				continue

			leafAntiOptions = getLeafAntiOptions(state.leavesTotal, [leafForbidden])

			sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
			if not (permutationSpace := _reduceLeafOptionsOfPiles(state, permutationSpace, pilesToUpdate, leafAntiOptions)):
				return None
			if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaves:
				permutationSpaceHasNewLeaf = True

	return permutationSpace

#-------- Functions that do NOT use the shared logic -----------------------------------------

@syntacticCurry
def _reducePermutationSpace_leafDomainIs1(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		leavesPinned, pilesUndetermined = bifurcatePermutationSpace(permutationSpace)

		leafAndItsDomainSize: Counter[Leaf] = Counter(chain(
			chain.from_iterable(map(getIteratorOfLeaves, DOTvalues(pilesUndetermined))),
			DOTvalues(leavesPinned)
		))

		if set(range(state.leavesTotal)).difference(leafAndItsDomainSize.keys()):
			return None

		leavesWithDomainOf1: set[Leaf] = set(DOTkeys(valfilter((1).__eq__, leafAndItsDomainSize))).difference(leavesPinned.values()).difference([state.leavesTotal])
		if leavesWithDomainOf1:
			permutationSpaceHasNewLeaf = True
			leaf: Leaf = leavesWithDomainOf1.pop()
			pile: Pile = one(DOTkeys(valfilter(leafIsInLeafOptions(leaf), pilesUndetermined)))
			sherpa: PermutationSpace | None = _reducePermutationSpace_LeafIsPinned(state, atPilePinLeaf(permutationSpace, pile, leaf))
			if (sherpa is None) or (not sherpa):
				return None
			else:
				permutationSpace = sherpa
	return permutationSpace
