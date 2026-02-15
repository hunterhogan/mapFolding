# ruff: noqa: ERA001
from collections import Counter, deque
from cytoolz.dicttoolz import dissoc, get_in, keyfilter, valfilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from functools import cache
from gmpy2 import bit_flip, bit_test as isBit1吗
from itertools import chain, combinations, product as CartesianProduct
from mapFolding import inclusive
from mapFolding._e import (
	bifurcatePermutationSpace, DimensionIndex, dimensionNearestTail, dimensionNearest首, DOTitems, DOTkeys, DOTvalues,
	getAntiPileRangeOfLeaves, getDictionaryConditionalLeafPredecessors, getIteratorOfLeaves, getLeafDomain,
	getLeavesCreaseAnte, getLeavesCreasePost, JeanValjean, Leaf, LeafOrPileRangeOfLeaves, mapShapeIs2上nDimensions,
	PermutationSpace, Pile, PileRangeOfLeaves, pileRangeOfLeavesAND, PilesWithPileRangeOfLeaves, PinnedLeaves, 一, 零, 首一,
	首零一)
from mapFolding._e.algorithms.iff import thisIsAViolation
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import (
	between吗, extractPilesWithPileRangeOfLeaves, extractPinnedLeaves, leafIsInPileRange, leafIsPinned, mappingHasKey,
	notLeafOriginOrLeaf零, notPileLast, thisHasThat, thisIsALeaf, thisIsAPileRangeOfLeaves)
from mapFolding._e.pinIt import atPilePinLeaf, disqualifyPinningLeafAtPile
from more_itertools import (
	filter_map, ilen as lenIterator, one, pairwise, partition as more_itertools_partition, split_at, split_when,
	triplewise)
from operator import getitem
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator

#======== Boolean filters ======================================

# SEMIOTICS `leafIsOddInDimension` clarify and converge with the generalized version.
@syntacticCurry
def leafIsOddInDimension(leaf: Leaf, dimension: int) -> bool:
	"""A specialized version of parity checking for 2^n-dimensional maps."""
	return isBit1吗(leaf, dimension)

#======== Reducing `PileRangeOfLeaves` =======

# TODO implement - The signs of the magnitudes alternate: if the difference between two leaves is 2, for example, then before there can be another difference of 2, there must be a difference of -2.
# TODO implement - Because `state.leavesTotal // 2` always has one more than `- state.leavesTotal // 2`, the first and last differences with magnitude `state.leavesTotal // 2` are positive.
# TODO implement The running total of the differences does not repeat in a Folding.
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

def _reducePileRangesOfLeaves(state: EliminationState, permutationSpace: PermutationSpace, pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]], antiPileRangeOfLeaves: PileRangeOfLeaves) -> PermutationSpace:
	permutationSpaceHasNewLeaf: bool = False
	while permutationSpace and pilesToUpdate and not permutationSpaceHasNewLeaf:
		pile, pileRangeOfLeaves = pilesToUpdate.pop()

		leafOrPileRangeOfLeaves: LeafOrPileRangeOfLeaves | None = JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, pileRangeOfLeaves))
		if leafOrPileRangeOfLeaves is not None:

			permutationSpace[pile] = leafOrPileRangeOfLeaves
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

		dequePileLeafOrPileRangeOfLeavesPileLeafOrPileRangeOfLeaves: deque[tuple[tuple[Pile, LeafOrPileRangeOfLeaves], tuple[Pile, LeafOrPileRangeOfLeaves]]] = deque(pairwise(sorted(permutationSpace.items())))
		while dequePileLeafOrPileRangeOfLeavesPileLeafOrPileRangeOfLeaves and not permutationSpaceHasNewLeaf:
			(pile_k, leafOrPileRangeOfLeaves_k), (pile_r, leafOrPileRangeOfLeaves_r) = dequePileLeafOrPileRangeOfLeavesPileLeafOrPileRangeOfLeaves.pop()

			antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, frozenset())
			leavesCrease: Iterator[Leaf] = iter(())
			pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque()

			if thisIsALeaf(leafOrPileRangeOfLeaves_k) and thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves_r):
				pilesToUpdate = deque([(pile_r, leafOrPileRangeOfLeaves_r)])
				leavesCrease = getLeavesCreasePost(state, leafOrPileRangeOfLeaves_k)
			elif thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves_k) and thisIsALeaf(leafOrPileRangeOfLeaves_r):
				pilesToUpdate = deque([(pile_k, leafOrPileRangeOfLeaves_k)])
				leavesCrease = getLeavesCreaseAnte(state, leafOrPileRangeOfLeaves_r)
			else:
				continue

			antiPileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease))

			sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
			if not (permutationSpace := _reducePileRangesOfLeaves(state, permutationSpace, pilesToUpdate, antiPileRangeOfLeaves)):
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
				# from the `PileRangeOfLeaves` at piles after `pile`.
				antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, dictionaryConditionalLeafPredecessors[leaf][pile])

				pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(DOTitems(extractPilesWithPileRangeOfLeaves(keyfilter(between吗(pile + inclusive, state.pileLast), permutationSpace))))

				sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reducePileRangesOfLeaves(state, permutationSpace, pilesToUpdate, antiPileRangeOfLeaves)):
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
				antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, frozenset())

				if leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
					antiPileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, (leaf_rCrease,))

					if pileOf_k < pileOf_r < pileOf_kCrease:
						pilesForbidden = frozenset([*range(pileOf_k), *range(pileOf_kCrease + 1, state.pileLast + inclusive)])
					elif pileOf_kCrease < pileOf_r < pileOf_k:
						pilesForbidden = frozenset([*range(pileOf_kCrease), *range(pileOf_k + 1, state.pileLast + inclusive)])
					elif (pileOf_r < pileOf_kCrease < pileOf_k) or (pileOf_kCrease < pileOf_k < pileOf_r):
						pilesForbidden = frozenset(range(pileOf_kCrease + 1, pileOf_k))
					elif (pileOf_r < pileOf_k < pileOf_kCrease) or (pileOf_k < pileOf_kCrease < pileOf_r):
						pilesForbidden = frozenset(range(pileOf_k + 1, pileOf_kCrease))

				elif not leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
					antiPileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, (leaf_kCrease,))

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

				pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(DOTitems(keyfilter(thisHasThat(pilesForbidden), extractPilesWithPileRangeOfLeaves(permutationSpace))))

				sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reducePileRangesOfLeaves(state, permutationSpace, pilesToUpdate, antiPileRangeOfLeaves)):
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

				antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, leavesForbidden)
				pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(extractPilesWithPileRangeOfLeaves(keyfilter(between吗(floor, ceiling), permutationSpace)).items())

				sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reducePileRangesOfLeaves(state, permutationSpace, pilesToUpdate, antiPileRangeOfLeaves)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaves:
					permutationSpaceHasNewLeaf = True

			if 0 < dimensionTail:
				leavesForbidden = range(0, state.sumsOfProductsOfDimensions[dimensionTail], 1)
				floor: Pile = pile + inclusive
				ceiling: Pile = state.pileLast

				antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, leavesForbidden)
				pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(extractPilesWithPileRangeOfLeaves(keyfilter(between吗(floor, ceiling), permutationSpace)).items())

				sumChecksForNewLeaves = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reducePileRangesOfLeaves(state, permutationSpace, pilesToUpdate, antiPileRangeOfLeaves)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaves:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_LeafIsPinned(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""Update or invalidate `permutationSpace`: for every `leaf` pinned at a `pile`, remove `leaf` from `PileRangeOfLeaves` from every other `pile`; or return `None` if the updated `permutationSpace` is invalid.

	If the `PileRangeOfLeaves` for a `pile` is reduced to one `leaf`, then convert from `pile: pileRangeOfLeaves` to `pile: leaf`.
	If that results in "beans without cornbread", then pin the complementary "cornbread" `leaf` at the appropriate adjacent
	`pile`.

	Parameters
	----------
	state : EliminationState
		A data basket to facilitate computations and actions.
	permutationSpace : PermutationSpace
		A dictionary of `pile: leaf` and/or `pile: pileRangeOfLeaves`.

	Returns
	-------
	updatedPermutationSpace : PermutationSpace | None
		An updated `permutationSpace` if valid; otherwise `None`.

	"""
	permutationSpaceHasNewLeaf: bool = True

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		leavesPinned, pilesWithPileRangeOfLeaves = bifurcatePermutationSpace(permutationSpace)

		antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, DOTvalues(leavesPinned))

		pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(pilesWithPileRangeOfLeaves.items())

		sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
		if not (permutationSpace := _reducePileRangesOfLeaves(state, permutationSpace, pilesToUpdate, antiPileRangeOfLeaves)):
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

		piles3consecutive: deque[tuple[
			tuple[Pile, LeafOrPileRangeOfLeaves], tuple[Pile, LeafOrPileRangeOfLeaves], tuple[Pile, LeafOrPileRangeOfLeaves]
			]] = deque(triplewise(sorted(DOTitems(permutationSpace))))

		while piles3consecutive and not permutationSpaceHasNewLeaf:
			(pile_k, leafOrPileRangeOfLeaves_k), (pile, leafOrPileRangeOfLeaves), (pile_r, leafOrPileRangeOfLeaves_r) = piles3consecutive.pop()

			antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, frozenset())
			leafForbidden: Leaf = 0
			pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque()

			if thisIsALeaf(leafOrPileRangeOfLeaves_k) and thisIsALeaf(leafOrPileRangeOfLeaves) and thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves_r):
				pilesToUpdate = deque([(pile_r, leafOrPileRangeOfLeaves_r)])
				differenceOfLeaves: int = leafOrPileRangeOfLeaves_k - leafOrPileRangeOfLeaves
				leafForbidden = leafOrPileRangeOfLeaves + differenceOfLeaves
			elif thisIsALeaf(leafOrPileRangeOfLeaves_k) and thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves) and thisIsALeaf(leafOrPileRangeOfLeaves_r):
				pilesToUpdate = deque([(pile, leafOrPileRangeOfLeaves)])
				leafForbidden = (leafOrPileRangeOfLeaves_k + leafOrPileRangeOfLeaves_r) // 2
			elif thisIsAPileRangeOfLeaves(leafOrPileRangeOfLeaves_k) and thisIsALeaf(leafOrPileRangeOfLeaves) and thisIsALeaf(leafOrPileRangeOfLeaves_r):
				pilesToUpdate = deque([(pile_k, leafOrPileRangeOfLeaves_k)])
				differenceOfLeaves: int = leafOrPileRangeOfLeaves - leafOrPileRangeOfLeaves_r
				leafForbidden = leafOrPileRangeOfLeaves - differenceOfLeaves
			else:
				continue

			antiPileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, [leafForbidden])

			sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
			if not (permutationSpace := _reducePileRangesOfLeaves(state, permutationSpace, pilesToUpdate, antiPileRangeOfLeaves)):
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

		leavesPinned, pilesWithPileRangeOfLeaves = bifurcatePermutationSpace(permutationSpace)

		leafAndItsDomainSize: Counter[Leaf] = Counter(chain(
			chain.from_iterable(map(getIteratorOfLeaves, DOTvalues(pilesWithPileRangeOfLeaves))),
			DOTvalues(leavesPinned)
		))

		if set(range(state.leavesTotal)).difference(leafAndItsDomainSize.keys()):
			return None

		leavesWithDomainOf1: set[Leaf] = set(DOTkeys(valfilter((1).__eq__, leafAndItsDomainSize))).difference(leavesPinned.values()).difference([state.leavesTotal])
		if leavesWithDomainOf1:
			permutationSpaceHasNewLeaf = True
			leaf: Leaf = leavesWithDomainOf1.pop()
			pile: Pile = one(DOTkeys(valfilter(leafIsInPileRange(leaf), pilesWithPileRangeOfLeaves)))
			sherpa: PermutationSpace | None = _reducePermutationSpace_LeafIsPinned(state, atPilePinLeaf(permutationSpace, pile, leaf))
			if (sherpa is None) or (not sherpa):
				return None
			else:
				permutationSpace = sherpa
	return permutationSpace

#-------- Not implemented / decommissioned ------------------------

@syntacticCurry
def _reducePermutationSpace_nakedSubset(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""New.

	- extract PilesWithPileRangeOfLeaves
	- sort by PileRangeOfLeaves
	- more_itertools.split_when PileRangeOfLeavesX != PileRangeOfLeavesY
	- filter: len of list == list[0].bit_count - 1
	- if a list comes out, it's a naked subset
	- the leaves of the naked subset become the antiPileRangeOfLeaves for all other PileRangeOfLeaves: invoke _reducePileRangesOfLeaves

	This is not supposed to be a comprehensive analysis of exact coverage.
		- High-throughput for a strong ROI
		- VERY STRONGLY mirror the structure of other functions because it is a near certainty some existing functions and some yet-to-exist functions will merge in the future.

	"""
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		pilesWithPileRangeOfLeaves: PilesWithPileRangeOfLeaves = extractPilesWithPileRangeOfLeaves(permutationSpace)

		listPileAndPileRangeOfLeavesSorted: list[tuple[Pile, PileRangeOfLeaves]] = sorted(
			pilesWithPileRangeOfLeaves.items()
			, key=lambda pileAndPileRangeOfLeaves: pileAndPileRangeOfLeaves[1]
		)

		iteratorPileAndPileRangeOfLeavesGrouped: Iterator[list[tuple[Pile, PileRangeOfLeaves]]] = map(
			list
			, split_when(
				listPileAndPileRangeOfLeavesSorted
				, lambda pileAndPileRangeOfLeavesLeft, pileAndPileRangeOfLeavesRight: (
					pileAndPileRangeOfLeavesLeft[1] != pileAndPileRangeOfLeavesRight[1]
				)
			)
		)

		iteratorNakedSubsets: Iterable[list[tuple[Pile, PileRangeOfLeaves]]] = filter(
			lambda listPileAndPileRangeOfLeavesGrouped: (
				len(listPileAndPileRangeOfLeavesGrouped)
				== (listPileAndPileRangeOfLeavesGrouped[0][1].bit_count() - 1)
			)
			, iteratorPileAndPileRangeOfLeavesGrouped
		)

		dequePileAndPileRangeOfLeavesGrouped: deque[list[tuple[Pile, PileRangeOfLeaves]]] = deque(iteratorNakedSubsets)

		while dequePileAndPileRangeOfLeavesGrouped and not permutationSpaceHasNewLeaf:
			listPileAndPileRangeOfLeavesGrouped: list[tuple[Pile, PileRangeOfLeaves]] = dequePileAndPileRangeOfLeavesGrouped.pop()
			pileRangeOfLeavesNakedSubset: PileRangeOfLeaves = listPileAndPileRangeOfLeavesGrouped[0][1]

			pilesNakedSubset: set[Pile] = {pileAndPileRangeOfLeaves[0] for pileAndPileRangeOfLeaves in listPileAndPileRangeOfLeavesGrouped}

			pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(filter(
				lambda pileAndPileRangeOfLeaves: pileAndPileRangeOfLeaves[0] not in pilesNakedSubset
				, pilesWithPileRangeOfLeaves.items()
			))

			antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(
				state.leavesTotal
				, getIteratorOfLeaves(pileRangeOfLeavesNakedSubset)
			)
			sumChecksForNewLeaves: int = sum(map(dimensionNearest首, permutationSpace.values()))
			if not (permutationSpace := _reducePileRangesOfLeaves(state, permutationSpace, pilesToUpdate, antiPileRangeOfLeaves)):
				return None
			if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaves:
				permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
# TODO Implement `sudoku` reduction: borrow ideas from `notEnoughOpenPiles`.
def sudoku(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:  # noqa: ARG001
	"""My implementation broke `eliminationCrease` and possibly other things.

	Sudoku trick:
	in a restricted space (square, row, or column), if two numbers have the same domain of two cells, then all other numbers are excluded from those two cells.
	^^^ generalizes to if n numbers have the same domain of n cells, all other numbers are excluded from that domain of n cells.
	"""
	return permutationSpace
def notEnoughOpenPiles(state: EliminationState) -> bool:
	"""Decommissioned: implement with the sudoku trick.

	Check `state.permutationSpace` for enough open piles for required leaves.

	Some leaves must be before or after other leaves, such as the dimension origin leaves. For each pinned leaf, get all of the
	required leaves for before and after, and check if there are enough open piles for all of them. If the set of open piles does
	not intersect with the domain of a required leaf, return True. If a required leaf can only be pinned in one pile of the open
	piles, pin it at that pile in stateOfOpenPiles. Use the real pinning functions with the disposable stateOfOpenPiles. With the required
	leaves that are not pinned, check if there are enough open piles for them.
	"""
	stateWorkbench = EliminationState(state.mapShape, pile=state.pile, permutationSpace=state.permutationSpace.copy())

	dictionaryConditionalLeafPredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = getDictionaryConditionalLeafPredecessors(state)

# DEVELOPMENT Reminder: I designed this function before `updateListPermutationSpace` AND BEFORE `PileRangeOfLeaves`. I'm not using
# `PileRangeOfLeaves` anywhere in this function, which seems odd. I should rethink the entire function. Actually, I should
# translate the broader concept--allocating limited piles to the leaves that must have them--into a subroutine of
# `updateListPermutationSpace`. And then eliminate this function.

# This general concept is the sudoku trick, right?

# DEVELOPMENT Flow control
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

# DEVELOPMENT Too many intermediate variables. And/or the wrong variables. And/or the wrong functions.
		leavesPinned, _pilesWithPileRangeOfLeaves = bifurcatePermutationSpace(stateWorkbench.permutationSpace)
		leavesFixed: tuple[Leaf, ...] = tuple(DOTvalues(leavesPinned))
		leavesNotPinned: frozenset[Leaf] = frozenset(range(stateWorkbench.leavesTotal)).difference(leavesFixed)
		pilesOpen: frozenset[Pile] = frozenset(range(stateWorkbench.pileLast + inclusive)).difference(leavesPinned.keys())

		dequePileLeaf: deque[tuple[Pile, Leaf]] =  deque(sorted(DOTitems(keyfilter(notPileLast(state.pileLast), valfilter(notLeafOriginOrLeaf零, leavesPinned)))))

		while dequePileLeaf and not permutationSpaceHasNewLeaf:
# DEVELOPMENT Iteration data
			pile, leaf = dequePileLeaf.pop()
			leavesFixedBeforePile, leavesFixedAfterPile = split_at(leavesFixed, leaf.__eq__, maxsplit=1)
			pilesOpenAfterLeaf, pilesOpenBeforeLeaf = more_itertools_partition(pile.__lt__, pilesOpen)

			@cache
			def leaf_kMustPrecede_leaf吗(leaf_k: Leaf, leaf: Leaf = leaf, pile: Pile = pile) -> bool:
				if dimensionNearest首(leaf_k) <= dimensionNearestTail(leaf):
					return True
				return leaf_k in get_in([leaf, pile], dictionaryConditionalLeafPredecessors, default=list[Leaf]())

			dimensionHead: DimensionIndex = dimensionNearest首(leaf)
			@cache
			def leafMustPrecede_leaf_r吗(leaf_r: Leaf, dimensionHead: DimensionIndex = dimensionHead) -> bool:
				return dimensionHead <= dimensionNearestTail(leaf_r)

			if any(map(leaf_kMustPrecede_leaf吗, leavesFixedAfterPile)) or any(map(leafMustPrecede_leaf_r吗, leavesFixedBeforePile)):
				return True

			leavesMustPrecede_leaf: deque[Leaf] = deque(filter(leaf_kMustPrecede_leaf吗, leavesNotPinned))
			leafMustPrecedeLeaves: deque[Leaf] = deque(filter(leafMustPrecede_leaf_r吗, leavesNotPinned))

			if (lenIterator(pilesOpenBeforeLeaf) < lenIterator(leavesMustPrecede_leaf)) or (lenIterator(pilesOpenAfterLeaf) < lenIterator(leafMustPrecedeLeaves)):
				return True

			while leavesMustPrecede_leaf and not permutationSpaceHasNewLeaf:
# DEVELOPMENT Iteration data, nested
				leaf_k: Leaf = leavesMustPrecede_leaf.pop()

				domain_k = getLeafDomain(stateWorkbench, leaf_k)
				pilesOpenFor_k: set[Pile] = set(pilesOpenBeforeLeaf).intersection(domain_k)

				if len(pilesOpenFor_k) == 0:
					return True
				if len(pilesOpenFor_k) == 1:
					stateWorkbench.permutationSpace = atPilePinLeaf(stateWorkbench.permutationSpace, pilesOpenFor_k.pop(), leaf_k)
					permutationSpaceHasNewLeaf = True

			while leafMustPrecedeLeaves and not permutationSpaceHasNewLeaf:
# DEVELOPMENT Iteration data, nested
				leaf_r: Leaf = leafMustPrecedeLeaves.pop()

				domain_r = getLeafDomain(stateWorkbench, leaf_r)
				pilesOpenFor_r: set[int] = set(pilesOpenAfterLeaf).intersection(domain_r)

				if len(pilesOpenFor_r) == 0:
					return True
				if len(pilesOpenFor_r) == 1:
					stateWorkbench.permutationSpace = atPilePinLeaf(stateWorkbench.permutationSpace, pilesOpenFor_r.pop(), leaf_r)
					permutationSpaceHasNewLeaf = True

	return False
