# ruff: noqa: ERA001
from collections import Counter, deque
from cytoolz.dicttoolz import itemfilter, keyfilter, valfilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from cytoolz.itertoolz import unique
from gmpy2 import bit_flip, bit_test as isBit1吗
from itertools import chain, combinations, product as CartesianProduct
from mapFolding import errorL33T, inclusive
from mapFolding._e import (
	bifurcatePermutationSpace, dimensionIndex, dimensionNearestTail, dimensionNearest首, DOTitems, DOTkeys, DOTvalues,
	getDictionaryConditionalLeafPredecessors, getIteratorOfLeaves, getLeafDomain, getLeavesCreaseAnte, getLeavesCreasePost,
	howManyLeavesInLeafOptions, JeanValjean, Leaf, LeafOptions, leafOptionsAND, leafOrigin, LeafSpace, makeLeafAntiOptions,
	mapShapeIs2上nDimensions, PermutationSpace, Pile, PinnedLeaves, UndeterminedPiles, 一, 零, 首一, 首零一)
from mapFolding._e.algorithms.iff import thisIsAViolation
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import (
	between吗, extractPinnedLeaves, extractUndeterminedPiles, leafIsInPileRange, leafIsPinned, mappingHasKey,
	notLeafOriginOrLeaf零, notPileLast, thisHasThat, thisIsALeaf, thisIsLeafOptions, thisNotHaveThat)
from mapFolding._e.pinIt import atPilePinLeaf, disqualifyPinningLeafAtPile
from more_itertools import filter_map, one, pairwise, triplewise
from typing import TYPE_CHECKING
from typing_extensions import TypeIs

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator

#======== Boolean filters ======================================

@syntacticCurry
def ImaOddLeaf2上nDimensional(leaf: Leaf, dimension: int) -> bool:
	"""A specialized version of parity checking for 2^n-dimensional maps."""
	return isBit1吗(leaf, dimension)

#======== Reducing `LeafOptions` ===============================

# TODO implement The running total of the differences does not repeat in a Folding. and - The running total is a distinct integer in the range `[0, state.leavesTotal)`.
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
	state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_leafDomainOf1(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_nakedSubset(state), listPermutationSpace))

	# listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	# state.listPermutationSpace = []
	# state.listPermutationSpace.extend(filter_map(Z0Z_reducePermutationSpace_dimensionRestrictions(state), listPermutationSpace))

	return state

#-------- Shared logic -----------------------------------------

def _reduceLeafSpace(state: EliminationState, permutationSpace: PermutationSpace, pilesToUpdate: deque[tuple[Pile, LeafOptions]], leafAntiOptions: LeafOptions) -> PermutationSpace:
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
						permutationSpace = atPilePinLeaf(permutationSpace, pileCornbread, leafCornbread) # pyright: ignore[reportUnknownVariableType]
				permutationSpaceHasNewLeaf = True
		else:
			permutationSpace = {}

	if permutationSpace and permutationSpaceHasNewLeaf:
		sherpa: PermutationSpace | None = _reducePermutationSpace_LeafIsPinned(state, permutationSpace) # pyright: ignore[reportUnknownArgumentType]
		if not sherpa:
			permutationSpace = {}
		else:
			permutationSpace = sherpa
	return permutationSpace # pyright: ignore[reportUnknownVariableType]

#-------- Functions that use the shared logic -----------------------------------------

@syntacticCurry
def _reducePermutationSpace_byCrease(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	leavesCrease: Iterator[Leaf] = iter(())
	pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque()

	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		dequePileLeafSpacePileLeafSpace: deque[tuple[tuple[Pile, LeafSpace], tuple[Pile, LeafSpace]]] = deque(pairwise(sorted(permutationSpace.items())))
		while dequePileLeafSpacePileLeafSpace and not permutationSpaceHasNewLeaf:
			(pile_k, leafSpace_k), (pile_r, leafSpace_r) = dequePileLeafSpacePileLeafSpace.pop()

			if thisIsALeaf(leafSpace_k) and thisIsLeafOptions(leafSpace_r):
				pilesToUpdate = deque([(pile_r, leafSpace_r)])
				leavesCrease = getLeavesCreasePost(state, leafSpace_k)
			elif thisIsLeafOptions(leafSpace_k) and thisIsALeaf(leafSpace_r):
				pilesToUpdate = deque([(pile_k, leafSpace_k)])
				leavesCrease = getLeavesCreaseAnte(state, leafSpace_r)
			else:
				continue

			sumChecksForNewLeaf: int = sum(map(dimensionNearest首, permutationSpace.values()))
			if not (permutationSpace := _reduceLeafSpace(state, permutationSpace, pilesToUpdate, makeLeafAntiOptions(state.leavesTotal, set(range(state.leavesTotal)).difference(leavesCrease)))):
				return None
			if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaf:
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

				sumChecksForNewLeaf: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafSpace(state, permutationSpace
						, pilesToUpdate = deque(DOTitems(extractUndeterminedPiles(keyfilter(between吗(pile + inclusive, state.pileLast), permutationSpace))))
						, leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, dictionaryConditionalLeafPredecessors[leaf][pile])
					)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaf:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_CrossedCreases(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	pileOf_kCrease: Pile = errorL33T
	pileOf_rCrease: Pile = errorL33T
	pilesForbidden: Iterable[Pile] = []
	leafAntiOptions: LeafOptions = makeLeafAntiOptions(state.leavesTotal, frozenset())

	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False
		for dimension in range(state.dimensionsTotal):

			dictionaryLeafToPile: dict[Leaf, Pile] = {leafValue: pileKey for pileKey, leafValue in extractPinnedLeaves(permutationSpace).items()}

			# For efficiency, I wish I could create the two dictionaries with one operation and without the intermediate `leavesPinned`.
			leavesPinned: PinnedLeaves = extractPinnedLeaves(permutationSpace)
			leavesPinnedEvenInDimension: PinnedLeaves = valfilter(complement(ImaOddLeaf2上nDimensional(dimension=dimension)), leavesPinned)
			leavesPinnedOddInDimension: PinnedLeaves = valfilter(ImaOddLeaf2上nDimensional(dimension=dimension), leavesPinned)

			dequePileLeafPileLeaf: deque[tuple[PinnedLeaves, tuple[tuple[Pile, Leaf], tuple[Pile, Leaf]]]] = deque(
												CartesianProduct((leavesPinnedOddInDimension,), combinations(leavesPinnedEvenInDimension.items(), 2)))
			dequePileLeafPileLeaf.extend(deque(CartesianProduct((leavesPinnedEvenInDimension,), combinations(leavesPinnedOddInDimension.items(), 2))))

			while dequePileLeafPileLeaf and not permutationSpaceHasNewLeaf:
				leavesPinnedParityOpposite, ((pileOf_k, leaf_k), (pileOf_r, leaf_r)) = dequePileLeafPileLeaf.pop()
				leaf_kCrease: Leaf = int(bit_flip(leaf_k, dimension))
				leaf_rCrease: Leaf = int(bit_flip(leaf_r, dimension))

				if (leaf_kCreaseIsPinned := leafIsPinned(leavesPinnedParityOpposite, leaf_kCrease)):
					pileOf_kCrease = dictionaryLeafToPile[leaf_kCrease]
				if (leaf_rCreaseIsPinned := leafIsPinned(leavesPinnedParityOpposite, leaf_rCrease)):
					pileOf_rCrease = dictionaryLeafToPile[leaf_rCrease]

				if leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
					leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, (leaf_rCrease,))

					if pileOf_k < pileOf_r < pileOf_kCrease:
						pilesForbidden = frozenset([*range(pileOf_k), *range(pileOf_kCrease + 1, state.pileLast + inclusive)])
					elif pileOf_kCrease < pileOf_r < pileOf_k:
						pilesForbidden = frozenset([*range(pileOf_kCrease), *range(pileOf_k + 1, state.pileLast + inclusive)])
					elif (pileOf_r < pileOf_kCrease < pileOf_k) or (pileOf_kCrease < pileOf_k < pileOf_r):
						pilesForbidden = frozenset(range(pileOf_kCrease + 1, pileOf_k))
					elif (pileOf_r < pileOf_k < pileOf_kCrease) or (pileOf_k < pileOf_kCrease < pileOf_r):
						pilesForbidden = frozenset(range(pileOf_k + 1, pileOf_kCrease))

				elif not leaf_kCreaseIsPinned and leaf_rCreaseIsPinned:
					leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, (leaf_kCrease,))

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
					continue

				else:
				# elif not leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
				# case leaves: leaf_k, leaf_r
				# I don't think I have enough information to do anything.
					continue

				sumChecksForNewLeaf: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafSpace(state, permutationSpace
					, pilesToUpdate = deque(DOTitems(keyfilter(thisHasThat(pilesForbidden), extractUndeterminedPiles(permutationSpace))))
					, leafAntiOptions=leafAntiOptions)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaf:
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

			dimensionHead: int = dimensionNearest首(leaf)
			if 0 < dimensionHead:
				sumChecksForNewLeaf: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafSpace(state, permutationSpace
						, pilesToUpdate = deque(extractUndeterminedPiles(keyfilter(between吗(2, pile - inclusive), permutationSpace)).items())
						, leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, range(state.productsOfDimensions[dimensionHead], state.leavesTotal, state.productsOfDimensions[dimensionHead]))
					)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaf:
					permutationSpaceHasNewLeaf = True

			dimensionTail: int = dimensionNearestTail(leaf)
			if 0 < dimensionTail:
				sumChecksForNewLeaf: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafSpace(state, permutationSpace
						, pilesToUpdate = deque(extractUndeterminedPiles(keyfilter(between吗(pile + inclusive, state.pileLast), permutationSpace)).items())
						, leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, range(leafOrigin, state.sumsOfProductsOfDimensions[dimensionTail]))
					)):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaf:
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
		sumChecksForNewLeaf: int = sum(map(dimensionNearest首, permutationSpace.values()))
		if not (permutationSpace := _reduceLeafSpace(state, permutationSpace, deque(pilesUndetermined.items()), makeLeafAntiOptions(state.leavesTotal, DOTvalues(leavesPinned)))):
			return None
		if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaf:
			permutationSpaceHasNewLeaf = True

	# Ensure beans-cornbread pairs are complete
	for leafBeans, leafCornbread in [(一 + 零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))]:
		if leafBeans in DOTvalues(permutationSpace) and leafCornbread not in DOTvalues(permutationSpace):
			pileBeans: Pile = next(pile for pile, leaf in DOTitems(permutationSpace) if leaf == leafBeans)
			domainCornbread = getLeafDomain(state, leafCornbread)

			for pileCornbread in (pileBeans - inclusive, pileBeans + inclusive):
				if pileCornbread in domainCornbread and pileCornbread in permutationSpace:
					if thisIsLeafOptions(permutationSpace[pileCornbread]):
						if permutationSpaceUpdated := atPilePinLeaf(permutationSpace, pileCornbread, leafCornbread):
							permutationSpace = permutationSpaceUpdated
							break
					elif permutationSpace[pileCornbread] == leafCornbread:
						break
			else:
				return None

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_nakedSubset(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	"""New.

	- extract UndeterminedPiles dict[Pile, LeafOptions]
	- group by LeafOptions dict[LeafOptions, list[Pile]]
	- filter LeafOptions.bit_count() - 1 == len(list)

	- if a list comes out, it's a naked subset
	- the leaves of the naked subset become the leafAntiOptions for all other LeafOptions: invoke _reducePileRangesOfLeaves

	This is not supposed to be a comprehensive analysis of exact coverage.
		- High-throughput for a strong ROI
		- VERY STRONGLY mirror the structure of other functions because it is a near certainty some existing functions and some yet-to-exist functions will merge in the future.

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

			sumChecksForNewLeaf: int = sum(map(dimensionNearest首, permutationSpace.values()))
			if not (permutationSpace := _reduceLeafSpace(state, permutationSpace
					, pilesToUpdate = deque(DOTitems(keyfilter(thisNotHaveThat(setPiles), pilesUndetermined)))
					, leafAntiOptions = makeLeafAntiOptions(state.leavesTotal, getIteratorOfLeaves(leafOptions))
				)):
				return None
			if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaf:
				permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_noConsecutiveDimensions(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	leafForbidden: Leaf = -errorL33T
	pilesToUpdate: deque[tuple[Pile, LeafOptions]] = deque()

	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		piles3consecutive: deque[tuple[tuple[Pile, LeafSpace], tuple[Pile, LeafSpace], tuple[Pile, LeafSpace]]] = deque(triplewise(sorted(DOTitems(permutationSpace))))

		while piles3consecutive and not permutationSpaceHasNewLeaf:
			(pile_k, leafSpace_k), (pile, leafSpace), (pile_r, leafSpace_r) = piles3consecutive.pop()

			if thisIsALeaf(leafSpace_k) and thisIsALeaf(leafSpace) and thisIsLeafOptions(leafSpace_r):
				pilesToUpdate = deque([(pile_r, leafSpace_r)])
				leafForbidden = leafSpace + (leafSpace - leafSpace_k)
			elif thisIsALeaf(leafSpace_k) and thisIsLeafOptions(leafSpace) and thisIsALeaf(leafSpace_r):
				pilesToUpdate = deque([(pile, leafSpace)])
				leafForbidden = (leafSpace_k + leafSpace_r) // 2
			elif thisIsLeafOptions(leafSpace_k) and thisIsALeaf(leafSpace) and thisIsALeaf(leafSpace_r):
				pilesToUpdate = deque([(pile_k, leafSpace_k)])
				leafForbidden = leafSpace - (leafSpace_r - leafSpace)
			else:
				continue

			if 0 <= leafForbidden < state.leavesTotal:
				sumChecksForNewLeaf: int = sum(map(dimensionNearest首, permutationSpace.values()))
				if not (permutationSpace := _reduceLeafSpace(state, permutationSpace, pilesToUpdate, makeLeafAntiOptions(state.leavesTotal, [leafForbidden]))):
					return None
				if sum(map(dimensionNearest首, permutationSpace.values())) < sumChecksForNewLeaf:
					permutationSpaceHasNewLeaf = True

	return permutationSpace

"""# TODO implement
1. The signs of the magnitudes alternate: if the difference between two leaves is +2, for example, then before there can be another difference of +2, there must be a difference of -2.
2. The total number of differences equal to `pos(state.leavesTotal // 2)` is always exactly one more than the total number of differences equal to `neg(state.leavesTotal // 2)`.
	1. Therefore, the first and last differences with magnitude `state.leavesTotal // 2` are positive.
3. For all other magnitudes in `state.productsOfDimensions[0:-2]`, the total number of positive and negative differences is always equal.
	1. Therefore, the first and last differences with those magnitudes must have opposite signs.
"""
@syntacticCurry
def Z0Z_reducePermutationSpace_dimensionRestrictions(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	thePile: int = 0
	theLeaf: int = 1
	forbiddenLeaves: set[Leaf] = set()
	Z0Z_limitSignFromOrigin: list[list[int]] = [[1]] + [[]] * len(state.productsOfDimensions[1:-2]) + [[-1]]
	Z0Z_limitSignFromEnd: list[list[int]] = [[1]] + [[]] * len(state.productsOfDimensions[1:-2]) + [[-1]]
	permutationSpaceHasNewLeaf: bool = True

	def pileLeaf吗(ww: tuple[Pile, LeafSpace]) -> TypeIs[tuple[Pile, Leaf]]:
		return thisIsALeaf(ww[1])

	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		leavesPinned = extractPinnedLeaves(permutationSpace)
		Z0Z_deque: deque[tuple[Pile, LeafSpace]] = deque(sorted(DOTitems(keyfilter(notPileLast(state.pileLast), valfilter(notLeafOriginOrLeaf零, permutationSpace)))))

		while (pileLeaf吗(Z0Z_deque[0])) or (pileLeaf吗(Z0Z_deque[-1])):
			thereIsNoLeftSpoon = Z0Z_deque.popleft()
			if pileLeaf吗(thereIsNoLeftSpoon):
				difference: int = thereIsNoLeftSpoon[theLeaf] - leavesPinned[thereIsNoLeftSpoon[thePile] - 1]
				if 0 < difference:
					sign = 1
				else:
					sign = -1
				indexDimension: int = dimensionIndex(abs(difference))
				Z0Z_limitSignFromOrigin[indexDimension].append(sign)
				if not Z0Z_limitSignFromEnd[indexDimension]:
					Z0Z_limitSignFromEnd[indexDimension].append(sign)
				# forbiddenLeaves
				leafAntiOptions: LeafOptions = makeLeafAntiOptions(state.leavesTotal, forbiddenLeaves)
			else:
				Z0Z_deque.appendleft(thereIsNoLeftSpoon)
				thereIsNoRightSpoon = Z0Z_deque.pop()
				if pileLeaf吗(thereIsNoRightSpoon):
					difference: int = leavesPinned[thereIsNoRightSpoon[thePile] + 1] - thereIsNoRightSpoon[theLeaf]
					if 0 < difference:
						sign = 1
					else:
						sign = -1
					indexDimension: int = dimensionIndex(abs(difference))
					Z0Z_limitSignFromEnd[indexDimension].append(sign)
					if not Z0Z_limitSignFromOrigin[indexDimension]:
						Z0Z_limitSignFromOrigin[indexDimension].append(sign)
					# forbiddenLeaves
					leafAntiOptions: LeafOptions = makeLeafAntiOptions(state.leavesTotal, forbiddenLeaves)  # noqa: F841
				else:
					message: str = 'I think this is a logic error.'
					raise ValueError(message)

		# while Z0Z_deque and not permutationSpaceHasNewLeaf:
		# 	something = Z0Z_deque.pop()

	return permutationSpace

#-------- Functions that do NOT use the shared logic -----------------------------------------

@syntacticCurry
def _reducePermutationSpace_leafDomainOf1(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		leavesPinned, pilesUndetermined = bifurcatePermutationSpace(permutationSpace)

		leafAndItsDomainSize: Counter[Leaf] = Counter(chain(chain.from_iterable(map(getIteratorOfLeaves, DOTvalues(pilesUndetermined))), DOTvalues(leavesPinned)))

		if set(range(state.leavesTotal)).difference(leafAndItsDomainSize.keys()):
			return None

		leavesWithDomainOf1: set[Leaf] = set(DOTkeys(valfilter((1).__eq__, leafAndItsDomainSize))).difference(leavesPinned.values()).difference([state.leavesTotal])
		if leavesWithDomainOf1:
			leaf: Leaf = leavesWithDomainOf1.pop()
			sherpa: PermutationSpace | None = _reducePermutationSpace_LeafIsPinned(state, atPilePinLeaf(permutationSpace, one(DOTkeys(valfilter(leafIsInPileRange(leaf), pilesUndetermined))), leaf))
			if (sherpa is None) or (not sherpa):
				return None
			else:
				permutationSpace = sherpa
			permutationSpaceHasNewLeaf = True
	return permutationSpace
