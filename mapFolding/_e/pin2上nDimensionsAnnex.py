# ruff: noqa: ERA001
from collections import Counter, deque
from collections.abc import Iterable
from cytoolz.dicttoolz import keyfilter, valfilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from functools import cache
from gmpy2 import bit_flip, bit_test as isBit1吗, xmpz
from hunterMakesPy import raiseIfNone
from itertools import chain, combinations, filterfalse, product as CartesianProduct
from mapFolding import inclusive
from mapFolding._e import (
	between, dimensionNearestTail, dimensionNearest首, DOTgetPileIfLeaf, DOTitems, DOTkeys, DOTvalues,
	extractPilesWithPileRangeOfLeaves, extractPinnedLeaves, getAntiPileRangeOfLeaves,
	getDictionaryConditionalLeafPredecessors, getLeafDomain, getLeavesCreaseBack, getLeavesCreaseNext, JeanValjean, Leaf,
	leafIsInPileRange, leafIsPinned, mappingHasKey, mapShapeIs2上nDimensions, notLeafOriginOrLeaf零, notPileLast,
	PermutationSpace, Pile, pileIsNotOpen, pileIsOpen, PileRangeOfLeaves, pileRangeOfLeavesAND, PilesWithPileRangeOfLeaves,
	PinnedLeaves, reverseLookup, thisHasThat, thisIsALeaf, 一, 零, 首一, 首零一)
from mapFolding._e.algorithms.iff import removePermutationSpaceViolations, thisIsAViolation
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pinIt import atPilePinLeaf, deconstructPermutationSpaceAtPile
from more_itertools import (
	filter_map, ilen as lenIterator, one, only, partition as more_itertools_partition, split_at, windowed_complete)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

#======== Boolean filters ======================================

@syntacticCurry
def leafParityInDimension(leaf: Leaf, dimension: int) -> bool:
	return isBit1吗(leaf, dimension)

#======== append `permutationSpace` at `pile` if qualified =======

def appendPermutationSpaceAtPile(state: EliminationState, leavesToPin: Iterable[Leaf]) -> EliminationState:
	sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, permutationSpace=state.permutationSpace)
	disqualify: Callable[[int], bool] = disqualifyAppendingLeafAtPile(state)
	beansOrCornbread: Callable[[PermutationSpace], bool] = beansWithoutCornbread(sherpa)

	leafToPermutationSpace: dict[Leaf, PermutationSpace] = deconstructPermutationSpaceAtPile(state.permutationSpace, state.pile, filterfalse(disqualify, leavesToPin))

	sherpa.listPermutationSpace.extend(DOTvalues(valfilter(complement(beansOrCornbread), leafToPermutationSpace)))

	for permutationSpace in DOTvalues(valfilter(beansOrCornbread, leafToPermutationSpace)):
		stateCornbread: EliminationState = pinLeafCornbread(EliminationState(state.mapShape, pile=state.pile, permutationSpace=permutationSpace))
		if stateCornbread.permutationSpace:
			sherpa.listPermutationSpace.append(stateCornbread.permutationSpace)

	sherpa = updateListPermutationSpace(sherpa)

	sherpa = removeInvalidPermutationSpace(sherpa)
	state.listPermutationSpace.extend(sherpa.listPermutationSpace)

	return state

@syntacticCurry
def disqualifyAppendingLeafAtPile(state: EliminationState, leaf: Leaf) -> bool:
	return any([_pileNotInRangeByLeaf(state, leaf), leafIsPinned(state.permutationSpace, leaf), pileIsNotOpen(state.permutationSpace, state.pile)])

def _pileNotInRangeByLeaf(state: EliminationState, leaf: Leaf) -> bool:
	return state.pile not in getLeafDomain(state, leaf)

#======== Updating `PileRangeOfLeaves` =======

def updateListPermutationSpace(state: EliminationState) -> EliminationState:
	"""Flow control to apply per-`PermutationSpace` functions to all of `state.listPermutationSpace`."""
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
	state.listPermutationSpace.extend(filter_map(_reducePermutationSpace_leafDomainIs1(state), listPermutationSpace))

# TODO implement - The signs of the magnitudes alternate: if the difference between two leaves is 2, for example, then before there can be another difference of 2, there must be a difference of -2.

	return state

# TODO These statements are syntactically necessary because I'm using subscripts AND walrus operators. Does that suggest there is
# a "better" flow paradigm, or is this merely a limitation of Python syntax?
def _reducePileRangesOfLeaves(state: EliminationState, permutationSpace: PermutationSpace, pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]], antiPileRangeOfLeaves: PileRangeOfLeaves) -> PermutationSpace:
	permutationSpaceHasNewLeaf: bool = False
	while pilesToUpdate and not permutationSpaceHasNewLeaf:
		pileToUpdate, pileRangeOfLeaves = pilesToUpdate.pop()
		if (ImaLeafOrPileRangeOfLeavesNotAWalrusSubscript := JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, pileRangeOfLeaves))) is None:
			return {}
		permutationSpace[pileToUpdate] = ImaLeafOrPileRangeOfLeavesNotAWalrusSubscript
		if thisIsALeaf(permutationSpace[pileToUpdate]):
			if beansWithoutCornbread(state, permutationSpace) and not (permutationSpace := pinLeafCornbread(EliminationState(state.mapShape, pile=pileToUpdate, permutationSpace=permutationSpace)).permutationSpace):
				return {}
			permutationSpaceHasNewLeaf = True
	if permutationSpaceHasNewLeaf:
		sherpa: PermutationSpace | None = _reducePermutationSpace_LeafIsPinned(state, permutationSpace)
		if not sherpa:
			return {}
		else:
			permutationSpace = sherpa
	return permutationSpace

@syntacticCurry
def _reducePermutationSpace_leafDomainIs1(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:
	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		pinnedLeaves: PinnedLeaves = extractPinnedLeaves(permutationSpace)
		pilesWithPileRangeOfLeaves: PilesWithPileRangeOfLeaves = extractPilesWithPileRangeOfLeaves(permutationSpace)

		# TODO turn this garbage into a functional statement.
		leafAndItsDomainSize = Counter(chain.from_iterable(
			[xmpz(pileRangesOfLeaves).iter_set() for pileRangesOfLeaves in pilesWithPileRangeOfLeaves.values()]
			# + [pinnedLeaves.values()]
			+ [[leaf] for leaf in pinnedLeaves.values()]
		))

		if set(range(state.leavesTotal)).difference(leafAndItsDomainSize.keys()):
			return None

		leavesWithDomainOf1: set[Leaf] = set(DOTkeys(valfilter((1).__eq__, leafAndItsDomainSize))).difference(pinnedLeaves.values()).difference([state.leavesTotal])
		if leavesWithDomainOf1:
			permutationSpaceHasNewLeaf = True
			leaf: Leaf = leavesWithDomainOf1.pop()
			pile: Pile = one(DOTkeys(valfilter(leafIsInPileRange(leaf), pilesWithPileRangeOfLeaves)))
			sherpa: PermutationSpace | None = _reducePermutationSpace_LeafIsPinned(state, atPilePinLeaf(permutationSpace, pile, leaf))
			if not sherpa:
				return None
			else:
				permutationSpace = sherpa
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
			leavesPinnedEvenInDimension: PinnedLeaves = valfilter(complement(leafParityInDimension(dimension=dimension)), leavesPinned)
			leavesPinnedOddInDimension: PinnedLeaves = valfilter(leafParityInDimension(dimension=dimension), leavesPinned)

			dequePileLeafPileLeaf: deque[tuple[PinnedLeaves, tuple[tuple[Pile, Leaf], tuple[Pile, Leaf]]]] = deque(
												CartesianProduct((leavesPinnedOddInDimension,), combinations(leavesPinnedEvenInDimension.items(), 2)))
			dequePileLeafPileLeaf.extend(deque(CartesianProduct((leavesPinnedEvenInDimension,), combinations(leavesPinnedOddInDimension.items(), 2))))

			while dequePileLeafPileLeaf and not permutationSpaceHasNewLeaf:
				leavesPinnedParityOpposite, ((pileOf_k, leaf_k), (pileOf_r, leaf_r)) = dequePileLeafPileLeaf.pop()
				leaf_kCrease: Leaf = int(bit_flip(leaf_k, dimension))
				leaf_rCrease: Leaf = int(bit_flip(leaf_r, dimension))

				if (leaf_kCreaseIsPinned := leafIsPinned(leavesPinnedParityOpposite, leaf_kCrease)):
					pileOf_kCrease: Pile = dictionaryLeafToPile[leaf_kCrease]
				if (leaf_rCreaseIsPinned := leafIsPinned(leavesPinnedParityOpposite, leaf_rCrease)):
					pileOf_rCrease: Pile = dictionaryLeafToPile[leaf_rCrease]

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
					if thisIsAViolation(pileOf_k, pileOf_kCrease, pileOf_r, pileOf_rCrease):
						return None

				elif not leaf_kCreaseIsPinned and not leaf_rCreaseIsPinned:
				# case leaves: k, r
				# I don't think I have enough information to do anything.
					pass

				pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(DOTitems(keyfilter(thisHasThat(pilesForbidden), extractPilesWithPileRangeOfLeaves(permutationSpace))))

				sumChecksForNewLeaves = sum(map(dimensionNearest首, permutationSpace.values()))
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

				pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(DOTitems(extractPilesWithPileRangeOfLeaves(keyfilter(between(pile + inclusive, state.pileLast), permutationSpace))))

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
				pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(extractPilesWithPileRangeOfLeaves(keyfilter(between(floor, ceiling), permutationSpace)).items())

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
				pilesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(extractPilesWithPileRangeOfLeaves(keyfilter(between(floor, ceiling), permutationSpace)).items())

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
		antiPileRangeOfLeaves: PileRangeOfLeaves = getAntiPileRangeOfLeaves(state.leavesTotal, DOTvalues(extractPinnedLeaves(permutationSpace)))

		pileRangesOfLeavesToUpdate: deque[tuple[Pile, PileRangeOfLeaves]] = deque(extractPilesWithPileRangeOfLeaves(permutationSpace).items())
		while pileRangesOfLeavesToUpdate and not permutationSpaceHasNewLeaf:
			pile, leafOrPileRangeOfLeaves = pileRangesOfLeavesToUpdate.pop()

			if (ImaLeafOrPileRangeOfLeavesNotAWalrusSubscript := JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
				return None

			permutationSpace[pile] = ImaLeafOrPileRangeOfLeavesNotAWalrusSubscript
			if thisIsALeaf(permutationSpace[pile]):
				if beansWithoutCornbread(state, permutationSpace) and not (permutationSpace := pinLeafCornbread(EliminationState(state.mapShape, pile=pile, permutationSpace=permutationSpace)).permutationSpace):
					return None
				permutationSpaceHasNewLeaf = True

	return permutationSpace

@syntacticCurry
# TODO Implement `sudoku` reduction.
def sudoku(state: EliminationState, permutationSpace: PermutationSpace) -> PermutationSpace | None:  # noqa: ARG001
	"""My implementation broke `eliminationCrease` and possibly other things.

	Sudoku trick:
	in a restricted space (square, row, or column), if two numbers have the same domain of two cells, then all other numbers are excluded from those two cells.
	^^^ generalizes to if n numbers have the same domain of n cells, all other numbers are excluded from that domain of n cells.
	"""
	return permutationSpace

#======== "Beans and cornbread" functions =======

@syntacticCurry
def beansWithoutCornbread(state: EliminationState, permutationSpace: PermutationSpace) -> bool:
	return any((beans in DOTvalues(permutationSpace)) ^ (cornbread in DOTvalues(permutationSpace)) for beans, cornbread in ((一+零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))))

def pinLeafCornbread(state: EliminationState) -> EliminationState:
	leafBeans: Leaf = raiseIfNone(DOTgetPileIfLeaf(state.permutationSpace, state.pile))
	if leafBeans in [一+零, 首一(state.dimensionsTotal)]:
		leafCornbread: Leaf = one(getLeavesCreaseNext(state, leafBeans))
		state.pile += 1
	else:
		leafCornbread = one(getLeavesCreaseBack(state, leafBeans))
		state.pile -= 1

	if disqualifyAppendingLeafAtPile(state, leafCornbread):
		state.permutationSpace = {}
	else:
		state.permutationSpace = atPilePinLeaf(state.permutationSpace, state.pile, leafCornbread)

	return state

#======== Remove or disqualify `PermutationSpace` dictionaries. =======

def removeInvalidPermutationSpace(state: EliminationState) -> EliminationState:
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	for permutationSpace in listPermutationSpace:
		state.permutationSpace = permutationSpace
		if disqualifyDictionary(state):
			continue
		state.listPermutationSpace.append(permutationSpace)
	return removePermutationSpaceViolations(state)

def disqualifyDictionary(state: EliminationState) -> bool:
	return any([notEnoughOpenPiles(state)])

def notEnoughOpenPiles(state: EliminationState) -> bool:  # noqa: PLR0911
	"""Prototype.

	Check `state.permutationSpace` for enough open piles for required leaves.

	Some leaves must be before or after other leaves, such as the dimension origin leaves. For each pinned leaf, get all of the
	required leaves for before and after, and check if there are enough open piles for all of them. If the set of open piles does
	not intersect with the domain of a required leaf, return True. If a required leaf can only be pinned in one pile of the open
	piles, pin it at that pile in stateOfOpenPiles. Use the real pinning functions with the disposable stateOfOpenPiles. With the required
	leaves that are not pinned, check if there are enough open piles for them.
	"""
	workbench = EliminationState(state.mapShape, pile=state.pile, permutationSpace=state.permutationSpace)

	dictionaryConditionalLeafPredecessors: dict[Leaf, dict[Pile, list[Leaf]]] = getDictionaryConditionalLeafPredecessors(state)

	permutationSpaceHasNewLeaf: bool = True
	while permutationSpaceHasNewLeaf:
		permutationSpaceHasNewLeaf = False

		leavesPinned: PinnedLeaves = extractPinnedLeaves(workbench.permutationSpace)
		leavesFixed: tuple[Leaf, ...] = tuple(DOTvalues(leavesPinned))
		leavesNotPinned: frozenset[Leaf] = frozenset(range(workbench.leavesTotal)).difference(leavesFixed)
		pilesOpen: frozenset[Pile] = frozenset(range(workbench.pileLast + inclusive)).difference(leavesPinned.keys())

		dequePileLeaf: deque[tuple[Pile, Leaf]] =  deque(sorted(DOTitems(keyfilter(notPileLast(state.pileLast), valfilter(notLeafOriginOrLeaf零, leavesPinned)))))

		while dequePileLeaf and not permutationSpaceHasNewLeaf:
			pile, leaf = dequePileLeaf.pop()
			dimensionTail: int = dimensionNearestTail(leaf)
			dimensionHead: int = dimensionNearest首(leaf)

			@cache
			def mustBeAfterLeaf(r: Leaf, dimensionHead: int = dimensionHead) -> bool:
				return dimensionNearestTail(r) >= dimensionHead

			@cache
			def mustBeBeforeLeaf(k: Leaf, leaf: Leaf = leaf, pile: Pile = pile, dimensionTail: int = dimensionTail) -> bool:
				if dimensionNearest首(k) <= dimensionTail:
					return True
				if (mappingHasKey(dictionaryConditionalLeafPredecessors, leaf)
				and mappingHasKey(dictionaryConditionalLeafPredecessors[leaf], pile)):
					return k in dictionaryConditionalLeafPredecessors[leaf][pile]
				return False

			leavesFixedBeforePile, leavesFixedAfterPile = split_at(leavesFixed, leaf.__eq__, maxsplit=1)
			if leavesRequiredBeforePile := set(filter(mustBeBeforeLeaf, filter(leaf.__ne__, range(一, workbench.leavesTotal)))
				).intersection(leavesFixedAfterPile):
				return True

			pilesOpenAfterLeaf, pilesOpenBeforeLeaf = more_itertools_partition(pile.__lt__, pilesOpen)
			if lenIterator(pilesOpenBeforeLeaf) < len(leavesNotPinnedRequiredBeforePile := leavesNotPinned.intersection(leavesRequiredBeforePile)):
				return True

			for leaf_k in leavesNotPinnedRequiredBeforePile:
				pilesOpenFor_k: set[int] = set(pilesOpenBeforeLeaf).intersection(xmpz(workbench.permutationSpace[leaf_k]).iter_set())
				match len(pilesOpenFor_k):
					case 0:
						return True
					case 1:
						workbench.permutationSpace = atPilePinLeaf(workbench.permutationSpace, pilesOpenFor_k.pop(), leaf_k)
						permutationSpaceHasNewLeaf = True
						break
					case _:
						pass

			if permutationSpaceHasNewLeaf:
				break

			if leavesRequiredAfterPile := set(filter(mustBeAfterLeaf, filter(leaf.__ne__, range(一, workbench.leavesTotal)))
					).intersection(leavesFixedBeforePile):
				return True
			if lenIterator(pilesOpenAfterLeaf) < len(leavesNotPinnedRequiredAfterPile := leavesNotPinned.intersection(leavesRequiredAfterPile)):
				return True

			for leaf_r in leavesNotPinnedRequiredAfterPile:
				pilesOpenFor_r: set[int] = set(pilesOpenAfterLeaf).intersection(xmpz(workbench.permutationSpace[leaf_r]).iter_set())
				match len(pilesOpenFor_r):
					case 0:
						return True
					case 1:
						workbench.permutationSpace = atPilePinLeaf(workbench.permutationSpace, pilesOpenFor_r.pop(), leaf_r)
						permutationSpaceHasNewLeaf = True
						break
					case _:
						pass

	return False
