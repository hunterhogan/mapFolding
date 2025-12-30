from collections.abc import Iterable
from copy import deepcopy
from cytoolz.dicttoolz import dissoc, keyfilter, valfilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from cytoolz.itertoolz import frequencies
from functools import cache
from gmpy2 import mpz
from hunterMakesPy import raiseIfNone
from itertools import filterfalse
from mapFolding import inclusive
from mapFolding._e import (
	between, dimensionNearestTail, dimensionNearest首, DOTvalues, getAntiPileRangeOfLeaves, getLeaf, getLeafDomain,
	getLeavesCreaseBack, getLeavesCreaseNext, getZ0Z_precedence, leafIsNotPinned, leafIsPinned, LeafOrPileRangeOfLeaves,
	mappingHasKey, mapShapeIs2上nDimensions, notLeafOriginOrLeaf零, notPileLast, oopsAllLeaves, oopsAllPileRangesOfLeaves,
	PermutationSpace, pileIsOpen, pileRangeOfLeavesAND, thisIsALeaf, Z0Z_JeanValjean, 一, 二, 零, 首一, 首零一)
from mapFolding._e._exclusions import dictionary2d5AtPileLeafExcludedByPile, dictionary2d6AtPileLeafExcludedByPile
from mapFolding._e.algorithms.iff import removePermutationSpaceViolations
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pinIt import atPilePinLeaf, deconstructPermutationSpaceAtPile
from more_itertools import filter_map, one
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

# ======= append `leavesPinned` at `pile` if qualified =======

def appendLeavesPinnedAtPile(state: EliminationState, leavesToPin: Iterable[int]) -> EliminationState:
	sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())
	disqualify: Callable[[int], bool] = disqualifyAppendingLeafAtPile(state)
	beansOrCornbread: Callable[[PermutationSpace], bool] = beansWithoutCornbread(sherpa)

	dictionaryPermutationSpace: dict[int, PermutationSpace] = deconstructPermutationSpaceAtPile(state.leavesPinned, state.pile, filterfalse(disqualify, leavesToPin))

	sherpa.listPermutationSpace.extend(DOTvalues(valfilter(complement(beansOrCornbread), dictionaryPermutationSpace)))

	for leavesPinned in DOTvalues(valfilter(beansOrCornbread, dictionaryPermutationSpace)):
		stateCornbread: EliminationState = pinLeafCornbread(EliminationState(state.mapShape, pile=state.pile, leavesPinned=leavesPinned))
		if stateCornbread.leavesPinned:
			sherpa.listPermutationSpace.append(stateCornbread.leavesPinned)

	sherpa = updateListPermutationSpacePileRangesOfLeaves(sherpa)

	sherpa = removeInvalidPermutationSpace(sherpa)
	state.listPermutationSpace.extend(sherpa.listPermutationSpace)

	return state

@syntacticCurry
def disqualifyAppendingLeafAtPile(state: EliminationState, leaf: int) -> bool:
	return any([_pileNotInRangeByLeaf(state, leaf), leafIsPinned(state.leavesPinned, leaf), not pileIsOpen(state.leavesPinned, state.pile)])

def _pileNotInRangeByLeaf(state: EliminationState, leaf: int) -> bool:
	return state.pile not in getLeafDomain(state, leaf)

# ======= Updating pile-ranges of leaves =======

def updateListPermutationSpacePileRangesOfLeaves(state: EliminationState) -> EliminationState:
	listPermutationSpace: list[PermutationSpace] = deepcopy(state.listPermutationSpace)
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_leafIsPinned(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = deepcopy(state.listPermutationSpace)
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_headsBeforeTails(state), listPermutationSpace))

	listPermutationSpace: list[PermutationSpace] = deepcopy(state.listPermutationSpace)
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_conditionalPredecessors(state), listPermutationSpace))

	return state

@syntacticCurry
def _conditionalPredecessors(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:
	if not mapShapeIs2上nDimensions(state.mapShape, youMustBeDimensionsTallToPinThis=6):
		return leavesPinned
	doItAgain: bool = True
	Z0Z_precedence: dict[int, dict[int, list[int]]] = getZ0Z_precedence(state)
	while doItAgain:
		doItAgain = False
		for pile, leaf in sorted(valfilter(mappingHasKey(Z0Z_precedence), keyfilter(notPileLast, valfilter(notLeafOriginOrLeaf零, oopsAllLeaves(leavesPinned)))).items()):
			if mappingHasKey(Z0Z_precedence[leaf], pile):
				antiPileRangeOfLeaves: mpz = getAntiPileRangeOfLeaves(state.leavesTotal, Z0Z_precedence[leaf][pile])
				for pileOf_k, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(keyfilter(between(pile + inclusive, state.pileLast), leavesPinned)).items():
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pileOf_k] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pileOf_k]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pileOf_k, leavesPinned=leavesPinned)).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, leavesPinned)):
						return None
					break
	return leavesPinned

@syntacticCurry
def _headsBeforeTails(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:  # noqa: PLR0911
	doItAgain: bool = True
	while doItAgain:
		doItAgain = False
		for pile, leaf in sorted(keyfilter(notPileLast, valfilter(notLeafOriginOrLeaf零, oopsAllLeaves(leavesPinned))).items()):
			dimensionTail: int = dimensionNearestTail(leaf)
			dimensionHead: int = dimensionNearest首(leaf)
			if 0 < dimensionHead:
				leaves_r = range(state.productsOfDimensions[dimensionHead], state.leavesTotal, state.productsOfDimensions[dimensionHead])
				antiPileRangeOfLeaves: mpz = getAntiPileRangeOfLeaves(state.leavesTotal, leaves_r)
				for pileOf_r, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(keyfilter(between(2, pile - inclusive), leavesPinned)).items():
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pileOf_r] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pileOf_r]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pileOf_r, leavesPinned=leavesPinned)).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, leavesPinned)):
						return None
					break
			if 0 < dimensionTail:
				leaves_k = range(0, state.sumsOfProductsOfDimensions[dimensionTail], 1)
				antiPileRangeOfLeaves: mpz = getAntiPileRangeOfLeaves(state.leavesTotal, leaves_k)
				for pileOf_k, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(keyfilter(between(pile + inclusive, state.pileLast), leavesPinned)).items():
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pileOf_k] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pileOf_k]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pileOf_k, leavesPinned=leavesPinned)).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, leavesPinned)):
						return None
					break
	return leavesPinned

@syntacticCurry
def _leafIsPinned(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:
	"""Remove `leaf` from the pile-range of leaves of other piles if `leaf` is pinned at a `pile`."""
	doItAgain: bool = True

	while doItAgain:
		doItAgain = False
		antiPileRangeOfLeaves: mpz = getAntiPileRangeOfLeaves(state.leavesTotal, DOTvalues(oopsAllLeaves(leavesPinned)))
		for pile, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(leavesPinned).items():
			if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
				return None
			leavesPinned[pile] = Z0Z_ImNotASubscript
			if thisIsALeaf(leavesPinned[pile]):
				if beansWithoutCornbread(state, leavesPinned):
					leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pile, leavesPinned=leavesPinned)).leavesPinned
					if not leavesPinned:
						return None
				doItAgain = True
				break
	return leavesPinned

@syntacticCurry
def suDONTku(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:
	"""My implementation breaks `eliminationCrease` and possibly other things.

	Sudoku trick:
	in a restricted space (square, row, or column), if two numbers have the same domain of two cells, then all other numbers are excluded from those two cells.
	^^^ generalizes to if n numbers have the same domain of n cells, all other numbers are excluded from that domain of n cells.
	"""
	doItAgain: bool = True
	while doItAgain:
		doItAgain = False
		ff: dict[mpz, int] = valfilter(between(2, 9001), frequencies(map(mpz, DOTvalues(oopsAllPileRangesOfLeaves(leavesPinned)))))
		for mpzPileRangesOfLeaves, howManyPiles in ff.items():
			pileWillAcceptThisManyDifferentLeaves: int = mpzPileRangesOfLeaves.bit_count() - 1
			if pileWillAcceptThisManyDifferentLeaves < howManyPiles:
				return None
			if pileWillAcceptThisManyDifferentLeaves == howManyPiles:
				for pile, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(leavesPinned).items():
					if mpzPileRangesOfLeaves == leafOrPileRangeOfLeaves:
						continue
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(mpzPileRangesOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pile] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pile]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pile, leavesPinned=leavesPinned)).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, leavesPinned)):
						return None
					break
	return leavesPinned

# ======= "Beans and cornbread" functions =======

@syntacticCurry
def beansWithoutCornbread(state: EliminationState, leavesPinned: PermutationSpace) -> bool:
	return any((beans in DOTvalues(leavesPinned)) ^ (cornbread in DOTvalues(leavesPinned)) for beans, cornbread in ((一+零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))))

def pinLeafCornbread(state: EliminationState) -> EliminationState:
	leafBeans: int = raiseIfNone(getLeaf(state.leavesPinned, state.pile))
	if leafBeans in [一+零, 首一(state.dimensionsTotal)]:
		leafCornbread: int = one(getLeavesCreaseNext(state, leafBeans))
		state.pile += 1
	else:
		leafCornbread = one(getLeavesCreaseBack(state, leafBeans))
		state.pile -= 1

	if disqualifyAppendingLeafAtPile(state, leafCornbread):
		state.leavesPinned = {}
	else:
		state.leavesPinned = atPilePinLeaf(state.leavesPinned, state.pile, leafCornbread)

	return state

# ======= Remove or disqualify `PermutationSpace` dictionaries. =======

def removeInvalidPermutationSpace(state: EliminationState) -> EliminationState:
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace
	state.listPermutationSpace = []
	for leavesPinned in listPermutationSpace:
		state.leavesPinned = leavesPinned
		if disqualifyDictionary(state):
			continue
		state.listPermutationSpace.append(leavesPinned)
	return removePermutationSpaceViolations(state)

def disqualifyDictionary(state: EliminationState) -> bool:
	return any([
		Z0Z_excluder(state)
	, notEnoughOpenPiles(state)
	])

def Z0Z_excluder(state: EliminationState) -> bool:
	"""{atPileExcluded: {leafExcluded: {byPileExcluder: listLeafExcluders}}}."""
	lookup: dict[int, dict[int, dict[int, list[int]]]]
	if state.dimensionsTotal == 二+一:
		lookup = dictionary2d6AtPileLeafExcludedByPile
	elif state.dimensionsTotal == 二+零:
		lookup = dictionary2d5AtPileLeafExcludedByPile
	else:
		return False

	for pileExcluded, leafExcluded in keyfilter(mappingHasKey(lookup), valfilter(notLeafOriginOrLeaf零, oopsAllLeaves(state.leavesPinned))).items():
		if pileExcluded == state.pileLast:
			continue
		if leafExcluded not in lookup[pileExcluded]:
			continue

		for pileExcluder, listLeafExcluders in keyfilter(mappingHasKey(state.leavesPinned), lookup[pileExcluded][leafExcluded]).items():
			leafExcluder: LeafOrPileRangeOfLeaves = state.leavesPinned[pileExcluder]
			if leafExcluder in listLeafExcluders:
				return True

	return False

def notEnoughOpenPiles(state: EliminationState) -> bool:
	"""Prototype.

	Some leaves must be before or after other leaves, such as the dimension origin leaves. For each pinned leaf, get all of the
	required leaves for before and after, and check if there are enough open piles for all of them. If the set of open piles does
	not intersect with the domain of a required leaf, return True. If a required leaf can only be pinned in one pile of the open
	piles, pin it at that pile in Z0Z_tester. Use the real pinning functions with the disposable Z0Z_tester. With the required
	leaves that are not pinned, somehow check if there are enough open piles for them.
	"""
	Z0Z_tester = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned)
	Z0Z_precedence: dict[int, dict[int, list[int]]] = getZ0Z_precedence(state)
	if state.dimensionsTotal < 6:
		Z0Z_precedence = {}

	doItAgain: bool = True
	while doItAgain:
		doItAgain = False

		def segregateLeavesPinned(pile: int, Z0Z_segregator: EliminationState = Z0Z_tester) -> tuple[set[int], set[int], set[int], set[int]]:
			leavesPinnedBeforePile: PermutationSpace = valfilter(thisIsALeaf, keyfilter(between(0, pile - 1), Z0Z_segregator.leavesPinned))
			pilesOpenBeforeLeaf: set[int] = set(filter(pileIsOpen(Z0Z_segregator.leavesPinned), range(pile)))
			leavesPinnedAfterPile: set[int] = set(filter(thisIsALeaf, DOTvalues(dissoc(Z0Z_segregator.leavesPinned, *leavesPinnedBeforePile.keys(), pile))))
			pilesOpenAfterLeaf: set[int] = set(filter(pileIsOpen(Z0Z_segregator.leavesPinned), range(pile + 1, Z0Z_segregator.pileLast + inclusive)))
			return pilesOpenAfterLeaf, pilesOpenBeforeLeaf, leavesPinnedAfterPile, set(filter(thisIsALeaf, DOTvalues(leavesPinnedBeforePile)))

		for pile, leaf in sorted(keyfilter(notPileLast, valfilter(notLeafOriginOrLeaf零, oopsAllLeaves(Z0Z_tester.leavesPinned))).items()):
			dimensionTail: int = dimensionNearestTail(leaf)
			dimensionHead: int = dimensionNearest首(leaf)

			def notLeaf(comparand: int, leaf: int = leaf) -> bool:
				return comparand != leaf

			@cache
			def mustBeAfterLeaf(r: int, dimensionHead: int = dimensionHead) -> bool:
				return dimensionNearestTail(r) >= dimensionHead

			@cache
			def mustBeBeforeLeaf(k: int, leaf: int = leaf, pile: int = pile, dimensionTail: int = dimensionTail) -> bool:
				if dimensionNearest首(k) <= dimensionTail:
					return True
				if mappingHasKey(Z0Z_precedence, leaf) and mappingHasKey(Z0Z_precedence[leaf], pile):
					return k in Z0Z_precedence[leaf][pile]
				return False

			pilesOpenAfterLeaf, pilesOpenBeforeLeaf, leavesPinnedAfterPile, leavesPinnedBeforePile = segregateLeavesPinned(pile, Z0Z_tester)

			if any ((
				leavesRequiredAfterPile := set(filter(mustBeAfterLeaf, filter(notLeaf, filter(notLeafOriginOrLeaf零, range(Z0Z_tester.leavesTotal))))
					).intersection(leavesPinnedBeforePile)
				, leavesRequiredBeforePile := set(filter(mustBeBeforeLeaf, filter(notLeaf, filter(notLeafOriginOrLeaf零, range(Z0Z_tester.leavesTotal))))
					).intersection(leavesPinnedAfterPile)
				, len(pilesOpenAfterLeaf) < len(leavesRequiredAfterPileNotPinned := set(filter(leafIsNotPinned(Z0Z_tester.leavesPinned), leavesRequiredAfterPile)))
				, len(pilesOpenBeforeLeaf) < len(leavesRequiredBeforePileNotPinned := set(filter(leafIsNotPinned(Z0Z_tester.leavesPinned), leavesRequiredBeforePile)))
			)):
				return True

			for k in leavesRequiredBeforePileNotPinned:
				pilesOpenFor_k: set[int] = pilesOpenBeforeLeaf.intersection(set(getLeafDomain(Z0Z_tester, k)))
				match len(pilesOpenFor_k):
					case 0:
						return True
					case 1:
						Z0Z_tester.leavesPinned = atPilePinLeaf(Z0Z_tester.leavesPinned, pilesOpenFor_k.pop(), k)
						doItAgain = True
						break
					case _:
						pass

			if doItAgain:
				break

			for r in leavesRequiredAfterPileNotPinned:
				pilesOpenFor_r: set[int] = pilesOpenAfterLeaf.intersection(set(getLeafDomain(Z0Z_tester, r)))
				match len(pilesOpenFor_r):
					case 0:
						return True
					case 1:
						Z0Z_tester.leavesPinned = atPilePinLeaf(Z0Z_tester.leavesPinned, pilesOpenFor_r.pop(), r)
						doItAgain = True
						break
					case _:
						pass
			if doItAgain:
				break

	return False

