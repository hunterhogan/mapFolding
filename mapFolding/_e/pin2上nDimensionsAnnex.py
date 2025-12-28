from collections.abc import Iterable
from copy import deepcopy
from cytoolz.dicttoolz import dissoc, keyfilter, keyfilter as pileFilter, valfilter, valfilter as leafFilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from cytoolz.itertoolz import frequencies
from functools import cache
from gmpy2 import bit_mask, mpz
from hunterMakesPy import raiseIfNone
from itertools import filterfalse
from mapFolding import decreasing, inclusive
from mapFolding._e import (
	between, dimensionNearestTail, dimensionNearest首, DOTvalues, exclude, getLeaf, getLeafDomain, getLeavesCreaseBack,
	getLeavesCreaseNext, getZ0Z_precedence, leafIsNotPinned, leafIsPinned, LeafOrPileRangeOfLeaves, mappingHasKey,
	mapShapeIs2上nDimensions, notLeafOriginOrLeaf零, notPileLast, oopsAllLeaves, oopsAllPileRangesOfLeaves, PermutationSpace,
	pileIsOpen, pileRangeOfLeavesAND, reverseLookup, thisIsALeaf, Z0Z_JeanValjean, 一, 二, 零, 首一, 首二, 首零, 首零一, 首零一二)
from mapFolding._e._exclusions import dictionary2d5AtPileLeafExcludedByPile, dictionary2d6AtPileLeafExcludedByPile
from mapFolding._e.algorithms.iff import removePermutationSpaceViolations
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pinIt import (
	atPilePinLeaf, deconstructPermutationSpaceAtPile, deconstructPermutationSpaceByDomainOfLeaf,
	get_mpzAntiPileRangeOfLeaves)
from more_itertools import filter_map, loops, one
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

	listPermutationSpace: list[PermutationSpace] = deepcopy(state.listPermutationSpace)
	state.listPermutationSpace = []
	state.listPermutationSpace.extend(filter_map(_sudoku(state), listPermutationSpace))

	return state

@syntacticCurry
def _conditionalPredecessors(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:
	if not mapShapeIs2上nDimensions(state, youMustBeDimensionsTallToPinThis=6):
		return leavesPinned
	doItAgain: bool = True
	Z0Z_precedence: dict[int, dict[int, list[int]]] = getZ0Z_precedence(state)
	while doItAgain:
		doItAgain = False
		for pile, leaf in sorted(leafFilter(mappingHasKey(Z0Z_precedence), pileFilter(notPileLast, leafFilter(notLeafOriginOrLeaf零, oopsAllLeaves(deepcopy(leavesPinned))))).items()):
			if mappingHasKey(Z0Z_precedence[leaf], pile):
				antiPileRangeOfLeaves: mpz = get_mpzAntiPileRangeOfLeaves(state.leavesTotal, Z0Z_precedence[leaf][pile])
				for pileOf_k, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(pileFilter(between(pile + inclusive, state.pileLast), leavesPinned)).items():
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pileOf_k] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pileOf_k]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pileOf_k, leavesPinned=deepcopy(leavesPinned))).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, deepcopy(leavesPinned))):
						return None
					break
	return leavesPinned

@syntacticCurry
def _headsBeforeTails(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:  # noqa: PLR0911
	doItAgain: bool = True
	while doItAgain:
		doItAgain = False
		for pile, leaf in sorted(pileFilter(notPileLast, leafFilter(notLeafOriginOrLeaf零, oopsAllLeaves(deepcopy(leavesPinned)))).items()):
			dimensionTail: int = dimensionNearestTail(leaf)
			dimensionHead: int = dimensionNearest首(leaf)
			if 0 < dimensionHead:
				leaves_r = range(state.productsOfDimensions[dimensionHead], state.leavesTotal, state.productsOfDimensions[dimensionHead])
				antiPileRangeOfLeaves: mpz = get_mpzAntiPileRangeOfLeaves(state.leavesTotal, leaves_r)
				for pileOf_r, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(pileFilter(between(2, pile - inclusive), leavesPinned)).items():
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pileOf_r] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pileOf_r]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pileOf_r, leavesPinned=deepcopy(leavesPinned))).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, deepcopy(leavesPinned))):
						return None
					break
			if 0 < dimensionTail:
				leaves_k = range(0, state.sumsOfProductsOfDimensions[dimensionTail], 1)
				antiPileRangeOfLeaves: mpz = get_mpzAntiPileRangeOfLeaves(state.leavesTotal, leaves_k)
				for pileOf_k, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(pileFilter(between(pile + inclusive, state.pileLast), leavesPinned)).items():
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pileOf_k] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pileOf_k]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pileOf_k, leavesPinned=deepcopy(leavesPinned))).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, deepcopy(leavesPinned))):
						return None
					break
	return leavesPinned

@syntacticCurry
def _leafIsPinned(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:
	"""Remove `leaf` from the pile-range of leaves of other piles if `leaf` is pinned at a `pile`."""
	doItAgain: bool = True

	while doItAgain:
		doItAgain = False
		antiPileRangeOfLeaves: mpz = get_mpzAntiPileRangeOfLeaves(state.leavesTotal, DOTvalues(oopsAllLeaves(deepcopy(leavesPinned))))
		for pile, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(deepcopy(leavesPinned)).items():
			if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(antiPileRangeOfLeaves, leafOrPileRangeOfLeaves))) is None:
				return None
			leavesPinned[pile] = Z0Z_ImNotASubscript
			if thisIsALeaf(leavesPinned[pile]):
				if beansWithoutCornbread(state, leavesPinned):
					leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pile, leavesPinned=deepcopy(leavesPinned))).leavesPinned
					if not leavesPinned:
						return None
				doItAgain = True
				break
	return leavesPinned

@syntacticCurry
def _sudoku(state: EliminationState, leavesPinned: PermutationSpace) -> PermutationSpace | None:
	"""There must be some algorithms for finding more of these cases.

	Sudoku trick:
	in a restricted space (square, row, or column), if two numbers have the same domain of two cells, then all other numbers are excluded from those two cells.
	^^^ generalizes to if n numbers have the same domain of n cells, all other numbers are excluded from that domain of n cells.
	"""
	doItAgain: bool = True
	while doItAgain:
		doItAgain = False
		ff: dict[mpz, int] = valfilter(between(2, 9001), frequencies(map(mpz, DOTvalues(oopsAllPileRangesOfLeaves(deepcopy(leavesPinned))))))
		for mpzPileRangesOfLeaves, howManyPiles in ff.items():
			pileWillAcceptThisManyDifferentLeaves: int = mpzPileRangesOfLeaves.bit_count() - 1
			if pileWillAcceptThisManyDifferentLeaves < howManyPiles:
				return None
			if pileWillAcceptThisManyDifferentLeaves == howManyPiles:
				for pile, leafOrPileRangeOfLeaves in oopsAllPileRangesOfLeaves(deepcopy(leavesPinned)).items():
					if mpzPileRangesOfLeaves == leafOrPileRangeOfLeaves:
						continue
					if (Z0Z_ImNotASubscript := Z0Z_JeanValjean(pileRangeOfLeavesAND(mpzPileRangesOfLeaves, leafOrPileRangeOfLeaves))) is None:
						return None
					leavesPinned[pile] = Z0Z_ImNotASubscript
					if thisIsALeaf(leavesPinned[pile]):
						if beansWithoutCornbread(state, leavesPinned):
							leavesPinned = pinLeafCornbread(EliminationState(state.mapShape, pile=pile, leavesPinned=deepcopy(leavesPinned))).leavesPinned
							if not leavesPinned:
								return None
						doItAgain = True
				if doItAgain:
					if not (leavesPinned := _leafIsPinned(state, deepcopy(leavesPinned))):
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
	listPermutationSpace: list[PermutationSpace] = state.listPermutationSpace.copy()
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

	for pileExcluded, leafExcluded in keyfilter(mappingHasKey(lookup), leafFilter(notLeafOriginOrLeaf零, oopsAllLeaves(state.leavesPinned))).items():
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
	Z0Z_tester = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())
	Z0Z_precedence: dict[int, dict[int, list[int]]] = getZ0Z_precedence(state)
	if state.dimensionsTotal < 6:
		Z0Z_precedence = {}

	doItAgain: bool = True
	while doItAgain:
		doItAgain = False

		def segregateLeavesPinned(pile: int, Z0Z_segregator: EliminationState = Z0Z_tester) -> tuple[set[int], set[int], set[int], set[int]]:
			leavesPinnedBeforePile: PermutationSpace = leafFilter(thisIsALeaf, pileFilter(between(0, pile - 1), Z0Z_segregator.leavesPinned))
			pilesOpenBeforeLeaf: set[int] = set(filter(pileIsOpen(Z0Z_segregator.leavesPinned), range(pile)))
			leavesPinnedAfterPile: set[int] = set(filter(thisIsALeaf, dissoc(Z0Z_segregator.leavesPinned, *leavesPinnedBeforePile.keys(), pile).values()))
			pilesOpenAfterLeaf: set[int] = set(filter(pileIsOpen(Z0Z_segregator.leavesPinned), range(pile + 1, Z0Z_segregator.pileLast + inclusive)))
			return pilesOpenAfterLeaf, pilesOpenBeforeLeaf, leavesPinnedAfterPile, set(filter(thisIsALeaf, leavesPinnedBeforePile.values()))

		for pile, leaf in sorted(pileFilter(notPileLast, leafFilter(notLeafOriginOrLeaf零, oopsAllLeaves(Z0Z_tester.leavesPinned))).items()):
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

# ======= Insanity-based subroutines for analyzing a specific `leaf`. =======

def pinLeaf首零Plus零(state: EliminationState) -> EliminationState:
	"""You need `state.listPermutationSpace`."""
	leaf: int = 首零(state.dimensionsTotal)+零
	listPermutationSpaceCopy: list[PermutationSpace] = state.listPermutationSpace.copy()
	state.listPermutationSpace = []
	qualifiedLeavesPinned: list[PermutationSpace] = []
	for leavesPinned in listPermutationSpaceCopy:
		state.leavesPinned = leavesPinned.copy()

		domainOfPilesForLeaf: list[int] = list(getLeafDomain(state, leaf))

		listIndicesPilesExcluded: list[int] = []
		leaf首零一: int = 首零一(state.dimensionsTotal)
		if (一+零 in state.leavesPinned.values()) and (leaf首零一 in state.leavesPinned.values()):
			pileOfLeaf一零: int = reverseLookup(state.leavesPinned, 一+零)
			pileOfLeaf首零一: int = reverseLookup(state.leavesPinned, leaf首零一)
			# Before the new symbols, I didn't see the symmetry of `leaf一零` and `leaf首零一`.

			pilesTotal: int = 首一(state.dimensionsTotal)

			bump: int = 1 - int(pileOfLeaf一零.bit_count() == 1)
			howMany: int = state.dimensionsTotal - (pileOfLeaf一零.bit_length() + bump)
			onesInBinary = int(bit_mask(howMany))
			ImaPattern: int = pilesTotal - onesInBinary

			if pileOfLeaf一零 == 二:
				listIndicesPilesExcluded.extend([零, 一, 二]) # These symbols make this pattern jump out.

			if 二 < pileOfLeaf一零 <= 首二(state.dimensionsTotal):
				stop: int = pilesTotal // 2 - 1
				listIndicesPilesExcluded.extend(range(1, stop))

				aDimensionPropertyNotFullyUnderstood = 5
				for _dimension in loops(state.dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
					start: int = 1 + stop
					stop += (stop+1) // 2
					listIndicesPilesExcluded.extend([*range(start, stop)])

				listIndicesPilesExcluded.extend([*range(1 + stop, ImaPattern)])

			if 首二(state.dimensionsTotal) < pileOfLeaf一零:
				listIndicesPilesExcluded.extend([*range(1, ImaPattern)])

			bump = 1 - int((state.leavesTotal - pileOfLeaf首零一).bit_count() == 1)
			howMany = state.dimensionsTotal - ((state.leavesTotal - pileOfLeaf首零一).bit_length() + bump)
			onesInBinary = int(bit_mask(howMany))
			ImaPattern = pilesTotal - onesInBinary

			aDimensionPropertyNotFullyUnderstood = 5

			if pileOfLeaf首零一 == state.leavesTotal-二:
				listIndicesPilesExcluded.extend([-零 -1, -(一) -1])
				if aDimensionPropertyNotFullyUnderstood <= state.dimensionsTotal:
					listIndicesPilesExcluded.extend([-二 -1])

			if ((首零一二(state.dimensionsTotal) < pileOfLeaf首零一 < state.leavesTotal-二)
				and (首二(state.dimensionsTotal) < pileOfLeaf一零 <= 首零(state.dimensionsTotal))):
				listIndicesPilesExcluded.extend([-1])

			if 首零一二(state.dimensionsTotal) <= pileOfLeaf首零一 < state.leavesTotal-二:
				stop: int = pilesTotal // 2 - 1
				listIndicesPilesExcluded.extend(range((1 + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing))

				for _dimension in loops(state.dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
					start: int = 1 + stop
					stop += (stop+1) // 2
					listIndicesPilesExcluded.extend([*range((start + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing)])

				listIndicesPilesExcluded.extend([*range((1 + stop + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])

				if 二 <= pileOfLeaf一零 <= 首零(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([零, 一, 二, pilesTotal//2])

			if ((pileOfLeaf首零一 == 首零一二(state.dimensionsTotal))
				and (首一(state.dimensionsTotal) < pileOfLeaf一零 <= 首零(state.dimensionsTotal))):
				listIndicesPilesExcluded.extend([-1])

			if 首零一(state.dimensionsTotal) < pileOfLeaf首零一 < 首零一二(state.dimensionsTotal):
				if pileOfLeaf一零 in [首一(state.dimensionsTotal), 首零(state.dimensionsTotal)]:
					listIndicesPilesExcluded.extend([-1])
				elif 二 < pileOfLeaf一零 < 首二(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([0])

			if pileOfLeaf首零一 < 首零一二(state.dimensionsTotal):
				listIndicesPilesExcluded.extend([*range((1 + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])

			pileOfLeaf一零ARCHETYPICAL: int = 首一(state.dimensionsTotal)
			bump = 1 - int(pileOfLeaf一零ARCHETYPICAL.bit_count() == 1)
			howMany = state.dimensionsTotal - (pileOfLeaf一零ARCHETYPICAL.bit_length() + bump)
			onesInBinary = int(bit_mask(howMany))
			ImaPattern = pilesTotal - onesInBinary

			if pileOfLeaf首零一 == state.leavesTotal-二:
				if pileOfLeaf一零 == 二:
					listIndicesPilesExcluded.extend([零, 一, 二, pilesTotal//2 -1, pilesTotal//2])
				if 二 < pileOfLeaf一零 <= 首零(state.dimensionsTotal):
					IDK = ImaPattern - 1
					listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4), *range(1 + 3 * pilesTotal // 4, IDK)])
				if 首一(state.dimensionsTotal) < pileOfLeaf一零 <= 首零(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([-1])

			if pileOfLeaf首零一 == 首零一(state.dimensionsTotal):
				if pileOfLeaf一零 == 首零(state.dimensionsTotal):
					listIndicesPilesExcluded.extend([-1])
				elif (二 < pileOfLeaf一零 < 首二(state.dimensionsTotal)) or (首二(state.dimensionsTotal) < pileOfLeaf一零 < 首一(state.dimensionsTotal)):
					listIndicesPilesExcluded.extend([0])
		domainOfPilesForLeaf = list(exclude(domainOfPilesForLeaf, listIndicesPilesExcluded))

		state.listPermutationSpace = deconstructPermutationSpaceByDomainOfLeaf(leavesPinned, leaf, domainOfPilesForLeaf)
		state = removeInvalidPermutationSpace(state)
		qualifiedLeavesPinned.extend(state.listPermutationSpace)
		state.listPermutationSpace = []
	state.listPermutationSpace = qualifiedLeavesPinned

	return state

