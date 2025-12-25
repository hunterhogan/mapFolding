from collections.abc import Iterable
from cytoolz.dicttoolz import dissoc, keyfilter, keyfilter as pileFilter, valfilter as leafFilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from functools import cache
from gmpy2 import bit_flip, bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from itertools import filterfalse
from mapFolding import (
	between, decreasing, exclude, inclusive, LeafOrPileRangeOfLeaves, mappingHasKey, PermutationSpace, reverseLookup)
from mapFolding._e import (
	dimensionNearestTail, dimensionNearest首, dimensionSecondNearest首, getDictionaryPileRanges, getLeafDomain,
	getLeavesCreaseBack, getLeavesCreaseNext, getZ0Z_precedence, getZ0Z_successor, howManyDimensionsHaveOddParity,
	leafInSubHyperplane, ptount, 一, 三, 二, 五, 四, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding._e._exclusions import dictionary2d5AtPileLeafExcludedByPile, dictionary2d6AtPileLeafExcludedByPile
from mapFolding._e.pinIt import (
	atPilePinLeaf, deconstructPermutationSpaceAtPile, deconstructPermutationSpaceByDomainOfLeaf, getLeaf, leafIsNotPinned,
	leafIsPinned, notLeafOriginOrLeaf零, notPileLast, oopsAllLeaves, pileIsOpen, thisIsALeaf)
from mapFolding.algorithms.iff import removePermutationSpaceViolations
from mapFolding.dataBaskets import EliminationState
from math import log, log2
from more_itertools import loops, one
from operator import add, neg, sub
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

#  ====== Boolean filters ======================

def _leafInLastPileOfDomain(pileLeaf: tuple[int, int], dimensionsTotal: int) -> bool:
	return pileLeaf[0] == int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(pileLeaf[1]))) - howManyDimensionsHaveOddParity(pileLeaf[1]) + 1

# ======= append `leavesPinned` at `pile` if qualified =======

def appendLeavesPinnedAtPile(state: EliminationState, leavesToPin: Iterable[int]) -> EliminationState:
	sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())
	disqualify: Callable[[int], bool] = disqualifyAppendingLeafAtPile(state)
	beansOrCornbread: Callable[[PermutationSpace], bool] = beansWithoutCornbread(sherpa)

	dictionaryPermutationSpace: dict[int, PermutationSpace] = deconstructPermutationSpaceAtPile(state.leavesPinned, state.pile, filterfalse(disqualify, leavesToPin))

	sherpa.listPermutationSpace.extend(leafFilter(complement(beansOrCornbread), dictionaryPermutationSpace).values())

	for leavesPinned in leafFilter(beansOrCornbread, dictionaryPermutationSpace).values():
		stateCornbread: EliminationState = pinLeafCornbread(EliminationState(state.mapShape, pile=state.pile, leavesPinned=leavesPinned))
		if stateCornbread.leavesPinned:
			sherpa.listPermutationSpace.append(stateCornbread.leavesPinned)

	sherpa = removeInvalidPermutationSpace(sherpa)
	state.listPermutationSpace.extend(sherpa.listPermutationSpace)

	return state

@syntacticCurry
def disqualifyAppendingLeafAtPile(state: EliminationState, leaf: int) -> bool:
	return any([_pileNotInRangeByLeaf(state, leaf), leafIsPinned(state.leavesPinned, leaf), not pileIsOpen(state.leavesPinned, state.pile)])

def _pileNotInRangeByLeaf(state: EliminationState, leaf: int) -> bool:
	return state.pile not in getLeafDomain(state, leaf)

# ======= "Beans and cornbread" functions =======

@syntacticCurry
def beansWithoutCornbread(state: EliminationState, leavesPinned: PermutationSpace) -> bool:
	return any((beans in leavesPinned.values()) ^ (cornbread in leavesPinned.values()) for beans, cornbread in ((一+零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))))

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
	return any([Z0Z_excluder(state), notEnoughOpenPiles(state)])

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

	Development notes for me:
	pileLeaf is atomic.
	is vs ought.
	ways to define ought:
		- if `dimensionNearest首(k) <= coordinatesOf0AtTail(r)`, then must `pileOf_k < pileOf_r`
		- generally:
			- before or after
			- parity of piles in the domain of the "other" leaf

	Sudoku trick:
		in a restricted space (square, row, or column), if two numbers have the same domain of two cells, then all other numbers are excluded from those two cells.
		^^^ generalizes to if n numbers have the same domain of n cells, all other numbers are excluded from that domain of n cells.
		Should I use this trick here, or should I use this to deconstruct dictionaries?

	"""
	Z0Z_tester = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())
	Z0Z_precedence: dict[int, dict[int, list[int]]] = getZ0Z_precedence(state)
	Z0Z_successor: dict[int, dict[int, list[int]]] = getZ0Z_successor(state)
	if state.dimensionsTotal < 6:
		Z0Z_precedence = {}
		Z0Z_successor = {}  # noqa: F841

	while True:
		Z0Z_restart = False

		def segregateLeavesPinned(pile: int, Z0Z_segregator: EliminationState = Z0Z_tester) -> tuple[set[int], set[int], set[int], set[int]]:
			leavesPinnedBeforePile: PermutationSpace = leafFilter(thisIsALeaf, pileFilter(between(0, pile - 1), Z0Z_segregator.leavesPinned))
			pilesOpenBeforeLeaf: set[int] = set(filter(pileIsOpen(Z0Z_segregator.leavesPinned), range(pile)))
			leavesPinnedAfterPile: set[int] = set(filter(thisIsALeaf, dissoc(Z0Z_segregator.leavesPinned, *leavesPinnedBeforePile.keys(), pile).values()))
			pilesOpenAfterLeaf: set[int] = set(filter(pileIsOpen(Z0Z_segregator.leavesPinned), range(pile + 1, Z0Z_segregator.pileLast + inclusive)))
			return pilesOpenAfterLeaf, pilesOpenBeforeLeaf, leavesPinnedAfterPile, set(filter(thisIsALeaf, leavesPinnedBeforePile.values()))

		for pile, leaf in sorted(pileFilter(notPileLast, leafFilter(notLeafOriginOrLeaf零, oopsAllLeaves(Z0Z_tester.leavesPinned))).items()):
			tailCoordinates: int = dimensionNearestTail(leaf)
			dimensionHead: int = dimensionNearest首(leaf)

			def notLeaf(comparand: int, leaf: int = leaf) -> bool:
				return comparand != leaf
# ruff: noqa: ERA001

			@cache
			def mustBeAfterLeaf(r: int, leaf: int = leaf, pile: int = pile, dimensionHead: int = dimensionHead, dimensionsTotal: int = Z0Z_tester.dimensionsTotal) -> bool:
				if dimensionNearestTail(r) >= dimensionHead:
					return True
				if _leafInLastPileOfDomain((pile, leaf), dimensionsTotal):
					return r == int(bit_flip(0, dimensionNearest首(leaf)).bit_flip(dimensionNearestTail(leaf)))
				# 	if r == int(bit_flip(0, dimensionNearest首(leaf)).bit_flip(dimensionNearestTail(leaf))):
				# 		return True
				# if mappingHasKey(Z0Z_successor, leaf):
				# 	if mappingHasKey(Z0Z_successor[leaf], pile):
				# 		return r in Z0Z_successor[leaf][pile]
				return False

			@cache
			def mustBeBeforeLeaf(k: int, leaf: int = leaf, pile: int = pile, tailCoordinates: int = tailCoordinates) -> bool:
				if dimensionNearest首(k) <= tailCoordinates:
					return True
				if mappingHasKey(Z0Z_precedence, leaf):
					if mappingHasKey(Z0Z_precedence[leaf], pile):
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
						Z0Z_restart = True
						break
					case _:
						pass

			if Z0Z_restart:
				break

			for r in leavesRequiredAfterPileNotPinned:
				pilesOpenFor_r: set[int] = pilesOpenAfterLeaf.intersection(set(getLeafDomain(Z0Z_tester, r)))
				match len(pilesOpenFor_r):
					case 0:
						return True
					case 1:
						Z0Z_tester.leavesPinned = atPilePinLeaf(Z0Z_tester.leavesPinned, pilesOpenFor_r.pop(), r)
						Z0Z_restart = True
						break
					case _:
						pass
			if Z0Z_restart:
				break

		if not Z0Z_restart:
			break

	return False

# ======= crease-based subroutines for analyzing a specific `pile`. =======

def _getLeavesCrease(state: EliminationState, leaf: int) -> tuple[int, ...]:
	if 0 < leaf:
		return tuple(getLeavesCreaseBack(state, abs(leaf)))
	return tuple(getLeavesCreaseNext(state, abs(leaf)))

# Second order
def pinPile一Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(getLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt首Less一: int | None = getLeaf(state.leavesPinned, state.leavesTotal - 一)

	if leafAt首Less一 and (0 < dimensionNearestTail(leafAt首Less一)):
		listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafAt首Less一) - 零, state.dimensionsTotal - 一)])
	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

def pinPile首Less一Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(getLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int | None = getLeaf(state.leavesPinned, 一)

	if leafAt一 and (leafAt一.bit_length() < state.dimensionsTotal):
		listCreaseIndicesExcluded.extend([*range(零, leafAt一.bit_length())])
	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

# Third order
def pinPile一零Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(getLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = raiseIfNone(getLeaf(state.leavesPinned, 一))
	leafAt首Less一: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - 一))

	if 1 < len(tupleLeavesCrease):
		listCreaseIndicesExcluded.append(0)
	if is_even(leafAt首Less一) and (leafAt一 == 首零(state.dimensionsTotal)+零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafAt首Less一) + 零, state.dimensionsTotal)])
	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

def pinPile首Less一零Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(getLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = raiseIfNone(getLeaf(state.leavesPinned, 一))
	leafAt首Less一: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - 一))

	if leafAt首Less一 < 首零一(state.dimensionsTotal):
		listCreaseIndicesExcluded.append(-1)
	if (leafAt首Less一 == 首零(state.dimensionsTotal)+零) and (leafAt一 != 一+零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一) - 零)])
	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

# Fourth order
def pinPile二Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(getLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = raiseIfNone(getLeaf(state.leavesPinned, 一))
	leafAt首Less一: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - 一))
	leafAt一零: int = raiseIfNone(getLeaf(state.leavesPinned, 一+零))
	leafAt首Less一零: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - (一+零)))

	if is_odd(leafAt一零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一零), 5), ptount(leafAt一零)])
		listCreaseIndicesExcluded.append((int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5)
	if is_even(leafAt一零):
		listCreaseIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][(state.dimensionsTotal - 3) - ((state.dimensionsTotal - 2) - leafInSubHyperplane(leafAt首Less一零 - (leafAt首Less一零.bit_count() - is_even(leafAt首Less一零))).bit_count()) % (state.dimensionsTotal - 2) - is_even(leafAt首Less一零): None])
		if is_even(leafAt首Less一):
			listCreaseIndicesExcluded.extend([*range(dimensionNearestTail(leafInSubHyperplane(leafAt首Less一)) - 一, (state.dimensionsTotal - 3))])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listCreaseIndicesExcluded.extend([(int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5, dimensionNearestTail(leafAt首Less一零) - 1])
		if 首零(state.dimensionsTotal)+零 < leafAt首Less一零:
			listCreaseIndicesExcluded.extend([*range(int(leafAt首Less一零 - int(bit_flip(0, dimensionNearest首(leafAt首Less一零)))).bit_length() - 1, state.dimensionsTotal - 2)])
		if ((0 < leafAt一零 - leafAt一 <= bit_flip(0, state.dimensionsTotal - 4)) and (0 < (leafAt首Less一 - leafAt一零) <= bit_flip(0, state.dimensionsTotal - 3))):
			listCreaseIndicesExcluded.extend([ptount(leafAt一零), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

def pinPile首less二Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = raiseIfNone(getLeaf(state.leavesPinned, direction(state.pile, 1)), f"I could not find an `int` type `leaf` at {direction(state.pile, 1)}.")
	tupleLeavesCrease: tuple[int, ...] = _getLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = raiseIfNone(getLeaf(state.leavesPinned, 一))
	leafAt首Less一: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - 一))
	leafAt一零: int = raiseIfNone(getLeaf(state.leavesPinned, 一+零))
	leafAt首Less一零: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - (一+零)))
	leafAt二: int = raiseIfNone(getLeaf(state.leavesPinned, 二))

	addendDimension首零: int = leafAt首Less一零 - leafAt首Less一
	addendDimension一零: int = leafAt二 - leafAt一零
	addendDimension一: int = 			 leafAt一零 - leafAt一
	addendDimension零: int =						 leafAt一 - 零

# ruff: noqa: SIM102

	if ((addendDimension一零 in [一, 二, 三, 四])
		or ((addendDimension一零 == 五) and (addendDimension首零 != 一))
		or (addendDimension一 in [二, 三])
		or ((addendDimension一 == 一) and not (addendDimension零 == addendDimension首零 and addendDimension一零 < 0))
	):
		if leafAt首Less一零 == 首一(state.dimensionsTotal):
			if addendDimension零 == 三:
				listCreaseIndicesExcluded.append(int(log2(二)))
			if addendDimension零 == 五:
				if addendDimension一 == 二:
					listCreaseIndicesExcluded.append(int(log2(二)))
				if addendDimension一 == 三:
					listCreaseIndicesExcluded.append(int(log2(三)))
			if addendDimension一零 == 三:
				listCreaseIndicesExcluded.append(int(log2(二)))

		if 0 < (tailCoordinates := dimensionNearestTail(leafAt首Less一零)) < 5:
			listCreaseIndicesExcluded.extend(list(range(tailCoordinates % 4)) or [int(log2(一))])

		if addendDimension首零 == neg(五):
			listCreaseIndicesExcluded.append(int(log2(一)))
		if addendDimension首零 == 一:
			listCreaseIndicesExcluded.append(int(log2(二)))
		if addendDimension首零 == 四:
			if addendDimension零 == 三:
				listCreaseIndicesExcluded.extend([*range(int(log2(一)), int(log2(二)) + inclusive)])
			if addendDimension一 == 一:
				if addendDimension一零 == 三:
					listCreaseIndicesExcluded.append(int(log2(二)))

		if addendDimension零 == 一:
			listCreaseIndicesExcluded.append(int(log2(一)))
			if addendDimension一零 == 三:
				listCreaseIndicesExcluded.extend([*range(int(log2(二)), int(log2(三)) + inclusive)])
			if addendDimension一零 == 四:
				listCreaseIndicesExcluded.extend([*range(int(log2(三)), int(log2(四)) + inclusive)])
		if addendDimension零 == 二:
			listCreaseIndicesExcluded.extend([*range(int(log2(一)), int(log2(二)) + inclusive)])
		if addendDimension零 == 三:
			listCreaseIndicesExcluded.append(int(log2(三)))

		if addendDimension一 == 二:
			listCreaseIndicesExcluded.append(int(log2(一)))
		if addendDimension一 == 三:
			listCreaseIndicesExcluded.extend([*range(int(log2(一)), int(log2(二)) + inclusive)])
		if addendDimension一 == 四:
			listCreaseIndicesExcluded.append(int(log2(一)))
			if addendDimension一零 == 三:
				listCreaseIndicesExcluded.extend([*range(int(log2(一)), int(log2(三)) + inclusive)])

		if addendDimension一零 == 一:
			listCreaseIndicesExcluded.append(int(log2(一)))
		if addendDimension一零 == 二:
			listCreaseIndicesExcluded.append(int(log2(二)))
		if addendDimension一零 == 三:
			listCreaseIndicesExcluded.append(int(log2(三)))
		if addendDimension一零 == 五:
			listCreaseIndicesExcluded.append(int(log2(一)))

	return list(exclude(tupleLeavesCrease, listCreaseIndicesExcluded))

# ======= Domain-based subroutines for analyzing a specific `pile`. =======

def pinPile首零Less零AfterFourthOrder(state: EliminationState) -> list[int]:
	leaf: int = -1
	leafAt一: int = raiseIfNone(getLeaf(state.leavesPinned, 一))
	leafAt首Less一: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - 一))
	leafAt一零: int = raiseIfNone(getLeaf(state.leavesPinned, 一+零))
	leafAt首Less一零: int = raiseIfNone(getLeaf(state.leavesPinned, state.leavesTotal - (一+零)))
	leafAt二: int = raiseIfNone(getLeaf(state.leavesPinned, 二))

	dictionaryPileToLeaves: dict[int, tuple[int, ...]] = getDictionaryPileRanges(state)
	listRemoveLeaves: list[int] = []

	pileExcluder: int = 一
	for dimension in range(state.dimensionsTotal):
		if dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt一:
				listRemoveLeaves.extend([一, 首零(state.dimensionsTotal) + leafAt一])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt一:
				listRemoveLeaves.extend([一 + leafAt一])
		if dimension == 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt一:
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + leafAt一 + 零])
		if dimension == state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt一:
				listRemoveLeaves.extend([首一(state.dimensionsTotal), 首一(state.dimensionsTotal) + leafAt一])
	del pileExcluder
	leaf = -1

	pileExcluder = state.leavesTotal - 一
	for dimension in range(state.dimensionsTotal):
		if dimension == 0:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt首Less一:
				listRemoveLeaves.extend([一])
		if dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt首Less一:
				listRemoveLeaves.extend([首一(state.dimensionsTotal) + leafAt首Less一])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt首Less一:
				listRemoveLeaves.extend([int(bit_flip(0, dimension)), 首一(state.dimensionsTotal) + leafAt首Less一 - (int(bit_flip(0, dimension)) - 零)])
		if 0 < dimension < state.dimensionsTotal - 3:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt首Less一:
				listRemoveLeaves.extend([零 + leafAt首Less一])
		if 0 < dimension < state.dimensionsTotal - 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAt首Less一:
				listRemoveLeaves.extend([首一(state.dimensionsTotal)])
	del pileExcluder
	leaf = -1

	pileExcluder = 一+零
	if leafAt一零 == 三+二+零:
		listRemoveLeaves.extend([二+一+零, 首零(state.dimensionsTotal)+二+零])
	if leafAt一零 == 首一(state.dimensionsTotal)+二+零:
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首一二(state.dimensionsTotal)+零, 首零一二(state.dimensionsTotal)])
	if leafAt一零 == 首一二(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零二(state.dimensionsTotal)+零])
	if leafAt一零 == 首零一(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	if is_odd(leafAt一零):
		listRemoveLeaves.extend([leafAt一零, state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAt一零))]])
		if leafAt一零 < 首零(state.dimensionsTotal):
			comebackOffset: int = state.sumsOfProductsOfDimensions[ptount(leafAt一零) + 1]
			listRemoveLeaves.extend([
				一
				, leafAt一零 + 首零(state.dimensionsTotal)-零
				, leafAt一零 + 首零(state.dimensionsTotal)-零 - comebackOffset
			])
			if ptount(leafAt一零) == 1:
				listRemoveLeaves.extend([
					state.productsOfDimensions[dimensionNearest首(leafAt一零)] + comebackOffset
					, 首零(state.dimensionsTotal) + comebackOffset
				])
		if 首零(state.dimensionsTotal) < leafAt一零:
			listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, state.productsOfDimensions[dimensionNearest首(leafAt一零) - 1]])
	del pileExcluder
	leaf = -1

	pileExcluder = state.leavesTotal - (一+零)
	if 首零(state.dimensionsTotal) < leafAt首Less一零:
		listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, leafAt首Less一零])
		if is_even(leafAt首Less一零):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])
			bit = 1
			if bit_test(leafAt首Less一零, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 2
			if bit_test(leafAt首Less一零, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				if 1 < dimensionNearestTail(leafAt首Less一零):
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 3
			if bit_test(leafAt首Less一零, bit):
				if 1 < dimensionNearestTail(leafAt首Less一零):
					listRemoveLeaves.extend([2**bit])
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
				if dimensionNearestTail(leafAt首Less一零) < bit:
					listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])

			sheepOrGoat = 0
			shepherdOfDimensions: int = int(bit_flip(0, state.dimensionsTotal - 5))
			if (leafAt首Less一零//shepherdOfDimensions) & bit_mask(5) == 0b10101:
				listRemoveLeaves.extend([二])
				sheepOrGoat = ptount(leafAt首Less一零//shepherdOfDimensions)
				if 0 < sheepOrGoat < state.dimensionsTotal - 3:
					comebackOffset = int(bit_flip(0, dimensionNearest首(leafAt首Less一零))) - 二
					listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])
				if 0 < sheepOrGoat < state.dimensionsTotal - 4:
					comebackOffset = int(bit_flip(0, raiseIfNone(dimensionSecondNearest首(leafAt首Less一零)))) - 二
					listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])

		if is_odd(leafAt首Less一零):
			listRemoveLeaves.extend([一])
			if leafAt首Less一零 & bit_mask(4) == 0b001001:
				listRemoveLeaves.extend([0b001011])
			sheepOrGoat = ptount(leafAt首Less一零)
			if 0 < sheepOrGoat < state.dimensionsTotal - 3:
				comebackOffset = int(bit_flip(0, dimensionNearest首(leafAt首Less一零))) - 一
				listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])
			if 0 < sheepOrGoat < state.dimensionsTotal - 4:
				comebackOffset = int(bit_flip(0, raiseIfNone(dimensionSecondNearest首(leafAt首Less一零)))) - 一
				listRemoveLeaves.extend([leafAt首Less一零 - comebackOffset])

	pileExcluder = 二
	if is_even(leafAt二):
		listRemoveLeaves.extend([一, leafAt二 + 零, 首零(state.dimensionsTotal)+一+零])
	if is_odd(leafAt二):
		listRemoveLeaves.extend([leafAt二 - 零])
		if 首一(state.dimensionsTotal) < leafAt二 < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零一(state.dimensionsTotal)+零])
		if 首零(state.dimensionsTotal) < leafAt二:
			listRemoveLeaves.extend([首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)+零])
			bit = 1
			if bit_test(leafAt二, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 2
			if bit_test(leafAt二, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 3
			if bit_test(leafAt二, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 4
			if bit_test(leafAt二, bit) and (leafAt二.bit_length() > 5):
				listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	del pileExcluder
	leaf = -1

	if (leafAt一零 != 首零一(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.append(一)
	if (leafAt首Less一零 != one(getLeavesCreaseBack(state, 首零(state.dimensionsTotal)+零))) and (leafAt一 == 一+零):
		listRemoveLeaves.append(首一(state.dimensionsTotal))
	if (leafAt一 == 首二(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首零一二(state.dimensionsTotal)])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt首Less一 + 零])
	if leafAt一.bit_length() < state.dimensionsTotal - 2:
		listRemoveLeaves.extend([一, leafAt首Less一 + 一])

	return sorted(set(dictionaryPileToLeaves[state.pile]).difference(set(listRemoveLeaves)))

# ======= Insanity-based subroutines for analyzing a specific `leaf`. =======

def pinLeaf首零Plus零(state: EliminationState) -> EliminationState:
	"""You need listPermutationSpace."""
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

