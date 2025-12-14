from cytoolz.dicttoolz import keyfilter, valfilter
from cytoolz.functoolz import complement, curry as syntacticCurry
from functools import cache
from gmpy2 import bit_flip, bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from itertools import filterfalse
from mapFolding import between, exclude, inclusive, mappingHasKey
from mapFolding._e import (
	dimensionNearest首, dimensionSecondNearest首, getDictionaryPileRanges, getLeafDomain, getListLeavesDecrease,
	getListLeavesIncrease, howMany0coordinatesAtTail, howManyDimensionsHaveOddParity, leafInSubHyperplane, PinnedLeaves,
	ptount, Z0Z_precedence, 一, 三, 二, 五, 四, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding._e._exclusions import dictionary2d5AtPileLeafExcludedByPile, dictionary2d6AtPileLeafExcludedByPile
from mapFolding._e.pinIt import (
	atPilePinLeaf, deconstructPinnedLeavesAtPile, leafIsNotPinned, leafIsPinned, notLeafOriginOrLeaf零, notPileLast)
from mapFolding.algorithms.iff import removePinnedLeavesViolations
from mapFolding.dataBaskets import EliminationState
from math import log, log2
from operator import add, neg, sub
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

#  ====== Boolean filters ======================

def _leafInEarlyPileOfDomain(pileLeaf: tuple[int, int]) -> bool:
	if mappingHasKey(Z0Z_precedence, pileLeaf[1]):
		return mappingHasKey(Z0Z_precedence[pileLeaf[1]], pileLeaf[0])
	return False

def _leafInLastPileOfDomain(pileLeaf: tuple[int, int], dimensionsTotal: int) -> bool:
	return pileLeaf[0] == int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(pileLeaf[1]))) - howManyDimensionsHaveOddParity(pileLeaf[1]) + 1

# ======= "Beans and cornbread" functions =======

@syntacticCurry
def beansWithoutCornbread(state: EliminationState, leavesPinned: PinnedLeaves) -> bool:
	return any((beans in leavesPinned.values()) ^ (cornbread in leavesPinned.values()) for beans, cornbread in ((一+零, 一), (首一(state.dimensionsTotal), 首零一(state.dimensionsTotal))))

def pinLeafCornbread(state: EliminationState) -> EliminationState:
	leafBeans: int = state.leavesPinned[state.pile]
	if leafBeans in [一+零, 首一(state.dimensionsTotal)]:
		leafCornbread: int = getListLeavesIncrease(state, leafBeans)[-1]
		state.pile += 1
	else:
		leafCornbread = getListLeavesDecrease(state, leafBeans)[-1]
		state.pile -= 1

	if disqualifyAppendingLeafAtPile(state, leafCornbread):
		state.leavesPinned = {}
	else:
		state.leavesPinned = atPilePinLeaf(state.leavesPinned, state.pile, leafCornbread)

	return state

# ======= append `leavesPinned` at `pile` if qualified =======

def appendLeavesPinnedAtPile(state: EliminationState, listLeavesAtPile: list[int]) -> EliminationState:
	disqualify: Callable[[int], bool] = disqualifyAppendingLeafAtPile(state)
	leavesToPin: list[int] = list(filterfalse(disqualify, listLeavesAtPile))

	dictionaryPinnedLeaves: dict[int, PinnedLeaves] = deconstructPinnedLeavesAtPile(state.leavesPinned, state.pile, leavesToPin)

	sherpa: EliminationState = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())

	beansOrCornbread: Callable[[PinnedLeaves], bool] = beansWithoutCornbread(sherpa)

	sherpa.listPinnedLeaves.extend(tuple(valfilter(complement(beansOrCornbread), dictionaryPinnedLeaves).values()))

	for leavesPinned in valfilter(beansOrCornbread, dictionaryPinnedLeaves).values():
		stateCornbread: EliminationState = pinLeafCornbread(EliminationState(state.mapShape, pile=state.pile, leavesPinned=leavesPinned))
		if stateCornbread.leavesPinned:
			sherpa.listPinnedLeaves.append(stateCornbread.leavesPinned)

	sherpa = removeInvalidPinnedLeaves(sherpa)
	state.listPinnedLeaves.extend(sherpa.listPinnedLeaves)

	return state

@syntacticCurry
def disqualifyAppendingLeafAtPile(state: EliminationState, leaf: int) -> bool:
		return any([_pileNotInRangeByLeaf(state, leaf), leafIsPinned(state.leavesPinned, leaf), mappingHasKey(state.leavesPinned, state.pile)])

def _pileNotInRangeByLeaf(state: EliminationState, leaf: int) -> bool:
	return state.pile not in list(getLeafDomain(state, leaf))

# ======= Remove or disqualify `PinnedLeaves` dictionaries. =======

def removeInvalidPinnedLeaves(state: EliminationState) -> EliminationState:
	listPinnedLeaves: list[PinnedLeaves] = state.listPinnedLeaves.copy()
	state.listPinnedLeaves = []
	for leavesPinned in listPinnedLeaves:
		state.leavesPinned = leavesPinned
		if disqualifyDictionary(state):
			continue
		state.listPinnedLeaves.append(leavesPinned)
	return removePinnedLeavesViolations(state)

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

	for pileExcluded, leafExcluded in keyfilter(mappingHasKey(lookup), valfilter(notLeafOriginOrLeaf零, state.leavesPinned)).items():
		if pileExcluded == state.pileLast:
			continue
		if leafExcluded not in lookup[pileExcluded]:
			continue

		for pileExcluder, listLeafExcluders in keyfilter(mappingHasKey(state.leavesPinned), lookup[pileExcluded][leafExcluded]).items():
			leafExcluder: int = state.leavesPinned[pileExcluder]
			if leafExcluder in listLeafExcluders:
				return True

	return False

def notEnoughOpenPiles(state: EliminationState) -> bool:  # noqa: PLR0911
	"""Prototype.

	Some leaves must be before or after other leaves, such as the dimension origin leaves. For each pinned leaf, get all of the
	required leaves for before and after, and check if there are enough open piles for all of them. If the set of open piles does
	not intersect with the domain of a required leaf, return True. If a required leaf can only be pinned in one pile of the open
	piles, pin it at that pile in Z0Z_tester. Use the real pinning functions with the disposable Z0Z_tester. With the required
	leaves that are not pinned, somehow check if there are enough open piles for them.
	"""
	Z0Z_tester = EliminationState(state.mapShape, pile=state.pile, leavesPinned=state.leavesPinned.copy())

	while True:
		pilesOpen: set[int] = set(range(Z0Z_tester.leavesTotal)) - set(Z0Z_tester.leavesPinned.keys())
		Z0Z_restart = False

		for pile, leaf in sorted(keyfilter(notPileLast, valfilter(notLeafOriginOrLeaf零, Z0Z_tester.leavesPinned)).items()):
			def notLeaf(k: int, leaf: int = leaf) -> bool:
				return k != leaf
			tailCoordinates = howMany0coordinatesAtTail(leaf)

			@cache
			def mustBeBeforeLeaf(k: int, leaf: int = leaf, pile: int = pile, tailCoordinates: int = tailCoordinates) -> bool:
				if dimensionNearest首(k) <= tailCoordinates:
					return True
				if _leafInEarlyPileOfDomain((pile, leaf)):
					return k == int(bit_flip(0, dimensionNearest首(leaf)).bit_flip(howMany0coordinatesAtTail(leaf)))
				return False

			leavesRequiredBefore: set[int] = set(filter(mustBeBeforeLeaf, filter(notLeaf, filter(notLeafOriginOrLeaf零, range(Z0Z_tester.leavesTotal)))))

			if leavesRequiredBefore.intersection(keyfilter(between(pile + 1, Z0Z_tester.pileLast), Z0Z_tester.leavesPinned).values()):
				return True

			leavesRequiredBeforeNotPinned: set[int] = set(filter(leafIsNotPinned(Z0Z_tester.leavesPinned), leavesRequiredBefore))

			pilesOpenBefore: set[int] = set(filter(between(0, pile - 1), pilesOpen))
			if len(leavesRequiredBeforeNotPinned) > len(pilesOpenBefore):
				return True

			for k in leavesRequiredBeforeNotPinned:
				Z0Z_pilesValid: set[int] = pilesOpenBefore.intersection(set(getLeafDomain(Z0Z_tester, k)))
				if not Z0Z_pilesValid:
					return True
				if len(Z0Z_pilesValid) == 1:
					Z0Z_tester.leavesPinned = atPilePinLeaf(Z0Z_tester.leavesPinned, Z0Z_pilesValid.pop(), k)
					Z0Z_restart = True
					break
			if Z0Z_restart:
				break

			# 2. Identify leaves that MUST be AFTER `leaf`
			dimensionHead = dimensionNearest首(leaf)

			@cache
			def mustBeAfterLeaf(r: int, leaf: int = leaf, pile: int = pile, dimensionHead: int = dimensionHead, dimensionsTotal: int = Z0Z_tester.dimensionsTotal) -> bool:
				if howMany0coordinatesAtTail(r) >= dimensionHead:
					return True
				if _leafInLastPileOfDomain((pile, leaf), dimensionsTotal):
					return r == int(bit_flip(0, dimensionNearest首(leaf)).bit_flip(howMany0coordinatesAtTail(leaf)))
				return False

			leavesRequiredAfter: set[int] = set(filter(mustBeAfterLeaf, filter(notLeaf, filter(notLeafOriginOrLeaf零, range(Z0Z_tester.leavesTotal)))))

			if leavesRequiredAfter.intersection(keyfilter(between(0, pile), Z0Z_tester.leavesPinned).values()):
				return True

			leavesRequiredAfterNotPinned: set[int] = set(filter(leafIsNotPinned(Z0Z_tester.leavesPinned), leavesRequiredAfter))

			pilesOpenAfter: set[int] = set(filter(between(pile + 1, Z0Z_tester.pileLast), pilesOpen))
			if len(leavesRequiredAfterNotPinned) > len(pilesOpenAfter):
				return True

			for r in leavesRequiredAfterNotPinned:
				Z0Z_pilesValid: set[int] = pilesOpenAfter.intersection(set(getLeafDomain(Z0Z_tester, r)))
				if not Z0Z_pilesValid:
					return True
				if len(Z0Z_pilesValid) == 1:
					Z0Z_tester.leavesPinned = atPilePinLeaf(Z0Z_tester.leavesPinned, Z0Z_pilesValid.pop(), r)
					Z0Z_restart = True
					break
			if Z0Z_restart:
				break

		if not Z0Z_restart:
			break

	return False

# ======= crease-based subroutines for analyzing a specific `pile`. =======

def _getListLeavesCrease(state: EliminationState, leaf: int) -> list[int]:
	if 0 < leaf:
		listLeavesCrease: list[int] = getListLeavesDecrease(state, abs(leaf))
	else:
		listLeavesCrease: list[int] = getListLeavesIncrease(state, abs(leaf))
	return listLeavesCrease

# Second order
def pinPile一Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt首Less一: int | None = state.leavesPinned.get(state.leavesTotal - 一)

	if leafAt首Less一 and (0 < howMany0coordinatesAtTail(leafAt首Less一)):
		listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafAt首Less一) - 零, state.dimensionsTotal - 一)])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

def pinPile首Less一Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int | None = state.leavesPinned.get(一)

	if leafAt一 and (leafAt一.bit_length() < state.dimensionsTotal):
		listCreaseIndicesExcluded.extend([*range(零, leafAt一.bit_length())])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

# Third order
def pinPile一零Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = state.leavesPinned[一]
	leafAt首Less一: int = state.leavesPinned[state.leavesTotal - 一]

	if 1 < len(listLeavesCrease):
		listCreaseIndicesExcluded.append(0)
	if is_even(leafAt首Less一) and (leafAt一 == 首零(state.dimensionsTotal)+零):
		listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafAt首Less一) + 零, state.dimensionsTotal)])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

def pinPile首Less一零Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = state.leavesPinned[一]
	leafAt首Less一: int = state.leavesPinned[state.leavesTotal - 一]

	if leafAt首Less一 < 首零一(state.dimensionsTotal):
		listCreaseIndicesExcluded.append(-1)
	if (leafAt首Less一 == 首零(state.dimensionsTotal)+零) and (leafAt一 != 一+零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一) - 零)])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

# Fourth order
def pinPile二Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = sub

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = state.leavesPinned[一]
	leafAt首Less一: int = state.leavesPinned[state.leavesTotal - 一]
	leafAt一零: int = state.leavesPinned[一+零]
	leafAt首Less一零: int = state.leavesPinned[state.leavesTotal - (一+零)]

	if is_odd(leafAt一零):
		listCreaseIndicesExcluded.extend([*range(dimensionNearest首(leafAt一零), 5), ptount(leafAt一零)])
		listCreaseIndicesExcluded.append((int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5)
	if is_even(leafAt一零):
		listCreaseIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][(state.dimensionsTotal - 3) - ((state.dimensionsTotal - 2) - leafInSubHyperplane(leafAt首Less一零 - (leafAt首Less一零.bit_count() - is_even(leafAt首Less一零))).bit_count()) % (state.dimensionsTotal - 2) - is_even(leafAt首Less一零): None])
		if is_even(leafAt首Less一):
			listCreaseIndicesExcluded.extend([*range(howMany0coordinatesAtTail(leafInSubHyperplane(leafAt首Less一)) - 一, (state.dimensionsTotal - 3))])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listCreaseIndicesExcluded.extend([(int(log(leafInSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5, howMany0coordinatesAtTail(leafAt首Less一零) - 1])
		if 首零(state.dimensionsTotal)+零 < leafAt首Less一零:
			listCreaseIndicesExcluded.extend([*range(int(leafAt首Less一零 - int(bit_flip(0, dimensionNearest首(leafAt首Less一零)))).bit_length() - 1, state.dimensionsTotal - 2)])
		if ((0 < leafAt一零 - leafAt一 <= bit_flip(0, state.dimensionsTotal - 4)) and (0 < (leafAt首Less一 - leafAt一零) <= bit_flip(0, state.dimensionsTotal - 3))):
			listCreaseIndicesExcluded.extend([ptount(leafAt一零), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

def pinPile首less二Crease(state: EliminationState) -> list[int]:
	direction: Callable[[int, int], int] = add

	listCreaseIndicesExcluded: list[int] = []
	leafRoot: int = state.leavesPinned[direction(state.pile, 1)]
	listLeavesCrease: list[int] = _getListLeavesCrease(state, direction(0, leafRoot))

	leafAt一: int = state.leavesPinned[一]
	leafAt首Less一: int = state.leavesPinned[state.leavesTotal - 一]
	leafAt一零: int = state.leavesPinned[一+零]
	leafAt首Less一零: int = state.leavesPinned[state.leavesTotal - (一+零)]
	leafAt二: int = state.leavesPinned[二]

	addendDimension首零: int = leafAt首Less一零 - leafAt首Less一
	addendDimension零: int = 						leafAt一 - 零
	addendDimension一: int = 			leafAt一零 - leafAt一
	addendDimension一零: int = leafAt二 - leafAt一零

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

		if 0 < (tailCoordinates := howMany0coordinatesAtTail(leafAt首Less一零)) < 5:
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

	return list(exclude(listLeavesCrease, listCreaseIndicesExcluded))

# ======= Domain-based subroutines for analyzing a specific `pile`. =======

def pinPile首零Less零AfterFourthOrder(state: EliminationState) -> list[int]:
	leaf: int = -1
	sumsProductsOfDimensions: list[int] = [sum(state.productsOfDimensions[0:dimension]) for dimension in range(state.dimensionsTotal + inclusive)]

	dictionaryPileToLeaves: dict[int, list[int]] = getDictionaryPileRanges(state)
	listRemoveLeaves: list[int] = []

	pileExcluder: int = 一
	leafAtPileExcluder: int = state.leavesPinned[pileExcluder]
	for dimension in range(state.dimensionsTotal):
		if dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一, 首零(state.dimensionsTotal) + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一 + leafAtPileExcluder])
		if dimension == 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + leafAtPileExcluder + 零])
		if dimension == state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal), 首一(state.dimensionsTotal) + leafAtPileExcluder])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	pileExcluder = state.leavesTotal - 一
	leafAtPileExcluder = state.leavesPinned[pileExcluder]
	for dimension in range(state.dimensionsTotal):
		if dimension == 0:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一])
		if dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal) + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([int(bit_flip(0, dimension)), 首一(state.dimensionsTotal) + leafAtPileExcluder - (int(bit_flip(0, dimension)) - 零)])
		if 0 < dimension < state.dimensionsTotal - 3:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([零 + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal)])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	pileExcluder = 一+零
	leafAtPileExcluder = state.leavesPinned[pileExcluder]
	if leafAtPileExcluder == 三+二+零:
		listRemoveLeaves.extend([二+一+零, 首零(state.dimensionsTotal)+二+零])
	if leafAtPileExcluder == 首一(state.dimensionsTotal)+二+零:
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首一二(state.dimensionsTotal)+零, 首零一二(state.dimensionsTotal)])
	if leafAtPileExcluder == 首一二(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零二(state.dimensionsTotal)+零])
	if leafAtPileExcluder == 首零一(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	if is_odd(leafAtPileExcluder):
		listRemoveLeaves.extend([leafAtPileExcluder, state.productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder))]])
		if leafAtPileExcluder < 首零(state.dimensionsTotal):
			comebackOffset: int = sumsProductsOfDimensions[ptount(leafAtPileExcluder) + 1]
			listRemoveLeaves.extend([
				一
				, leafAtPileExcluder + 首零(state.dimensionsTotal)-零
				, leafAtPileExcluder + 首零(state.dimensionsTotal)-零 - comebackOffset
			])
			if ptount(leafAtPileExcluder) == 1:
				listRemoveLeaves.extend([
					state.productsOfDimensions[dimensionNearest首(leafAtPileExcluder)] + comebackOffset
					, 首零(state.dimensionsTotal) + comebackOffset
				])
		if 首零(state.dimensionsTotal) < leafAtPileExcluder:
			listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, state.productsOfDimensions[dimensionNearest首(leafAtPileExcluder) - 1]])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	pileExcluder = state.leavesTotal - (一+零)
	leafAtPileExcluder = state.leavesPinned[pileExcluder]
	if 首零(state.dimensionsTotal) < leafAtPileExcluder:
		listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, leafAtPileExcluder])
		if is_even(leafAtPileExcluder):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])
			bit = 1
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 2
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				if 1 < howMany0coordinatesAtTail(leafAtPileExcluder):
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 3
			if bit_test(leafAtPileExcluder, bit):
				if 1 < howMany0coordinatesAtTail(leafAtPileExcluder):
					listRemoveLeaves.extend([2**bit])
					listRemoveLeaves.extend([state.leavesTotal - sum(state.productsOfDimensions[bit: state.dimensionsTotal - 2])])
				if howMany0coordinatesAtTail(leafAtPileExcluder) < bit:
					listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])

			sheepOrGoat = 0
			shepherdOfDimensions: int = int(bit_flip(0, state.dimensionsTotal - 5))
			if (leafAtPileExcluder//shepherdOfDimensions) & bit_mask(5) == 0b10101:
				listRemoveLeaves.extend([二])
				sheepOrGoat = ptount(leafAtPileExcluder//shepherdOfDimensions)
				if 0 < sheepOrGoat < state.dimensionsTotal - 3:
					comebackOffset = int(bit_flip(0, dimensionNearest首(leafAtPileExcluder))) - 二
					listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])
				if 0 < sheepOrGoat < state.dimensionsTotal - 4:
					comebackOffset = int(bit_flip(0, raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder)))) - 二
					listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])

		if is_odd(leafAtPileExcluder):
			listRemoveLeaves.extend([一])
			if leafAtPileExcluder & bit_mask(4) == 0b001001:
				listRemoveLeaves.extend([0b001011])
			sheepOrGoat = ptount(leafAtPileExcluder)
			if 0 < sheepOrGoat < state.dimensionsTotal - 3:
				comebackOffset = int(bit_flip(0, dimensionNearest首(leafAtPileExcluder))) - 一
				listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])
			if 0 < sheepOrGoat < state.dimensionsTotal - 4:
				comebackOffset = int(bit_flip(0, raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder)))) - 一
				listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])

	pileExcluder = 二
	leafAtPileExcluder = state.leavesPinned[pileExcluder]

	if is_even(leafAtPileExcluder):
		listRemoveLeaves.extend([一, leafAtPileExcluder + 零, 首零(state.dimensionsTotal)+一+零])
	if is_odd(leafAtPileExcluder):
		listRemoveLeaves.extend([leafAtPileExcluder - 零])
		if 首一(state.dimensionsTotal) < leafAtPileExcluder < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零一(state.dimensionsTotal)+零])
		if 首零(state.dimensionsTotal) < leafAtPileExcluder:
			listRemoveLeaves.extend([首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)+零])
			bit = 1
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 2
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 3
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 4
			if bit_test(leafAtPileExcluder, bit) and (leafAtPileExcluder.bit_length() > 5):
				listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	del leafAtPileExcluder, pileExcluder
	leaf = -1

	leafAt一: int = state.leavesPinned[一]
	leafAt首Less一: int = state.leavesPinned[state.leavesTotal - 一]
	leafAt一零: int = state.leavesPinned[一+零]
	leafAt首Less一零: int = state.leavesPinned[state.leavesTotal - (一+零)]

	if (leafAt一零 != 首零一(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.append(一)
	if (leafAt首Less一零 != getListLeavesDecrease(state, 首零(state.dimensionsTotal)+零)[0]) and (leafAt一 == 一+零):
		listRemoveLeaves.append(首一(state.dimensionsTotal))
	if (leafAt一 == 首二(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首零一二(state.dimensionsTotal)])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt首Less一 + 零])
	if leafAt一.bit_length() < state.dimensionsTotal - 2:
		listRemoveLeaves.extend([一, leafAt首Less一 + 一])

	return sorted(set(dictionaryPileToLeaves[state.pile]).difference(set(listRemoveLeaves)))

