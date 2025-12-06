from collections.abc import Callable, Iterable, Sequence
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache
from gmpy2 import bit_flip, bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import writePython
from mapFolding import between, consecutive, decreasing, exclude, inclusive, noDuplicates, packageSettings
from mapFolding._e import (
	dimensionFourthNearest首, dimensionNearest首, dimensionSecondNearest首, dimensionThirdNearest首, howMany0coordinatesAtTail,
	howManyDimensionsHaveOddParity, leafOrigin, pileOrigin, 一, 三, 二, 零, 首一, 首一二, 首三, 首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding.dataBaskets import EliminationState
from operator import add, sub
from pathlib import Path, PurePath
from typing import Any
import pandas

# ======= Boolean filters =================================

@syntacticCurry
def filterCeiling(pile: int, dimensionsTotal: int, leaf: int) -> bool:
	return pile <  int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(leaf))) - howManyDimensionsHaveOddParity(leaf) + 2 - (leaf == leafOrigin)

@syntacticCurry
def filterFloor(pile: int, leaf: int) -> bool:
	return int(bit_flip(0, howMany0coordinatesAtTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin) <= pile

@syntacticCurry
def filterParity(pile: int, leaf: int) -> bool:
	return (pile & 1) == ((int(bit_flip(0, howMany0coordinatesAtTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)) & 1)

@syntacticCurry
def filterDoubleParity(pile: int, dimensionsTotal: int, leaf: int) -> bool:
	if leaf != 首零(dimensionsTotal)+零:
		return True
	return (pile >> 1 & 1) == ((int(bit_flip(0, howMany0coordinatesAtTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)) >> 1 & 1)

# ======= Creases and addends =================================

def getListLeavesIncrease(state: EliminationState, leaf: int) -> list[int]:
	return _getCreases(state, leaf, increase=True)

def getListLeavesDecrease(state: EliminationState, leaf: int) -> list[int]:
	return _getCreases(state, leaf, increase=False)

def _getCreases(state: EliminationState, leaf: int, *, increase: bool = True) -> list[int]:
	(listLeavesIncrease, listLeavesDecrease) = _makeCreases(leaf, state.dimensionsTotal)
	listTarget: list[int] = listLeavesIncrease if increase else listLeavesDecrease
	return list(listTarget)
@cache
def _makeCreases(leaf: int, dimensionsTotal: int) -> tuple[list[int], list[int]]:
	listLeavesCrease: list[int] = [int(bit_flip(leaf, dimension)) for dimension in range(dimensionsTotal)]

	if leaf == leafOrigin:
		listLeavesIncrease: list[int] = [1]
		listLeavesDecrease: list[int] = []
	else:
		slicingIndexStart: int = (leaf.bit_count() - 1) & 1 ^ 1
		slicingIndexEnd: int | None = dimensionNearest首(leaf) * (slicingIndexStart ^ 1) or None

		if (slicingIndexStart == 1) and is_even(leaf):
			slicingIndexStart += howMany0coordinatesAtTail(leaf)
		listLeavesIncrease = listLeavesCrease[slicingIndexStart: slicingIndexEnd]

		slicingIndexStart = (leaf.bit_count() - 1) & 1
		slicingIndexEnd = dimensionNearest首(leaf) * (slicingIndexStart ^ 1) or None

		if (slicingIndexStart == 1) and is_even(leaf):
			slicingIndexStart += howMany0coordinatesAtTail(leaf)
		listLeavesDecrease = listLeavesCrease[slicingIndexStart: slicingIndexEnd]

		if leaf == 1:
			listLeavesDecrease = [0]
	return (listLeavesIncrease, listLeavesDecrease)

# ======= (mathematical) ranges of piles ====================

def getPileRange(state: EliminationState, pile: int) -> Iterable[int]:
	return _getPileRange(pile, state.dimensionsTotal, state.mapShape, state.leavesTotal)
@cache
def _getPileRange(pile: int, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> tuple[int, ...]:
	if (dimensionsTotal > 3) and all(dimensionLength == 2 for dimensionLength in mapShape):
		parityMatch: Callable[[int], bool] = filterParity(pile)
		pileAboveFloor: Callable[[int], bool] = filterFloor(pile)
		pileBelowCeiling: Callable[[int], bool] = filterCeiling(pile, dimensionsTotal)
		matchLargerStep: Callable[[int], bool] = filterDoubleParity(pile, dimensionsTotal)

		pileRange: Iterable[int] = range(leavesTotal)
		pileRange = filter(parityMatch, pileRange)
		pileRange = filter(pileAboveFloor, pileRange)
		pileRange = filter(pileBelowCeiling, pileRange)
		return tuple(filter(matchLargerStep, pileRange))

	return tuple(range(leavesTotal))

def getDictionaryPileRanges(state: EliminationState) -> dict[int, list[int]]:
	"""At `pile`, which `leaf` values may be found in a `folding`: the mathematical range, not a Python `range` object."""
	return {pile: list(getPileRange(state, pile)) for pile in range(state.leavesTotal)}

# ======= Leaf domains ====================================

def getLeafDomain(state: EliminationState, leaf: int) -> range:
	return _getLeafDomain(leaf, state.dimensionsTotal, state.mapShape, state.leavesTotal)
@cache
def _getLeafDomain(leaf: int, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> range:
	"""The subroutines assume `dimensionLength == 2`, but I think the concept could be extended to other `mapShape`."""
	if (dimensionsTotal > 3) and all(dimensionLength == 2 for dimensionLength in mapShape):
		originPinned =  leaf == leafOrigin
		return range(
					int(bit_flip(0, howMany0coordinatesAtTail(leaf) + 1))									# `start`, first value included in the `range`.
						+ howManyDimensionsHaveOddParity(leaf)
						- 1 - originPinned
					, int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(leaf)))	# `stop`, first value excluded from the `range`.
						- howManyDimensionsHaveOddParity(leaf)
						+ 2 - originPinned
					, 2 + (2 * (leaf == 首零(dimensionsTotal)+零))											# `step`
				)
	return range(leavesTotal)

def getDomainDimension一(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
	"""The beans and cornbread and beans and cornbread dimension.

	(leaf一零, leaf一, leaf首一, leaf首零一)
	^^^ Can you see the symmetry? ^^^

	Accurate in 6 dimensions.
	"""
	domain一零: tuple[int, ...] = tuple(getLeafDomain(state, 一+零))
	domain首一: tuple[int, ...] = tuple(getLeafDomain(state, 首一(state.dimensionsTotal)))
	return _getDomainDimension一(domain一零, domain首一, state.dimensionsTotal)
@cache
def _getDomainDimension一(domain一零: tuple[int, ...], domain首一: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int, int, int], ...]:
	domainCombined: list[tuple[int, int, int, int]] = []

	for pileOfLeaf一零 in domain一零:
		domainOfLeaf首一: tuple[int, ...] = domain首一
		pilesTotal: int = len(domainOfLeaf首一)

		listIndicesPilesExcluded: list[int] = []

		if pileOfLeaf一零 <= 首二(dimensionsTotal):
			pass

		elif 首二(dimensionsTotal) < pileOfLeaf一零 < 首一(dimensionsTotal):
			listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2), *range(1 + pilesTotal // 2, 3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首一(dimensionsTotal):
			listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2)])

		elif 首一(dimensionsTotal) < pileOfLeaf一零 < 首零(dimensionsTotal)-一:
			listIndicesPilesExcluded.extend([*range(3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首零(dimensionsTotal)-一:
			listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首零(dimensionsTotal):
			listIndicesPilesExcluded.extend([*range(2, pilesTotal // 2)])

		domainOfLeaf首一 = tuple(exclude(domainOfLeaf首一, listIndicesPilesExcluded))

		domainCombined.extend([(pileOfLeaf一零, pileOfLeaf一零 + 1, pileOfLeaf首一, pileOfLeaf首一 + 1) for pileOfLeaf首一 in domainOfLeaf首一])

	return tuple(filter(noDuplicates, domainCombined))

# TODO Figure out a system for choosing the order of leaves in multi-leaf domains. If I change the order of dimension二 from (leaf
# 二一, leaf二一零, leaf二零, leaf二), the `consecutive` filter probably won't work. But that's not a reason to choose the system.
# `getDomain二一零and二一` is opposite of dimension二, which is leaf二一, leaf二一零, ..., but because it is in that order, I got
# lucky and the logic is almost identical to `getDomain二零and二`. I might be able to make them identical if I figure out more things.

# Roughly ascending:
# 2d6		Absolute	First corner
# 二一		4			4
# 二一零	3			5
# 二零		2			6
# 二		7			7

# (less) Roughly ascending:
# 首二		15			15
# 首零二	16			16
# 首零一二	17			17
# 首一二	16			18

# These are "mirror" images: the dimensions are reversed and the order is reversed, which suggests to me it is "right".
# 1. 二一, 2. 二一零, 3. 二零, 4. 二
# 1. 首二, 2. 首零二, 3. 首零一二, 4. 首一二

# 1. 二一, 4. 首一二
# 2. 二一零, 3. 首零一二,
# 3. 二零, 2. 首零二,
# 4. 二, 1. 首二,

def getDomainDimension二(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
	"""(leaf二一, leaf二一零, leaf二零, leaf二)."""
	domain二零and二: tuple[tuple[int, int], ...] = getDomain二零and二(state)
	domain二一零and二一: tuple[tuple[int, int], ...] = getDomain二一零and二一(state)
	return _getDomainDimension二(domain二零and二, domain二一零and二一, state.dimensionsTotal)
@cache
def _getDomainDimension二(domain二零and二: tuple[tuple[int, int], ...], domain二一零and二一: tuple[tuple[int, int], ...], dimensionsTotal: int) -> tuple[tuple[int, int, int, int], ...]:
	"""Domain is over-inclusive."""
	domain二零and二corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive, domain二零and二))
	domain二一零and二一corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive, domain二一零and二一))

	domainCombined: list[tuple[int, int, int, int]] = []

	# corners, 33 surplus total.
	for index, (pileOfLeaf二一零, pileOfLeaf二一) in enumerate(domain二一零and二一corners):
		domainLeaves一and0: tuple[tuple[int, int], ...] = domain二零and二corners
		pilesTotal: int = len(domainLeaves一and0)

		listIndicesPilesExcluded: list[int] = []

		excludeBelow: int = index
		listIndicesPilesExcluded.extend(range(excludeBelow))

# TODO `excludeAbove` is just wrong. Look for a different approach.

		if pileOfLeaf二一 == 二:
			excludeAbove: int = 零
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
		if pileOfLeaf二一 <= 首二(dimensionsTotal):
			excludeAbove: int = (int(bit_mask(dimensionsTotal - 1) ^ bit_mask(dimensionsTotal - 1 - dimensionNearest首(pileOfLeaf二一))) - howManyDimensionsHaveOddParity(pileOfLeaf二一)) // 2 - index
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
		if 首二(dimensionsTotal) < pileOfLeaf二一 <= 首一(dimensionsTotal):
# Surplus in excludeAbove.
# NOTE (10, 11, 38, 39),
# NOTE (14, 15, 36, 37), (14, 15, 38, 39), (14, 15, 40, 41), (14, 15, 42, 43),
# NOTE (16, 17, 42, 43), (16, 17, 44, 45), (16, 17, 46, 47),
			excludeAbove: int = index + dimensionNearest首(pileOfLeaf二一) + pilesTotal // 2
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

		# All other surplus is in the knock-out indices.
# if (首一(dimensionsTotal) < pileOfLeaf二一 < 首一二(dimensionsTotal)) and (howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1):
# 	start = 3 * pilesTotal // 4
# 	stop = 3 * pilesTotal // 4 + 4
# 	listIndicesPilesExcluded.extend([*range(start, stop, 2)])

		domainLeaves一and0 = tuple(exclude(domainLeaves一and0, listIndicesPilesExcluded))

		domainCombined.extend([(pileOfLeaf二一, pileOfLeaf二一零, pileOfLeaf二零, pileOfLeaf二) for pileOfLeaf二零, pileOfLeaf二 in domainLeaves一and0])

	# Non-corners, no surplus.
	domain二零and二nonCorners: tuple[tuple[int, int], ...] = tuple(set(domain二零and二).difference(set(domain二零and二corners)))
	domainCombined.extend([(pileOfLeaf二 - 1, pileOfLeaf二零 + 1, pileOfLeaf二零, pileOfLeaf二) for pileOfLeaf二零, pileOfLeaf二 in domain二零and二nonCorners])

	return tuple(sorted(filter(noDuplicates, set(domainCombined))))

def getDomainDimension首二(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
	"""(leaf首二, leaf首零二, leaf首零一二, leaf首一二)."""
	domainOfLeaf首二: tuple[int, ...] = tuple(getLeafDomain(state, 首二(state.dimensionsTotal)))
	domainOfLeaf首零一二: tuple[int, ...] = tuple(getLeafDomain(state, 首零一二(state.dimensionsTotal)))
	domain首零二and首二: tuple[tuple[int, int], ...] = getDomain首零二and首二(state)
	domain首零一二and首一二: tuple[tuple[int, int], ...] = getDomain首零一二and首一二(state)
	return _getDomainDimension首二(domainOfLeaf首二, domainOfLeaf首零一二, state.dimensionsTotal, state.leavesTotal, state.pileLast, domain首零二and首二, domain首零一二and首一二)
@cache
def _getDomainDimension首二(domainOfLeaf首二: tuple[int, ...], domainOfLeaf首零一二: tuple[int, ...], dimensionsTotal: int, leavesTotal: int, pileLast: int, domain首零二and首二: tuple[tuple[int, int], ...], domain首零一二and首一二: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int, int, int], ...]:
	domainCombined: list[tuple[int, int, int, int]] = []

	for pileOfLeaf首二 in domainOfLeaf首二:
		ceiling = pileOfLeaf首二 + 首零(dimensionsTotal)

		if pileOfLeaf首二 < 首一二(dimensionsTotal) - 1:
			ceiling = pileOfLeaf首二 ^ bit_mask(dimensionsTotal)

		if (首一二(dimensionsTotal) - 1 <= pileOfLeaf首二) and (howMany0coordinatesAtTail(pileOfLeaf首二 + 1) > 1):
			ceiling = leavesTotal - 二 - 一 - 零

		if (首零(dimensionsTotal) + 2 < pileOfLeaf首二) and (howMany0coordinatesAtTail(pileOfLeaf首二 + 1) == 1):
			ceiling = leavesTotal - (leavesTotal - pileOfLeaf首二) // 2

		for pileOfLeaf首零一二 in filter(between(pileOfLeaf首二 + 2, ceiling), domainOfLeaf首零一二):
			domainCombined.append((pileOfLeaf首二, pileOfLeaf首二 + 1, pileOfLeaf首零一二, pileOfLeaf首零一二 + 1))  # noqa: PERF401

	for pileOfLeaf首二 in filter(between(pileOrigin, 首零(dimensionsTotal)), domainOfLeaf首二):
		ceiling = pileOfLeaf首二 + 首零(dimensionsTotal) + inclusive
		floor = leavesTotal - 2 - pileOfLeaf首二
		step = 2
		for pileOfLeaf首零一二 in tuple(filter(between(floor, ceiling), domainOfLeaf首零一二))[0:None:step]:
			domainCombined.append((pileOfLeaf首二, pileOfLeaf首零一二 + 1, pileOfLeaf首零一二, pileOfLeaf首二 + 1))  # noqa: PERF401

	for pileOfLeaf首二 in filter(between(首零(dimensionsTotal), pileLast), domainOfLeaf首二):
		ceiling = pileLast
		floor = pileOfLeaf首二 + 4
		step = 2
		for pileOfLeaf首零一二 in tuple(filter(between(floor, ceiling), domainOfLeaf首零一二))[0:None:step]:
			domainCombined.append((pileOfLeaf首二, pileOfLeaf首零一二 + 1, pileOfLeaf首零一二, pileOfLeaf首二 + 1))  # noqa: PERF401

	domainCombined = list(filter(noDuplicates, domainCombined))
	index首二 = 0
	index首零二 = 1
	index首零一二 = 2
	index首一二 = 3
	domainCombined = [domain for domain in domainCombined if (domain[index首零二], domain[index首二]) in domain首零二and首二 and (domain[index首零一二], domain[index首一二]) in domain首零一二and首一二]

	return tuple(sorted(set(domainCombined)))

def getDomain二零and二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf二零 and leaf二."""
	domain二零: tuple[int, ...] = tuple(getLeafDomain(state, 二+零))
	domain二: tuple[int, ...] = tuple(getLeafDomain(state, 二))
	return _getDomain二零and二(domain二零, domain二, state.dimensionsTotal)
@cache
def _getDomain二零and二(domain二零: tuple[int, ...], domain二: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	domainCombined: list[tuple[int, int]] = []

	domain零: tuple[int, ...] = domain二零
	domain0: tuple[int, ...] = domain二

# ======= By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	direction: Callable[[Any, Any], Any] = add
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

# ======= By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for index, pileOfLeaf零 in enumerate(filter(between(pileOrigin, 首零(dimensionsTotal)-零), domain零)):
		listIndicesPilesExcluded: list[int] = []

		tailDimensions: int = howMany0coordinatesAtTail(pileOfLeaf零 - is_odd(pileOfLeaf零))

# ******* All differences between `_getDomain二零and二` and `_getDomain二零and二` *******
		excludeBelowAddend: int = 0
		steppingBasisForUnknownReasons: int = int(bit_mask(tailDimensions - 1).bit_flip(0))

		if pileOfLeaf零 == 二:
			listIndicesPilesExcluded.extend([*range(index + 1)])
		if pileOfLeaf零 == (首一(dimensionsTotal) + 首二(dimensionsTotal) + 首三(dimensionsTotal)):
			indexDomain0 = int(7 * pilesTotal / 8)
			indexDomain0 -= pilesFewerDomain0
			listIndicesPilesExcluded.extend([indexDomain0])

# ------- `excludeBelow` `index` ---------------------------------
		excludeBelow: int = index + excludeBelowAddend
		excludeBelow -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeBelow))

# ------- `excludeAbove` `index` ---------------------------------
		if pileOfLeaf零 <= 首一(dimensionsTotal):
			excludeAbove: int = index + (3 * pilesTotal // 4)
			excludeAbove -= pilesFewerDomain0
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
		if 首一(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal):
			excludeAbove = int(pileOfLeaf零 ^ bit_mask(dimensionsTotal)) // 2
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

# ------- Exclude by stepping: exclude ((2^tailDimensions - 1) / (2^tailDimensions))-many indices, e.g., 1/2, 3/4, 15/16, after `index` -----------------
		for dimension in range(tailDimensions):
			listIndicesPilesExcluded.extend(range(steppingBasisForUnknownReasons + int(bit_mask(dimension)), pilesTotal, int(bit_flip(0, dimension + 1))))

# ------- Exclude "knock-out" indices ---------------------------------
		if tailDimensions == 1:
			if (首二(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal)-零) and (2 < dimensionNearest首(pileOfLeaf零)):
				if dimensionSecondNearest首(pileOfLeaf零) == 零:
					indexDomain0: int = pilesTotal // 2
					indexDomain0 -= pilesFewerDomain0
					if 4 < domain0[indexDomain0].bit_length():
						listIndicesPilesExcluded.extend([indexDomain0])
					if 首一(dimensionsTotal) < pileOfLeaf零:
						indexDomain0 = -(pilesTotal // 4 - is_odd(pileOfLeaf零))
						indexDomain0 -= -(pilesFewerDomain0)
						listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					indexDomain0 = pilesTotal // 2 + 2
					indexDomain0 -= pilesFewerDomain0
					if domain0[indexDomain0] < 首零(dimensionsTotal):
						listIndicesPilesExcluded.extend([indexDomain0])
					indexDomain0 = -(pilesTotal // 4 - 2)
					indexDomain0 -= -(pilesFewerDomain0)
					if 首一(dimensionsTotal) < pileOfLeaf零:
						listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 一+零:
					indexDomain0 = -(pilesTotal // 4)
					indexDomain0 -= -(pilesFewerDomain0)
					listIndicesPilesExcluded.extend([indexDomain0])

				indexDomain0 = 3 * pilesTotal // 4
				indexDomain0 -= pilesFewerDomain0
				if pileOfLeaf零 < 首一二(dimensionsTotal):
					listIndicesPilesExcluded.extend([indexDomain0])

				if dimensionThirdNearest首(pileOfLeaf零) == 零:
					if dimensionSecondNearest首(pileOfLeaf零) == 一+零:
						listIndicesPilesExcluded.extend([indexDomain0 - 2])
					if dimensionNearest首(pileOfLeaf零) == 一+零:
						listIndicesPilesExcluded.extend([indexDomain0 - 2])

		elif 首一(dimensionsTotal) + 首三(dimensionsTotal) + is_odd(pileOfLeaf零) == pileOfLeaf零:
			indexDomain0 = (3 * pilesTotal // 4) - 1
			indexDomain0 -= pilesFewerDomain0
			listIndicesPilesExcluded.extend([indexDomain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, listIndicesPilesExcluded)])

	return tuple(sorted(set(domainCombined)))

def getDomain二一零and二一(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf二一零 and leaf二一."""
	domain二一零: tuple[int, ...] = tuple(getLeafDomain(state, 二+一+零))
	domain二一: tuple[int, ...] = tuple(getLeafDomain(state, 二+一))
	return _getDomain二一零and二一(domain二一零, domain二一, state.dimensionsTotal)
@cache
def _getDomain二一零and二一(domain二一零: tuple[int, ...], domain二一: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	domainCombined: list[tuple[int, int]] = []

	domain零: tuple[int, ...] = domain二一零
	domain0: tuple[int, ...] = domain二一

# ======= By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	direction: Callable[[Any, Any], Any] = sub
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

# ======= By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for index, pileOfLeaf零 in enumerate(filter(between(pileOrigin, 首零(dimensionsTotal)-零), domain零)):
		listIndicesPilesExcluded: list[int] = []

		tailDimensions: int = howMany0coordinatesAtTail(pileOfLeaf零 - is_odd(pileOfLeaf零))

# ******* All differences between `_getDomain二零and二` and `_getDomain二一零and二一` *******
		excludeBelowAddend: int = int(is_even(index) or tailDimensions)
		steppingBasisForUnknownReasons: int = index

# ------- `excludeBelow` `index` ---------------------------------
		excludeBelow: int = index + excludeBelowAddend
		excludeBelow -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeBelow))

# ------- `excludeAbove` `index` ---------------------------------
		if pileOfLeaf零 <= 首一(dimensionsTotal):
			excludeAbove: int = index + (3 * pilesTotal // 4)
			excludeAbove -= pilesFewerDomain0
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
		if 首一(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal):
			excludeAbove = int(pileOfLeaf零 ^ bit_mask(dimensionsTotal)) // 2
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

# ------- Exclude by stepping: exclude ((2^tailDimensions - 1) / (2^tailDimensions))-many indices, e.g., 1/2, 3/4, 15/16, after `index` -----------------
		for dimension in range(tailDimensions):
			listIndicesPilesExcluded.extend(range(steppingBasisForUnknownReasons + int(bit_mask(dimension)), pilesTotal, int(bit_flip(0, dimension + 1))))

# ------- Exclude "knock-out" indices ---------------------------------
		if tailDimensions == 1:
			if (首二(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal)-零) and (2 < dimensionNearest首(pileOfLeaf零)):
				if dimensionSecondNearest首(pileOfLeaf零) == 零:
					indexDomain0: int = pilesTotal // 2
					indexDomain0 -= pilesFewerDomain0
					if 4 < domain0[indexDomain0].bit_length():
						listIndicesPilesExcluded.extend([indexDomain0])
					if 首一(dimensionsTotal) < pileOfLeaf零:
						indexDomain0 = -(pilesTotal // 4 - is_odd(pileOfLeaf零))
						indexDomain0 -= -(pilesFewerDomain0)
						listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					indexDomain0 = pilesTotal // 2 + 2
					indexDomain0 -= pilesFewerDomain0
					if domain0[indexDomain0] < 首零(dimensionsTotal):
						listIndicesPilesExcluded.extend([indexDomain0])
					indexDomain0 = -(pilesTotal // 4 - 2)
					indexDomain0 -= -(pilesFewerDomain0)
					if 首一(dimensionsTotal) < pileOfLeaf零:
						listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 一+零:
					indexDomain0 = -(pilesTotal // 4)
					indexDomain0 -= -(pilesFewerDomain0)
					listIndicesPilesExcluded.extend([indexDomain0])

				indexDomain0 = 3 * pilesTotal // 4
				indexDomain0 -= pilesFewerDomain0
				if pileOfLeaf零 < 首一二(dimensionsTotal):
					listIndicesPilesExcluded.extend([indexDomain0])

				if dimensionThirdNearest首(pileOfLeaf零) == 零:
					if dimensionSecondNearest首(pileOfLeaf零) == 一+零:
						listIndicesPilesExcluded.extend([indexDomain0 - 2])
					if dimensionNearest首(pileOfLeaf零) == 一+零:
						listIndicesPilesExcluded.extend([indexDomain0 - 2])

		elif 首一(dimensionsTotal) + 首三(dimensionsTotal) + is_odd(pileOfLeaf零) == pileOfLeaf零:
			indexDomain0 = (3 * pilesTotal // 4) - 1
			indexDomain0 -= pilesFewerDomain0
			listIndicesPilesExcluded.extend([indexDomain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, listIndicesPilesExcluded)])

	return tuple(sorted(set(domainCombined)))

def getDomain首零二and首二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf首零二 and leaf首二."""
	domain首零二: tuple[int, ...] = tuple(getLeafDomain(state, 首零二(state.dimensionsTotal)))
	domain首二: tuple[int, ...] = tuple(getLeafDomain(state, 首二(state.dimensionsTotal)))
	return _getDomain首零二and首二(domain首零二, domain首二, state.dimensionsTotal)
@cache
def _getDomain首零二and首二(domain首零二: tuple[int, ...], domain首二: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	domainCombined: list[tuple[int, int]] = []

	domain零: tuple[int, ...] = domain首零二
	domain0: tuple[int, ...] = domain首二

# ======= By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	direction: Callable[[Any, Any], Any] = sub
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

# ======= By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for index, pileOfLeaf零 in enumerate(domain零):
		if pileOfLeaf零 < 首零(dimensionsTotal)+零:
			continue
		listIndicesPilesExcluded: list[int] = []

		tailDimensions: int = howMany0coordinatesAtTail(direction(pileOfLeaf零, is_odd(pileOfLeaf零)))

# # ------- `excludeBelow` `index` ---------------------------------
		if 首零一(dimensionsTotal) < pileOfLeaf零:
			excludeBelow: int = index + 3 - (3 * pilesTotal // 4)
		else:
			excludeBelow = 2 + (首零一(dimensionsTotal) - direction(pileOfLeaf零, is_odd(pileOfLeaf零))) // 2
		excludeBelow -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeBelow))

# ------- `excludeAbove` `index` ---------------------------------
		excludeAbove: int = index + 2 - int(bit_mask(tailDimensions))
		excludeAbove -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

# ------- Exclude by stepping: exclude ((2^tailDimensions - 1) / (2^tailDimensions))-many indices, e.g., 1/2, 3/4, 15/16, after `index` -----------------
		countFromTheEnd: int = pilesTotal - 1
		countFromTheEnd -= pilesFewerDomain0
		steppingBasisForUnknownReasons: int = countFromTheEnd - int(bit_mask(tailDimensions - 1).bit_flip(0))
		for dimension in range(tailDimensions):
			listIndicesPilesExcluded.extend(range(steppingBasisForUnknownReasons - int(bit_mask(dimension)), decreasing, decreasing * int(bit_flip(0, dimension + 1))))

# ------- Exclude "knock-out" indices ---------------------------------
		if tailDimensions == 1:
			if (dimensionThirdNearest首(pileOfLeaf零) == 一) and (二+零 <= dimensionNearest首(pileOfLeaf零)):
				indexDomain0: int = (pilesTotal // 2) + 1
				indexDomain0 -= pilesFewerDomain0
				listIndicesPilesExcluded.extend([indexDomain0])
				indexDomain0 = (pilesTotal // 4) + 1
				indexDomain0 -= pilesFewerDomain0
				listIndicesPilesExcluded.extend([indexDomain0])
				if pileOfLeaf零 < 首零一(dimensionsTotal):
					listIndicesPilesExcluded.extend([indexDomain0 - 2])

			if howManyDimensionsHaveOddParity(pileOfLeaf零) == 一:
				indexDomain0 = (pilesTotal // 4) + 3
				indexDomain0 -= pilesFewerDomain0
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 二:
					listIndicesPilesExcluded.extend([indexDomain0])
				if (((dimensionNearest首(pileOfLeaf零) == dimensionsTotal - 1) and (dimensionSecondNearest首(pileOfLeaf零) == dimensionsTotal - 3))
					or (dimensionSecondNearest首(pileOfLeaf零) == 二)):
					listIndicesPilesExcluded.extend([indexDomain0 - 2])
					indexDomain0 = (pilesTotal // 2) - 1
					indexDomain0 -= pilesFewerDomain0
					listIndicesPilesExcluded.extend([indexDomain0])

		elif (首零一(dimensionsTotal) - direction(首三(dimensionsTotal), is_odd(pileOfLeaf零))) == pileOfLeaf零:
			indexDomain0 = (pilesTotal // 4) + 2
			indexDomain0 -= pilesFewerDomain0
			listIndicesPilesExcluded.extend([indexDomain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, listIndicesPilesExcluded)])

	return tuple(sorted(set(domainCombined)))

def getDomain首零一二and首一二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf首零一二 and leaf首一二."""
	domain首零一二: tuple[int, ...] = tuple(getLeafDomain(state, 首零一二(state.dimensionsTotal)))
	domain首一二: tuple[int, ...] = tuple(getLeafDomain(state, 首一二(state.dimensionsTotal)))
	return _getDomain首零一二and首一二(domain首零一二, domain首一二, state.dimensionsTotal)
@cache
def _getDomain首零一二and首一二(domain首零一二: tuple[int, ...], domain首一二: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	domainCombined: list[tuple[int, int]] = []

	domain零: tuple[int, ...] = domain首零一二
	domain0: tuple[int, ...] = domain首一二

# ======= By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	direction: Callable[[Any, Any], Any] = add
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

# ======= By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for index, pileOfLeaf零 in enumerate(domain零):
		if pileOfLeaf零 < 首零(dimensionsTotal):
			continue
		listIndicesPilesExcluded: list[int] = []

		tailDimensions: int = howMany0coordinatesAtTail(direction(pileOfLeaf零, is_odd(pileOfLeaf零)))

# ------- `excludeBelow` `index` ---------------------------------
		if 首零一(dimensionsTotal) < pileOfLeaf零:
			excludeBelow: int = index + 1 - (3 * pilesTotal // 4)
		else:
			excludeBelow = (首零一(dimensionsTotal) - direction(pileOfLeaf零, is_odd(pileOfLeaf零))) // 2
		excludeBelow -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeBelow))

# ------- `excludeAbove` `index` ---------------------------------
		excludeAbove: int = index + 1 - int(bit_mask(tailDimensions))
		excludeAbove -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

# ------- Exclude by stepping: exclude ((2^tailDimensions - 1) / (2^tailDimensions))-many indices, e.g., 1/2, 3/4, 15/16, after `index` -----------------
		steppingBasisForUnknownReasons: int = index
		for dimension in range(tailDimensions):
			listIndicesPilesExcluded.extend(range(steppingBasisForUnknownReasons - int(bit_mask(dimension)), decreasing, decreasing * int(bit_flip(0, dimension + 1))))

# ------- Exclude "knock-out" indices ---------------------------------
		if tailDimensions == 1:
			if (dimensionThirdNearest首(pileOfLeaf零) == 一) and (二+零 <= dimensionNearest首(pileOfLeaf零)):
				indexDomain0: int = pilesTotal // 2
				indexDomain0 -= pilesFewerDomain0
				listIndicesPilesExcluded.extend([indexDomain0])
				indexDomain0 = pilesTotal // 4
				indexDomain0 -= pilesFewerDomain0
				listIndicesPilesExcluded.extend([indexDomain0])
				if pileOfLeaf零 < 首零一(dimensionsTotal):
					listIndicesPilesExcluded.extend([indexDomain0 - 2])
			if dimensionThirdNearest首(pileOfLeaf零) == 一+零:
				indexDomain0 = pilesTotal // 4
				indexDomain0 -= pilesFewerDomain0
				if dimensionFourthNearest首(pileOfLeaf零) == 一:
					listIndicesPilesExcluded.extend([indexDomain0])
			if howManyDimensionsHaveOddParity(pileOfLeaf零) == 一:
				indexDomain0 = (pilesTotal // 4) + 2
				indexDomain0 -= pilesFewerDomain0
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 二:
					listIndicesPilesExcluded.extend([indexDomain0])
				if (首零二(dimensionsTotal) < pileOfLeaf零) and (二+零 <= dimensionNearest首(pileOfLeaf零)):
					listIndicesPilesExcluded.extend([indexDomain0 - 2])
					indexDomain0 = (pilesTotal // 2) - 2
					indexDomain0 -= pilesFewerDomain0
					listIndicesPilesExcluded.extend([indexDomain0])

		elif (首零一(dimensionsTotal) - direction(首三(dimensionsTotal), is_odd(pileOfLeaf零))) == pileOfLeaf零:
			indexDomain0 = (pilesTotal // 4) + 1
			indexDomain0 -= pilesFewerDomain0
			listIndicesPilesExcluded.extend([indexDomain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, listIndicesPilesExcluded)])

	return tuple(sorted(set(domainCombined)))

def getDictionaryLeafDomains(state: EliminationState) -> dict[int, range]:
	"""For each `leaf`, the associated Python `range` defines the mathematical domain:
	1. every `pile` at which `leaf` may be found in a `folding` and
	2. in the set of all valid foldings, every `pile` at which `leaf` must be found.
	"""  # noqa: D205
	return {leaf: getLeafDomain(state, leaf) for leaf in range(state.leavesTotal)}

# ======= Specialized tools ===============================

def getDataFrameFoldings(state: EliminationState) -> pandas.DataFrame:
	pathFilename = Path(f'{packageSettings.pathPackage}/tests/dataSamples/arrayFoldingsP2d{state.dimensionsTotal}.pkl')
	arrayFoldings = pandas.read_pickle(pathFilename)  # noqa: S301
	return pandas.DataFrame(arrayFoldings)

def makeVerificationDataLeavesDomain(listDimensions: Sequence[int], listLeaves: Sequence[int | Callable[[int], int]], pathFilename: PurePath | None = None, settings: dict[str, dict[str, Any]] | None = None) -> PurePath:
	"""Create a Python module containing combined domain data for multiple leaves across multiple mapShapes.

	This function extracts the actual combined domain (the set of valid pile position tuples) for a group of leaves from pickled
	folding data. The data is used for verification in pytest tests comparing computed domains against empirical data.

	The combined domain is a set of tuples where each tuple represents the pile positions for the specified leaves in a valid
	folding. For example, if `listLeaves` is `[4, 5, 6, 7]`, each tuple has 4 elements representing the pile where each of those
	leaves appears in a folding.

	Parameters
	----------
	listDimensions : Sequence[int]
		The dimension counts to process (e.g., `[4, 5, 6]` for 2^4, 2^5, 2^6 leaf maps).
	listLeaves : Sequence[int | Callable[[int], int]]
		The leaves whose combined domain to extract. Elements can be:
		- Integers for absolute leaf indices (e.g., `4`, `5`, `6`, `7`)
		- Callables that take `dimensionsTotal` and return a leaf index (e.g., `首二`, `首零二`)
	pathFilename : PurePath | None = None
		The output file path. If `None`, defaults to `tests/dataSamples/p2DnDomain{leafNames}.py`.
	settings : dict[str, dict[str, Any]] | None = None
		Settings for `writePython` formatter. If `None`, uses defaults.

	Returns
	-------
	pathFilename : PurePath
		The path where the module was written.

	"""
	def resolveLeaf(leafSpec: int | Callable[[int], int], dimensionsTotal: int) -> int:
		return leafSpec(dimensionsTotal) if callable(leafSpec) else leafSpec

	def getLeafName(leafSpec: int | Callable[[int], int]) -> str:
		return leafSpec.__name__ if callable(leafSpec) else str(leafSpec)

	listLeafNames: list[str] = [getLeafName(leafSpec) for leafSpec in listLeaves]
	filenameLeafPart: str = '_'.join(listLeafNames)

	if pathFilename is None:
		pathFilename = Path(f"{packageSettings.pathPackage}/tests/dataSamples/p2DnDomain{filenameLeafPart}.py")
	else:
		pathFilename = Path(pathFilename)

	dictionaryDomainsByDimensions: dict[int, list[tuple[int, ...]]] = {}

	for dimensionsTotal in listDimensions:
		mapShape: tuple[int, ...] = (2,) * dimensionsTotal
		state: EliminationState = EliminationState(mapShape)
		dataframeFoldings: pandas.DataFrame = getDataFrameFoldings(state)

		listResolvedLeaves: list[int] = [resolveLeaf(leafSpec, dimensionsTotal) for leafSpec in listLeaves]

		listCombinedTuples: list[tuple[int, ...]] = []
		for indexRow in range(len(dataframeFoldings)):
			rowFolding: pandas.Series = dataframeFoldings.iloc[indexRow]
			tuplePiles: tuple[int, ...] = tuple(int(rowFolding[rowFolding == leaf].index[0]) for leaf in listResolvedLeaves)
			listCombinedTuples.append(tuplePiles)

		listUniqueTuples: list[tuple[int, ...]] = sorted(set(listCombinedTuples))
		dictionaryDomainsByDimensions[dimensionsTotal] = listUniqueTuples

	listPythonSource: list[str] = [
		'"""Verification data for combined leaf domains.',
		'',
		'This module contains empirically extracted combined domain data for leaves',
		f'{listLeafNames} across multiple mapShape configurations.',
		'',
		'Each list is named `listDomain2D{dimensionsTotal}` where `dimensionsTotal`',
		'is the exponent in the 2^dimensionsTotal mapShape, and it contains tuples representing',
		'valid pile positions for the specified leaves. The tuple order follows the original',
		'leaf argument order.',
		'"""',
		'',
	]

	for dimensionsTotal in sorted(dictionaryDomainsByDimensions):
		variableName: str = f"listDomain2D{dimensionsTotal}"
		listPythonSource.append(f'{variableName}: list[tuple[int, ...]] = {dictionaryDomainsByDimensions[dimensionsTotal]!r}')
		listPythonSource.append('')

	pythonSource: str = '\n'.join(listPythonSource)
	writePython(pythonSource, pathFilename, settings)

	return pathFilename

