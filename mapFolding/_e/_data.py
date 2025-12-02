from collections.abc import Callable, Iterable
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache
from gmpy2 import bit_flip, bit_mask, bit_test, is_even
from mapFolding import between, consecutive, decreasing, exclude, inclusive, noDuplicates
from mapFolding._e import (
	dimensionNearest首, dimensionSecondNearest首, howMany0coordinatesAtTail, howManyDimensionsHaveOddParity, leafOrigin,
	pileOrigin, 一, 三, 二, 零, 首一, 首一二, 首三, 首二, 首零, 首零一二, 首零二)
from mapFolding.dataBaskets import EliminationState

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

# TODO Figure out a system for choosing the order of leaves in multi-leaf domains.
# If I change the order of dimension二 from (leaf二一, leaf二一零, leaf二零, leaf二), the `consecutive` filter probably won't work. But that's not a reason to choose the system.

def getDomainDimension二(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
	"""(leaf二一, leaf二一零, leaf二零, leaf二).

	Clarified identifier semantics: each local variable whose value sequence enumerates possible piles for a specific leaf
	is now prefixed with `domainOfLeaf` and each individual pile loop variable is prefixed with `pileOfLeaf`.
	"""
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
		if (首一(dimensionsTotal) < pileOfLeaf二一 < 首一二(dimensionsTotal)) and (howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1):
			start = 3 * pilesTotal // 4
			stop = 3 * pilesTotal // 4 + 4
			listIndicesPilesExcluded.extend([*range(start, stop, 2)])

		domainLeaves一and0 = tuple(exclude(domainLeaves一and0, listIndicesPilesExcluded))

		domainCombined.extend([(pileOfLeaf二一, pileOfLeaf二一零, pileOfLeaf二零, pileOfLeaf二) for pileOfLeaf二零, pileOfLeaf二 in domainLeaves一and0])

	# Non-corners, no surplus.
	domain二零and二nonCorners: tuple[tuple[int, int], ...] = tuple(set(domain二零and二).difference(set(domain二零and二corners)))
	domainCombined.extend([(pileOfLeaf二 - 1, pileOfLeaf二零 + 1, pileOfLeaf二零, pileOfLeaf二) for pileOfLeaf二零, pileOfLeaf二 in domain二零and二nonCorners])

	return tuple(domainCombined)

def getDomainDimension首二(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
	domainOfLeaf首二: tuple[int, ...] = tuple(getLeafDomain(state, 首二(state.dimensionsTotal)))
	domainOfLeaf首零一二: tuple[int, ...] = tuple(getLeafDomain(state, 首零一二(state.dimensionsTotal)))
	return _getDomainDimension首二(domainOfLeaf首二, domainOfLeaf首零一二, state.dimensionsTotal, state.leavesTotal, state.pileLast)
@cache
def _getDomainDimension首二(domainOfLeaf首二: tuple[int, ...], domainOfLeaf首零一二: tuple[int, ...], dimensionsTotal: int, leavesTotal: int, pileLast: int) -> tuple[tuple[int, int, int, int], ...]:
	domainCombined: list[tuple[int, int, int, int]] = []

	# 46 surplus tuples
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

	# 22 surplus tuples
	for pileOfLeaf首二 in filter(between(pileOrigin, 首零(dimensionsTotal)), domainOfLeaf首二):
		ceiling = pileOfLeaf首二 + 首零(dimensionsTotal) + inclusive
		floor = leavesTotal - 2 - pileOfLeaf首二
		step = 2
		for pileOfLeaf首零一二 in tuple(filter(between(floor, ceiling), domainOfLeaf首零一二))[0:None:step]:
			domainCombined.append((pileOfLeaf首二, pileOfLeaf首零一二 + 1, pileOfLeaf首零一二, pileOfLeaf首二 + 1))  # noqa: PERF401

	# 18 surplus tuples
	for pileOfLeaf首二 in filter(between(首零(dimensionsTotal), pileLast), domainOfLeaf首二):
		ceiling = pileLast
		floor = pileOfLeaf首二 + 4
		step = 2
		for pileOfLeaf首零一二 in tuple(filter(between(floor, ceiling), domainOfLeaf首零一二))[0:None:step]:
			domainCombined.append((pileOfLeaf首二, pileOfLeaf首零一二 + 1, pileOfLeaf首零一二, pileOfLeaf首二 + 1))  # noqa: PERF401

	return tuple(filter(noDuplicates, domainCombined))

def getDomain二零and二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf二零 and leaf二 with clarified identifier semantics."""
	domainOfLeaf二零: tuple[int, ...] = tuple(getLeafDomain(state, 二+零))
	domainOfLeaf二: tuple[int, ...] = tuple(getLeafDomain(state, 二))
	return _getDomain二零and二(domainOfLeaf二零, domainOfLeaf二, state.dimensionsTotal)
@cache
def _getDomain二零and二(domainOfLeaf二零: tuple[int, ...], domainOfLeaf二: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	pilesTotal: int = len(domainOfLeaf二零)
	pilesFewerDomainOfLeaf二: int = pilesTotal - len(domainOfLeaf二)
	domainCombined: list[tuple[int, int]] = []

	domainCombined.extend([(pileOfLeaf二零, pileOfLeaf二零+零) for pileOfLeaf二零 in domainOfLeaf二零 if pileOfLeaf二零+零 in domainOfLeaf二])

	for index, pileOfLeaf二零 in enumerate(filter(between(pileOrigin, 首零(dimensionsTotal)-零), domainOfLeaf二零)):
		domainOfLeaf二Working: tuple[int, ...] = domainOfLeaf二

		listIndicesPilesExcluded: list[int] = []

		tailDimensions = howMany0coordinatesAtTail(pileOfLeaf二零)
		if (首二(dimensionsTotal) < pileOfLeaf二零 < 首零(dimensionsTotal)-零) and (3 < pileOfLeaf二零.bit_length()) and (tailDimensions == 1):
			if dimensionSecondNearest首(pileOfLeaf二零) == 1:
				if 4 < domainOfLeaf二Working[pilesTotal // 2 - pilesFewerDomainOfLeaf二].bit_length():
					listIndicesPilesExcluded.extend([pilesTotal // 2 - pilesFewerDomainOfLeaf二])
				if 首一(dimensionsTotal) < pileOfLeaf二零:
					listIndicesPilesExcluded.extend([-(pilesTotal // 4) + pilesFewerDomainOfLeaf二])
			if dimensionSecondNearest首(pileOfLeaf二零) == 2:
				if domainOfLeaf二Working[pilesTotal // 2 + 2 - pilesFewerDomainOfLeaf二] < 首零(dimensionsTotal):
					listIndicesPilesExcluded.extend([pilesTotal // 2 + 2 - pilesFewerDomainOfLeaf二])
				if 首一(dimensionsTotal) < pileOfLeaf二零:
					listIndicesPilesExcluded.extend([-(pilesTotal // 4 - 2) + pilesFewerDomainOfLeaf二])
			if dimensionSecondNearest首(pileOfLeaf二零) == 3:
				listIndicesPilesExcluded.extend([-(pilesTotal // 4) + pilesFewerDomainOfLeaf二, -(pilesTotal // 4 - 2) + pilesFewerDomainOfLeaf二])
			if pileOfLeaf二零 < 首一二(dimensionsTotal):
				listIndicesPilesExcluded.extend([3 * pilesTotal // 4 - pilesFewerDomainOfLeaf二])
			if bit_test(pileOfLeaf二零, 3):
				if pileOfLeaf二零.bit_count() == 3:
					listIndicesPilesExcluded.extend([3 * pilesTotal // 4 - 2 - pilesFewerDomainOfLeaf二])
				if pileOfLeaf二零.bit_count() == 4:
					listIndicesPilesExcluded.extend([3 * pilesTotal // 4 - pilesFewerDomainOfLeaf二])

		dimensionSize = int(bit_flip(0, tailDimensions))
		startExclude = abs(dimensionSize - 3)
		stepExclude = dimensionSize
		listIndicesPilesExcluded.extend(range(startExclude, pilesTotal, stepExclude))
		tailDimensions -= 1

		for dimension in range(tailDimensions + decreasing, decreasing, decreasing):
			startExclude -= int(bit_flip(0, dimension))
			stepExclude //= 2
			listIndicesPilesExcluded.extend(range(startExclude, pilesTotal, stepExclude))

		excludeBelow: int = index - pilesFewerDomainOfLeaf二
		if pileOfLeaf二零 == 二:
			excludeBelow = index + 1
		listIndicesPilesExcluded.extend(range(0, excludeBelow))  # noqa: PIE808
		if pileOfLeaf二零 <= 首一(dimensionsTotal):
			excludeAbove: int = index - pilesFewerDomainOfLeaf二 + (3 * pilesTotal // 4)
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
		if 首一(dimensionsTotal) < pileOfLeaf二零 < 首零(dimensionsTotal)-零:
			excludeAbove = int(pileOfLeaf二零 ^ bit_mask(dimensionsTotal)) + 2 + inclusive
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
		if 首一(dimensionsTotal) + 首三(dimensionsTotal) == pileOfLeaf二零:
			listIndicesPilesExcluded.extend([(3 * pilesTotal // 4) - 1 - pilesFewerDomainOfLeaf二])
		if 首一(dimensionsTotal) + 首二(dimensionsTotal) + 首三(dimensionsTotal) == pileOfLeaf二零:
			listIndicesPilesExcluded.extend([int(7 * pilesTotal / 8) - pilesFewerDomainOfLeaf二])

		domainOfLeaf二Working = tuple(exclude(domainOfLeaf二Working, listIndicesPilesExcluded))

		domainCombined.extend([(pileOfLeaf二零, pileOfLeaf二) for pileOfLeaf二 in domainOfLeaf二Working])

	return tuple(sorted(set(domainCombined)))

def getDomain二一零and二一(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain.

	This combined domain has the same basic pattern as `getDomain二零and二`, with the parity switched.

	Interestingly, the 22 pairs of `leaf二一, leaf二一零` in consecutive piles cover 6241 of 7840 foldsTotal for (2,) * 6 maps.
	The combined domain is very small, only 76 pairs, but 22 pairs cover 80% and the other 54 pairs only cover 20%. Furthermore,
	in the 22 pairs, `leaf二一零` follows `leaf二一`, but in the rest of the domain, `leaf二一` always follows `leaf二一零`.
	"""
	domainOfLeaf二一零: tuple[int, ...] = tuple(getLeafDomain(state, 二+一+零))
	domainOfLeaf二一: tuple[int, ...] = tuple(getLeafDomain(state, 二+一))
	return _getDomain二一零and二一(domainOfLeaf二一零, domainOfLeaf二一, state.dimensionsTotal)
@cache
def _getDomain二一零and二一(domain二一零: tuple[int, ...], domain二一: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	pilesTotal: int = len(domain二一零)
	domainCombined: list[tuple[int, int]] = []

	# NOTE Include corners.
	domainCombined.extend([(pileOfLeaf二一零, pileOfLeaf二一零-零) for pileOfLeaf二一零 in domain二一零 if pileOfLeaf二一零-零 in domain二一])

	for index, pileOfLeaf二一零 in enumerate(filter(between(pileOrigin, 首零(dimensionsTotal)-零), domain二一零)):
		domainOfLeaf二一: tuple[int, ...] = domain二一
		listIndicesPilesExcluded: list[int] = []

		tailDimensions: int = howMany0coordinatesAtTail(pileOfLeaf二一零 - 零)

		# NOTE Exclude powers of 2.
		for dimension in range(tailDimensions):
			listIndicesPilesExcluded.extend(range(index + int(bit_mask(dimension)), pilesTotal, int(bit_flip(0, dimension + 1))))

		excludeBelow: int = index + (is_even(index) or tailDimensions)
		listIndicesPilesExcluded.extend(range(excludeBelow))

		if pileOfLeaf二一零 <= 首一(dimensionsTotal):
			excludeAbove: int = index + (3 * pilesTotal // 4)
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
		if 首一(dimensionsTotal) < pileOfLeaf二一零 < 首零(dimensionsTotal):
			excludeAbove = int(pileOfLeaf二一零 ^ bit_mask(dimensionsTotal)) // 2
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

		if 首一(dimensionsTotal) + 首三(dimensionsTotal) + 1 == pileOfLeaf二一零:
			listIndicesPilesExcluded.extend([(3 * pilesTotal // 4) - 1])

		if (首二(dimensionsTotal) < pileOfLeaf二一零 < 首零(dimensionsTotal)-零) and (3 < pileOfLeaf二一零.bit_length()) and (tailDimensions == 1):
			if dimensionSecondNearest首(pileOfLeaf二一零) == 1:
				if 4 < domainOfLeaf二一[pilesTotal // 2].bit_length():
					listIndicesPilesExcluded.extend([pilesTotal // 2])
				if 首一(dimensionsTotal) < pileOfLeaf二一零:
					listIndicesPilesExcluded.extend([-(pilesTotal // 4 - 1)])
			if dimensionSecondNearest首(pileOfLeaf二一零) == 2:
				if domainOfLeaf二一[pilesTotal // 2 + 2] < 首零(dimensionsTotal):
					listIndicesPilesExcluded.extend([pilesTotal // 2 + 2])
				if 首一(dimensionsTotal) < pileOfLeaf二一零:
					listIndicesPilesExcluded.extend([-(pilesTotal // 4 - 2)])
			if dimensionSecondNearest首(pileOfLeaf二一零) == 3:
				listIndicesPilesExcluded.extend([-(pilesTotal // 4), -(pilesTotal // 4 - 2)])
			if pileOfLeaf二一零 < 首一二(dimensionsTotal):
				listIndicesPilesExcluded.extend([3 * pilesTotal // 4])
			if bit_test(pileOfLeaf二一零, 3):
				if pileOfLeaf二一零.bit_count() == 3:
					listIndicesPilesExcluded.extend([3 * pilesTotal // 4])
				if pileOfLeaf二一零.bit_count() == 4:
					listIndicesPilesExcluded.extend([3 * pilesTotal // 4 - 2])

		domainOfLeaf二一 = tuple(exclude(domainOfLeaf二一, listIndicesPilesExcluded))
		domainCombined.extend([(pileOfLeaf二一零, pileOfLeaf二一) for pileOfLeaf二一 in domainOfLeaf二一])

	return tuple(domainCombined)

def getDictionaryLeafDomains(state: EliminationState) -> dict[int, range]:
	"""For each `leaf`, the associated Python `range` defines the mathematical domain:
	1. every `pile` at which `leaf` may be found in a `folding` and
	2. in the set of all valid foldings, every `pile` at which `leaf` must be found.
	"""  # noqa: D205
	return {leaf: getLeafDomain(state, leaf) for leaf in range(state.leavesTotal)}

