from collections.abc import Callable, Iterable
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache
from gmpy2 import bit_flip, bit_mask, is_even
from mapFolding import between, exclude, inclusive, noDuplicates
from mapFolding._e import (
	dimensionNearest首, howMany0coordinatesAtTail, howManyDimensionsHaveOddParity, leafOrigin, pileOrigin, 一, 三, 二, 零, 首一,
	首一二, 首二, 首零, 首零一二, 首零二)
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
	domain一零: tuple[int, ...] = tuple(getLeafDomain(state, 一+零))
	domain首一: tuple[int, ...] = tuple(getLeafDomain(state, 首一(state.dimensionsTotal)))
	return _getDomainDimension一(domain一零, domain首一, state.dimensionsTotal)
@cache
def _getDomainDimension一(domain一零: tuple[int, ...], domain首一: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int, int, int], ...]:
	domainCombined: list[tuple[int, int, int, int]] = []

	for pileOfLeaf一零 in domain一零:
		domain首: tuple[int, ...] = domain首一
		pilesTotal: int = len(domain首)

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

		domain首 = tuple(exclude(domain首, listIndicesPilesExcluded))

		domainCombined.extend([(pileOfLeaf一零, pileOfLeaf一零 + 1, pileOfLeaf首一, pileOfLeaf首一 + 1) for pileOfLeaf首一 in domain首])

	return tuple(filter(noDuplicates, domainCombined))

def getDomainDimension二(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
	domain二零: tuple[int, ...] = tuple(getLeafDomain(state, 二+零))
	domain二: tuple[int, ...] = tuple(getLeafDomain(state, 二))
	domain二一: tuple[int, ...] = tuple(getLeafDomain(state, 二+一))
	return _getDomainDimension二(domain二零, domain二, domain二一, state.dimensionsTotal, state.pileLast)
@cache
def _getDomainDimension二(domain二零: tuple[int, ...], domain二: tuple[int, ...], domain二一: tuple[int, ...], dimensionsTotal: int, pileLast: int) -> tuple[tuple[int, int, int, int], ...]:
	"""Domain is over-inclusive."""
	domain二combined: list[tuple[int, int, int, int]] = []

	domain零: tuple[int, ...] = domain二零
	for pile二一 in filter(between(首一二(dimensionsTotal), pileLast), domain二一):
		for pile二零 in filter(between(pile二一 + 2, pileLast), domain零):
			domain二combined.append((pile二一, pile二一 + 1, pile二零, pile二零 + 1))  # noqa: PERF401

	for pile二一 in filter(between(0, 首一二(dimensionsTotal) - 1), domain二一):
		floor: int = pile二一 + 2
		ceiling: int = pileLast
		step = 1

		domain零 = domain二零

		if pile二一 <= 首一(dimensionsTotal):
			ceiling = 首零二(dimensionsTotal) + 零

		if pile二一 <= 首二(dimensionsTotal):
			ceiling = int(bit_mask(dimensionsTotal - 1) ^ bit_mask(dimensionsTotal - 1 - dimensionNearest首(pile二一))) - howManyDimensionsHaveOddParity(pile二一) + 2

		if pile二一 == 二:
			ceiling = pile二一 + 一

		domain零 = tuple(filter(between(floor, ceiling), domain零))[0:None:step]

		# no pile 42
		domain二combined.extend([(pile二一, pile二一 + 1, pile二零, pile二零 + 1) for pile二零 in domain零])

	domain零 = domain二零
	for pile二零 in filter(between(pileOrigin, 首零(dimensionsTotal)-零), domain零):
		domain0: tuple[int, ...] = domain二
		floor: int = pile二零 + 零
		if pile二零 == 二:
			floor = 三+二+零
		ceiling = pileLast
		step = int(bit_flip(0, howMany0coordinatesAtTail(pile二零)))
		if pile二零 <= 首一(dimensionsTotal):
			ceiling: int = pile二零 + 首零(dimensionsTotal) + 零
		if 首一(dimensionsTotal) < pile二零 < 首零(dimensionsTotal)-零:
			ceiling = int(pile二零 ^ bit_mask(dimensionsTotal)) + 2 + inclusive

		domain0 = tuple(filter(between(floor, ceiling), domain0))[0:None:step]

		if (首一(dimensionsTotal) < pile二零 < 首零(dimensionsTotal)-零) and (howMany0coordinatesAtTail(pile二零) == 1):
			sherpa = list(domain0)
			if 9 + pile二零 in domain0:
				sherpa.remove(9 + pile二零)
			if 25 + pile二零 in domain0:
				sherpa.remove(25 + pile二零)

			domain0 = tuple(sherpa)

		domain二combined.extend([(pile二 - 1, pile二零 + 1, pile二零, pile二) for pile二 in domain0])

	return tuple(filter(noDuplicates, domain二combined))

def getDomainDimension首二(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
	domain首二: tuple[int, ...] = tuple(getLeafDomain(state, 首二(state.dimensionsTotal)))
	domain首零一二: tuple[int, ...] = tuple(getLeafDomain(state, 首零一二(state.dimensionsTotal)))
	return _getDomainDimension首二(domain首二, domain首零一二, state.dimensionsTotal, state.leavesTotal, state.pileLast)
@cache
def _getDomainDimension首二(domain首二: tuple[int, ...], domain首零一二: tuple[int, ...], dimensionsTotal: int, leavesTotal: int, pileLast: int) -> tuple[tuple[int, int, int, int], ...]:
	domainCombined: list[tuple[int, int, int, int]] = []

	# 46 surplus tuples
	for pile首二 in domain首二:
		ceiling = pile首二 + 首零(dimensionsTotal)

		if pile首二 < 首一二(dimensionsTotal) - 1:
			ceiling = pile首二 ^ bit_mask(dimensionsTotal)

		if (首一二(dimensionsTotal) - 1 <= pile首二) and (howMany0coordinatesAtTail(pile首二 + 1) > 1):
			ceiling = leavesTotal - 二 - 一 - 零

		if (首零(dimensionsTotal) + 2 < pile首二) and (howMany0coordinatesAtTail(pile首二 + 1) == 1):
			ceiling = leavesTotal - (leavesTotal - pile首二) // 2

		for pile首零一二 in filter(between(pile首二 + 2, ceiling), domain首零一二):
			domainCombined.append((pile首二, pile首二 + 1, pile首零一二, pile首零一二 + 1))  # noqa: PERF401

	# 22 surplus tuples
	for pile首二 in filter(between(pileOrigin, 首零(dimensionsTotal)), domain首二):
		ceiling = pile首二 + 首零(dimensionsTotal) + inclusive
		floor = leavesTotal - 2 - pile首二
		step = 2
		for pile首零一二 in tuple(filter(between(floor, ceiling), domain首零一二))[0:None:step]:
			domainCombined.append((pile首二, pile首零一二 + 1, pile首零一二, pile首二 + 1))  # noqa: PERF401

	# 18 surplus tuples
	for pile首二 in filter(between(首零(dimensionsTotal), pileLast), domain首二):
		ceiling = pileLast
		floor = pile首二 + 4
		step = 2
		for pile首零一二 in tuple(filter(between(floor, ceiling), domain首零一二))[0:None:step]:
			domainCombined.append((pile首二, pile首零一二 + 1, pile首零一二, pile首二 + 1))  # noqa: PERF401

	return tuple(filter(noDuplicates, domainCombined))

def getDomain二零and二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain.

	The domain (nonCorners) is over-inclusive at the following:
	pile二零 pile二
	001010  011011
	001010  100111
	001110  011111
	001110  100011
	001110  100111
	010010  011011
	010010  100111
	010010  101011
	010100  100101
	010110  011111
	010110  100111
	011010  100011
	"""
	domain二零: tuple[int, ...] = tuple(getLeafDomain(state, 二+零))
	domain二: tuple[int, ...] = tuple(getLeafDomain(state, 二))
	return _getDomain二零and二(domain二零, domain二, state.dimensionsTotal, state.pileLast)
@cache
def _getDomain二零and二(domain二零: tuple[int, ...], domain二: tuple[int, ...], dimensionsTotal: int, pileLast: int) -> tuple[tuple[int, int], ...]:
	domain二零and二: list[tuple[int, int]] = []

	domain二零and二.extend([(pile二零, pile二零+零) for pile二零 in filter(between(首零(dimensionsTotal), pileLast), domain二零)])

	for pile二零 in filter(between(pileOrigin, 首零(dimensionsTotal)-零), domain二零):
		domain0: tuple[int, ...] = domain二
		floor: int = pile二零 + 零
		if pile二零 == 二:
			floor = 三+二+零
		ceiling = pileLast
		step = int(bit_flip(0, howMany0coordinatesAtTail(pile二零)))
		if pile二零 <= 首一(dimensionsTotal):
			ceiling: int = pile二零 + 首零(dimensionsTotal) + 零
		if 首一(dimensionsTotal) < pile二零 < 首零(dimensionsTotal)-零:
			ceiling = int(pile二零 ^ bit_mask(dimensionsTotal)) + 2 + inclusive

		domain0 = tuple(filter(between(floor, ceiling), domain0))[0:None:step]

		domain二零and二.extend([(pile二零, pile二) for pile二 in domain0])

	return tuple(domain二零and二)

def getDomain二一零and二一(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain.

	This combined domain has the same basic pattern as `getDomain二零and二`, with the parity switched. Similarly, there is at
	least one pattern I haven't figured out yet.

	Interestingly, the 22 pairs of `leaf二一, leaf二一零` in consecutive piles cover 6241 of 7840 foldsTotal for (2,) * 6 maps.
	The combined domain is very small, only 76 pairs, but 22 pairs cover 80% and the other 54 pairs only cover 20%. Furthermore,
	in the 22 pairs, `leaf二一零` follows `leaf二一`, but in the rest of the domain, `leaf二一` always follows `leaf二一零`.

	The domain is over-inclusive at the following:
	pile二一零 pile二一
	001011  011010
	001011  100110
	001111  011110
	001111  100010
	001111  100110
	010011  011010
	010011  100110
	010011  101010
	010101  100100
	010111  011110
	010111  100110
	011011  100010
	011111  100110
	011111  101010
	011111  101110
	"""
	domain二一零: tuple[int, ...] = tuple(getLeafDomain(state, 二+一+零))
	domain二一: tuple[int, ...] = tuple(getLeafDomain(state, 二+一))
	return _getDomain二一零and二一(domain二一零, domain二一, state.dimensionsTotal, state.pileLast)
@cache
def _getDomain二一零and二一(domain二一零: tuple[int, ...], domain二一: tuple[int, ...], dimensionsTotal: int, pileLast: int) -> tuple[tuple[int, int], ...]:
	domain二一零and二一: list[tuple[int, int]] = []

	domain二一零and二一.extend([(pile二一 + 零, pile二一) for pile二一 in domain二一 if pile二一 + 零 in domain二一零])

	for pile二一零 in filter(between(pileOrigin, 首零(dimensionsTotal)-零), domain二一零):
		domain二一Filtered: tuple[int, ...] = domain二一
		step = int(bit_flip(0, howMany0coordinatesAtTail(pile二一零 - 零)))
		floor: int = pile二一零 - 零 + step * 2
		ceiling: int = pileLast
		if pile二一零 <= 首一(dimensionsTotal):
			ceiling = pile二一零 + 首零(dimensionsTotal) - 零 + inclusive
		if 首一(dimensionsTotal) < pile二一零 < 首零(dimensionsTotal)-零:
			ceiling = int(pile二一零 ^ bit_mask(dimensionsTotal)) + 2 + inclusive

		domain二一Filtered = tuple(filter(between(floor, ceiling), domain二一Filtered))[0:None:step]

		domain二一零and二一.extend([(pile二一零, pile二一) for pile二一 in domain二一Filtered])

	return tuple(domain二一零and二一)

def getDictionaryLeafDomains(state: EliminationState) -> dict[int, range]:
	"""For each `leaf`, the associated Python `range` defines the mathematical domain:
	1. every `pile` at which `leaf` may be found in a `folding` and
	2. in the set of all valid foldings, every `pile` at which `leaf` must be found.
	"""  # noqa: D205
	return {leaf: getLeafDomain(state, leaf) for leaf in range(state.leavesTotal)}

