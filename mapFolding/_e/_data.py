from collections.abc import Callable, Iterable, Sequence
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache
from gmpy2 import bit_flip, bit_mask, bit_test, is_even
from itertools import product as CartesianProduct
from mapFolding import decreasing, inclusive
from mapFolding._e import (
	dimensionNearest首, howMany0coordinatesAtTail, howManyDimensionsHaveOddParity, leafOrigin, pileOrigin, 一, 三, 二, 零, 首一,
	首一二, 首二, 首零, 首零二)
from mapFolding.dataBaskets import EliminationState
from math import prod

# ======= Boolean filters =================================

@syntacticCurry
def between(floor: int, ceiling: int, pile: int) -> bool:
	return floor <= pile <= ceiling

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

def noDuplicates(sequenceHashable: Sequence[int]) -> bool:
	return len(sequenceHashable) == len(set(sequenceHashable))

# ======= Creases and addends =================================

def getListLeavesIncrease(state: EliminationState, leaf: int) -> list[int]:
	return _getCreases(state, leaf, increase=True)

def getListLeavesDecrease(state: EliminationState, leaf: int) -> list[int]:
	return _getCreases(state, leaf, increase=False)

def _getCreases(state: EliminationState, leaf: int, *, increase: bool = True) -> list[int]:
	@cache
	def workhorse(leaf: int, dimensionsTotal: int) -> tuple[list[int], list[int]]:
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

	(listLeavesIncrease, listLeavesDecrease) = workhorse(leaf, state.dimensionsTotal)

	return listLeavesIncrease if increase else listLeavesDecrease

def getDictionaryAddends4Next(state: EliminationState) -> dict[int, list[int]]:
	@cache
	def workhorse(mapShape: tuple[int, ...], dimensionsTotal: int, leavesTotal: int) -> dict[int, list[int]]:
		dictionaryAddends: dict[int, list[int]] = {leafOrigin: [1]}

		productsOfDimensions: list[int] = [prod(mapShape[0:dimension], start=1) for dimension in range(dimensionsTotal)]

		for leaf in range(零, leavesTotal):
			products下_leaf: list[int] = productsOfDimensions.copy()

			theMaskOfDirectionality = bit_mask(leavesTotal - 零) & leaf
			for index in range(dimensionsTotal):
				if bit_test(theMaskOfDirectionality, index):
					products下_leaf[index] *= -1

			slicingIndexStart: int = (leaf.bit_count() - 1) & 1 ^ 1
			slicingIndexEnd = dimensionNearest首(leaf) * (slicingIndexStart ^ 1) or None

			if (slicingIndexStart == 1) and is_even(leaf):
				slicingIndexStart += howMany0coordinatesAtTail(leaf)

			products下_leaf = products下_leaf[slicingIndexStart:None]
			products下_leaf = products下_leaf[0:slicingIndexEnd]
			dictionaryAddends[leaf] = products下_leaf

		return dictionaryAddends
	return workhorse(state.mapShape, state.dimensionsTotal, state.leavesTotal)

def getDictionaryAddends4Prior(state: EliminationState) -> dict[int, list[int]]:
	@cache
	def workhorse(mapShape: tuple[int, ...], dimensionsTotal: int, leavesTotal: int) -> dict[int, list[int]]:
		dictionaryAddends: dict[int, list[int]] = {leafOrigin: [], 零: [-1]}

		productsOfDimensions: list[int] = [prod(mapShape[0:dimension], start=1) for dimension in range(dimensionsTotal)]

		for leaf in range(leavesTotal + decreasing, 1, decreasing):
			products下_leaf: list[int] = productsOfDimensions.copy()

			theMaskOfDirectionality = bit_mask(leavesTotal - 零) & leaf
			for index in range(dimensionsTotal):
				if bit_test(theMaskOfDirectionality, index):
					products下_leaf[index] *= -1

			slicingIndexStart: int = (leaf.bit_count() - 1) & 1
			slicingIndexEnd = dimensionNearest首(leaf) * (slicingIndexStart ^ 1) or None

			if (slicingIndexStart == 1) and is_even(leaf):
				slicingIndexStart += howMany0coordinatesAtTail(leaf)

			products下_leaf = products下_leaf[slicingIndexStart:None]
			products下_leaf = products下_leaf[0:slicingIndexEnd]
			dictionaryAddends[leaf] = products下_leaf

		return dictionaryAddends
	return workhorse(state.mapShape, state.dimensionsTotal, state.leavesTotal)

# ======= (mathematical) ranges of piles ====================

def getPileRange(state: EliminationState, pile: int) -> Iterable[int]:
	@cache
	def workhorse(pile: int, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> Iterable[int]:
		if (dimensionsTotal > 3) and all(dimensionLength == 2 for dimensionLength in mapShape):
			parityMatch: Callable[[int], bool] = filterParity(pile)
			pileAboveFloor: Callable[[int], bool] = filterFloor(pile)
			pileBelowCeiling: Callable[[int], bool] = filterCeiling(pile, dimensionsTotal)
			matchLargerStep: Callable[[int], bool] = filterDoubleParity(pile, dimensionsTotal)

			pileRange = range(leavesTotal)
			pileRange = filter(parityMatch, pileRange)
			pileRange = filter(pileAboveFloor, pileRange)
			pileRange = filter(pileBelowCeiling, pileRange)
			return filter(matchLargerStep, pileRange)

		else:
			return range(leavesTotal)
	return workhorse(pile, state.dimensionsTotal, state.mapShape, state.leavesTotal)

def getDictionaryPileRanges(state: EliminationState) -> dict[int, list[int]]:
	"""At `pile`, which `leaf` values may be found in a `folding`: the mathematical range, not a Python `range` object."""
	@cache
	def workhorse(dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> dict[int, list[int]]:
		dictionaryPileRanges: dict[int, list[int]] = {pile: [] for pile in range(leavesTotal)}

		for pile, leaf in CartesianProduct(range(leavesTotal), range(leavesTotal)):
			rangeStart: int = leaf.bit_count() + (2**(howMany0coordinatesAtTail(leaf) + 1) - 2)

			binary: str = ('1' * dimensionNearest首(leaf)).ljust(dimensionsTotal, '0')
			rangeStop: int = int(binary, mapShape[0]) + 2 - (leaf.bit_count() - 1) - (leaf == leafOrigin)

			specialCase: bool = (leaf == 首零(dimensionsTotal)+零)
			rangeStep: int = 2 + (2 * specialCase)

			if rangeStart <= pile < rangeStop and (pile - rangeStart) % rangeStep == 0:
				dictionaryPileRanges[pile].append(leaf)

		return dictionaryPileRanges
	return workhorse(state.dimensionsTotal, state.mapShape, state.leavesTotal)

# ======= Leaf domains ====================================

def getLeafDomain(state: EliminationState, leaf: int) -> range:
	@cache
	def workhorse(leaf: int, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> range:
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
		else:
			return range(leavesTotal)
	return workhorse(leaf, state.dimensionsTotal, state.mapShape, state.leavesTotal)

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

def getDomain二零and二corners(state: EliminationState) -> tuple[tuple[int, int], ...]:
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
	return _getDomain二零and二(domain二零, domain二, state.dimensionsTotal, state.pileLast, nonCorners=False)

def getDomain二零and二nonCorners(state: EliminationState) -> tuple[tuple[int, int], ...]:
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
	return _getDomain二零and二(domain二零, domain二, state.dimensionsTotal, state.pileLast, corners=False)

@cache
def _getDomain二零and二(domain二零: tuple[int, ...], domain二: tuple[int, ...], dimensionsTotal: int, pileLast: int, *, corners: bool = True, nonCorners: bool = True) -> tuple[tuple[int, int], ...]:
	domain二零and二: list[tuple[int, int]] = []

	if corners:
		domain二零and二.extend([(pile二零, pile二零+零) for pile二零 in filter(between(首零(dimensionsTotal), pileLast), domain二零)])

	if nonCorners:
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

def getDomain二一零and二一corners(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain for corners only (consecutive piles).

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
	return _getDomain二一零and二一(domain二一零, domain二一, state.dimensionsTotal, state.pileLast, nonCorners=False)

def getDomain二一零and二一nonCorners(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain for non-corners only (non-consecutive piles).

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
	return _getDomain二一零and二一(domain二一零, domain二一, state.dimensionsTotal, state.pileLast, corners=False)

@cache
def _getDomain二一零and二一(domain二一零: tuple[int, ...], domain二一: tuple[int, ...], dimensionsTotal: int, pileLast: int, *, corners: bool = True, nonCorners: bool = True) -> tuple[tuple[int, int], ...]:
	domain二一零and二一: list[tuple[int, int]] = []

	if corners:
		domain二一零and二一.extend([(pile二一 + 零, pile二一) for pile二一 in domain二一 if pile二一 + 零 in domain二一零])

	if nonCorners:
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

def getDomain二combined(state: EliminationState) -> tuple[tuple[int, ...], ...]:
	domain二零: tuple[int, ...] = tuple(getLeafDomain(state, 二+零))
	domain二: tuple[int, ...] = tuple(getLeafDomain(state, 二))
	domain二一: tuple[int, ...] = tuple(getLeafDomain(state, 二+一))
	return _getDomain二combined(domain二零, domain二, domain二一, state.dimensionsTotal, state.pileLast)

@cache
def _getDomain二combined(domain二零: tuple[int, ...], domain二: tuple[int, ...], domain二一: tuple[int, ...], dimensionsTotal: int, pileLast: int) -> tuple[tuple[int, ...], ...]:
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

		domain二combined = list(filter(noDuplicates, domain二combined))

	return tuple(domain二combined)

def getDictionaryLeafDomains(state: EliminationState) -> dict[int, range]:
	"""For each `leaf` (not `leaf`), the associated `range` defines
	1. every `pile` at which `leaf` may be found in a `folding` and
	2. in the set of all valid foldings, every `pile` at which `leaf` must be found.
	"""  # noqa: D205
	return {leaf: getLeafDomain(state, leaf) for leaf in range(state.leavesTotal)}

