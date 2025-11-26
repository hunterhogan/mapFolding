from collections.abc import Callable, Iterable
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache
from gmpy2 import bit_flip, bit_mask, bit_test, is_even
from itertools import product as CartesianProduct
from mapFolding import decreasing
from mapFolding._e import (
	dimensionNearest首, howMany0coordinatesAtTail, howManyDimensionsHaveOddParity, leafOrigin, 零, 首零)
from mapFolding.dataBaskets import EliminationState
from math import prod

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

def getDictionaryLeafDomains(state: EliminationState) -> dict[int, range]:
	"""For each `leaf` (not `leaf`), the associated `range` defines
	1. every `pile` at which `leaf` may be found in a `folding` and
	2. in the set of all valid foldings, every `pile` at which `leaf` must be found.
	"""  # noqa: D205
	return {leaf: getLeafDomain(state, leaf) for leaf in range(state.leavesTotal)}

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
