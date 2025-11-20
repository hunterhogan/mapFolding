from functools import cache
from itertools import product as CartesianProduct
from mapFolding._e import coordinatesOf0AtTail, decreasing, indexLeaf0, 零, 首零
from mapFolding.dataBaskets import EliminationState
from math import prod
import gmpy2

def getDictionaryAddends4Next(state: EliminationState) -> dict[int, list[int]]:
	@cache
	def workhorse(mapShape: tuple[int, ...], dimensionsTotal: int, leavesTotal: int) -> dict[int, list[int]]:
		dictionaryAddends: dict[int, list[int]] = {indexLeaf0: [1]}

		productsOfDimensions: list[int] = [prod(mapShape[0:dimension], start=1) for dimension in range(dimensionsTotal)]

		for indexLeaf in range(零, leavesTotal):
			products下_indexLeaf: list[int] = productsOfDimensions.copy()

			theMaskOfDirectionality = gmpy2.bit_mask(leavesTotal - 零) & indexLeaf
			for index in range(dimensionsTotal):
				if gmpy2.bit_test(theMaskOfDirectionality, index):
					products下_indexLeaf[index] *= -1

			slicingIndexStart: int = (indexLeaf.bit_count() - 1) & 1 ^ 1
			slicingIndexEnd = (indexLeaf.bit_length() - 1) * (slicingIndexStart ^ 1) or None

			if (slicingIndexStart == 1) and (gmpy2.is_even(indexLeaf)):
				slicingIndexStart += coordinatesOf0AtTail(indexLeaf)

			products下_indexLeaf = products下_indexLeaf[slicingIndexStart:None]
			products下_indexLeaf = products下_indexLeaf[0:slicingIndexEnd]
			dictionaryAddends[indexLeaf] = products下_indexLeaf

		return dictionaryAddends
	return workhorse(state.mapShape, state.dimensionsTotal, state.leavesTotal)

def getDictionaryAddends4Prior(state: EliminationState) -> dict[int, list[int]]:
	@cache
	def workhorse(mapShape: tuple[int, ...], dimensionsTotal: int, leavesTotal: int) -> dict[int, list[int]]:
		dictionaryAddends: dict[int, list[int]] = {indexLeaf0: [], 零: [-1]}

		productsOfDimensions: list[int] = [prod(mapShape[0:dimension], start=1) for dimension in range(dimensionsTotal)]

		for indexLeaf in range(leavesTotal + decreasing, 1, decreasing):
			products下_indexLeaf: list[int] = productsOfDimensions.copy()

			theMaskOfDirectionality = gmpy2.bit_mask(leavesTotal - 零) & indexLeaf
			for index in range(dimensionsTotal):
				if gmpy2.bit_test(theMaskOfDirectionality, index):
					products下_indexLeaf[index] *= -1

			slicingIndexStart: int = (indexLeaf.bit_count() - 1) & 1
			slicingIndexEnd = (indexLeaf.bit_length() - 1) * (slicingIndexStart ^ 1) or None

			if (slicingIndexStart == 1) and (gmpy2.is_even(indexLeaf)):
				slicingIndexStart += coordinatesOf0AtTail(indexLeaf)

			products下_indexLeaf = products下_indexLeaf[slicingIndexStart:None]
			products下_indexLeaf = products下_indexLeaf[0:slicingIndexEnd]
			dictionaryAddends[indexLeaf] = products下_indexLeaf

		return dictionaryAddends
	return workhorse(state.mapShape, state.dimensionsTotal, state.leavesTotal)

def getDictionaryIndexLeafDomains(state: EliminationState) -> dict[int, range]:
	"""For each `indexLeaf` (not `leaf`), the associated `range` defines
	1. every `pile` at which `indexLeaf` may be found in a `folding` and
	2. in the set of all valid foldings, every `pile` at which `indexLeaf` must be found.
	"""  # noqa: D205
	return {indexLeaf: getIndexLeafDomain(state, indexLeaf) for indexLeaf in range(state.leavesTotal)}

def getDictionaryPileToIndexLeaves(state: EliminationState) -> dict[int, list[int]]:
	"""At `pile`, which `indexLeaf` values may be found in a `folding`."""
	dictionaryPileToLeaves: dict[int, list[int]] = {pile: [] for pile in range(state.leavesTotal)}

# TODO create the per-pile function analogous to getIndexLeafDomain().

	for pile, indexLeaf in CartesianProduct(range(state.leavesTotal), range(state.leavesTotal)):
		rangeStart: int = indexLeaf.bit_count() + (2**(coordinatesOf0AtTail(indexLeaf) + 1) - 2)

		binary: str = ('1' * (indexLeaf.bit_length() - 1)).ljust(state.dimensionsTotal, '0')
		rangeStop: int = int(binary, state.mapShape[0]) + 2 - (indexLeaf.bit_count() - 1) - (indexLeaf == indexLeaf0)

		specialCase: bool = (indexLeaf == 首零(state.dimensionsTotal) + 1)
		rangeStep: int = 2 + (2 * specialCase)

		if rangeStart <= pile < rangeStop and (pile - rangeStart) % rangeStep == 0:
			dictionaryPileToLeaves[pile].append(indexLeaf)

	return dictionaryPileToLeaves

def getIndexLeafDomain(state: EliminationState, indexLeaf: int) -> range:

	@cache
	def workhorse(indexLeaf: int, dimensionsTotal: int, dimensionLength: int) -> range:
		return range(indexLeaf.bit_count() + (2**(coordinatesOf0AtTail(indexLeaf) + 1) - 2)
					, int(('1' * (indexLeaf.bit_length()-1)).ljust(dimensionsTotal, '0'), dimensionLength) + 2 - (indexLeaf.bit_count() - 1) - (indexLeaf == 0)
									, 2 + (2 * (indexLeaf == 首零(dimensionsTotal) + 1)))
	return workhorse(indexLeaf, state.dimensionsTotal, state.mapShape[0])

if __name__ == '__main__':
	from mapFolding.tests.verify import (  # pyright: ignore[reportUnusedImport]
		verifyDictionaryAddends4Next, verifyDictionaryAddends4Prior, verifyDictionaryLeafRanges, verifyDictionaryPileToLeaves)

	state = EliminationState((2,) * 5)
	dictionaryLeafRanges = getDictionaryIndexLeafDomains(state)
	# verifyDictionaryLeafRanges(state, dictionaryLeafRanges)  # noqa: ERA001
	dictionaryAddends4Next = getDictionaryAddends4Next(state)
	# verifyDictionaryAddends4Next(state, dictionaryAddends4Next)  # noqa: ERA001
	dictionaryAddends4Prior = getDictionaryAddends4Prior(state)
	# verifyDictionaryAddends4Prior(state, dictionaryAddends4Prior)  # noqa: ERA001
	dictionaryPileToLeaves = getDictionaryPileToIndexLeaves(state)
	# verifyDictionaryPileToLeaves(state, dictionaryPileToLeaves)  # noqa: ERA001
