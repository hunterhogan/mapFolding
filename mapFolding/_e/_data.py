from functools import cache
from itertools import product as CartesianProduct
from mapFolding._e import coordinatesOf0AtTail, decreasing, leaf0, 零, 首零
from mapFolding.dataBaskets import EliminationState
from math import prod
import gmpy2

def getDictionaryAddends4Next(state: EliminationState) -> dict[int, list[int]]:
	@cache
	def workhorse(mapShape: tuple[int, ...], dimensionsTotal: int, leavesTotal: int) -> dict[int, list[int]]:
		dictionaryAddends: dict[int, list[int]] = {leaf0: [1]}

		productsOfDimensions: list[int] = [prod(mapShape[0:dimension], start=1) for dimension in range(dimensionsTotal)]

		for leaf in range(零, leavesTotal):
			products下_leaf: list[int] = productsOfDimensions.copy()

			theMaskOfDirectionality = gmpy2.bit_mask(leavesTotal - 零) & leaf
			for index in range(dimensionsTotal):
				if gmpy2.bit_test(theMaskOfDirectionality, index):
					products下_leaf[index] *= -1

			slicingIndexStart: int = (leaf.bit_count() - 1) & 1 ^ 1
			slicingIndexEnd = (leaf.bit_length() - 1) * (slicingIndexStart ^ 1) or None

			if (slicingIndexStart == 1) and (gmpy2.is_even(leaf)):
				slicingIndexStart += coordinatesOf0AtTail(leaf)

			products下_leaf = products下_leaf[slicingIndexStart:None]
			products下_leaf = products下_leaf[0:slicingIndexEnd]
			dictionaryAddends[leaf] = products下_leaf

		return dictionaryAddends
	return workhorse(state.mapShape, state.dimensionsTotal, state.leavesTotal)

def getDictionaryAddends4Prior(state: EliminationState) -> dict[int, list[int]]:
	@cache
	def workhorse(mapShape: tuple[int, ...], dimensionsTotal: int, leavesTotal: int) -> dict[int, list[int]]:
		dictionaryAddends: dict[int, list[int]] = {leaf0: [], 零: [-1]}

		productsOfDimensions: list[int] = [prod(mapShape[0:dimension], start=1) for dimension in range(dimensionsTotal)]

		for leaf in range(leavesTotal + decreasing, 1, decreasing):
			products下_leaf: list[int] = productsOfDimensions.copy()

			theMaskOfDirectionality = gmpy2.bit_mask(leavesTotal - 零) & leaf
			for index in range(dimensionsTotal):
				if gmpy2.bit_test(theMaskOfDirectionality, index):
					products下_leaf[index] *= -1

			slicingIndexStart: int = (leaf.bit_count() - 1) & 1
			slicingIndexEnd = (leaf.bit_length() - 1) * (slicingIndexStart ^ 1) or None

			if (slicingIndexStart == 1) and (gmpy2.is_even(leaf)):
				slicingIndexStart += coordinatesOf0AtTail(leaf)

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

def getDictionaryPileToLeaves(state: EliminationState) -> dict[int, list[int]]:
	"""At `pile`, which `leaf` values may be found in a `folding`."""
	dictionaryPileToLeaves: dict[int, list[int]] = {pile: [] for pile in range(state.leavesTotal)}

# TODO create the per-pile function analogous to getLeafDomain().

	for pile, leaf in CartesianProduct(range(state.leavesTotal), range(state.leavesTotal)):
		rangeStart: int = leaf.bit_count() + (2**(coordinatesOf0AtTail(leaf) + 1) - 2)

		binary: str = ('1' * (leaf.bit_length() - 1)).ljust(state.dimensionsTotal, '0')
		rangeStop: int = int(binary, state.mapShape[0]) + 2 - (leaf.bit_count() - 1) - (leaf == leaf0)

		specialCase: bool = (leaf == 首零(state.dimensionsTotal)+零)
		rangeStep: int = 2 + (2 * specialCase)

		if rangeStart <= pile < rangeStop and (pile - rangeStart) % rangeStep == 0:
			dictionaryPileToLeaves[pile].append(leaf)

	return dictionaryPileToLeaves

def getLeafDomain(state: EliminationState, leaf: int) -> range:

	@cache
	def workhorse(leaf: int, dimensionsTotal: int, dimensionLength: int) -> range:
		return range(leaf.bit_count() + (2**(coordinatesOf0AtTail(leaf) + 1) - 2)
					, int(('1' * (leaf.bit_length()-1)).ljust(dimensionsTotal, '0'), dimensionLength) + 2 - (leaf.bit_count() - 1) - (leaf == leaf0)
									, 2 + (2 * (leaf == 首零(dimensionsTotal)+零)))
	return workhorse(leaf, state.dimensionsTotal, state.mapShape[0])

if __name__ == '__main__':
	from mapFolding.tests.verify import (  # pyright: ignore[reportUnusedImport]
		verifyDictionaryAddends4Next, verifyDictionaryAddends4Prior, verifyDictionaryLeafRanges, verifyDictionaryPileToLeaves)

	state = EliminationState((2,) * 6)
	dictionaryLeafRanges = getDictionaryLeafDomains(state)
	# verifyDictionaryLeafRanges(state, dictionaryLeafRanges)  # noqa: ERA001
	dictionaryAddends4Next = getDictionaryAddends4Next(state)
	# verifyDictionaryAddends4Next(state, dictionaryAddends4Next)  # noqa: ERA001
	dictionaryAddends4Prior = getDictionaryAddends4Prior(state)
	# verifyDictionaryAddends4Prior(state, dictionaryAddends4Prior)  # noqa: ERA001
	dictionaryPileToLeaves = getDictionaryPileToLeaves(state)
	# verifyDictionaryPileToLeaves(state, dictionaryPileToLeaves)  # noqa: ERA001
