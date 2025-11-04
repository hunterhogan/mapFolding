# ruff: noqa: ERA001
from hunterMakesPy import raiseIfNone
from mapFolding.dataBaskets import EliminationState
from math import prod
import gmpy2

def multiplicityOfPrimeFactor2(integerAbove0: int, /) -> int:
	"""Compute the number of times `integerAbove0` is divisible by 2; aka 'CTZ', Count Trailing Zeros in the binary form."""
	return raiseIfNone(gmpy2.bit_scan1(integerAbove0))

def distanceFromPowerOf2(integerAbove0: int, /) -> int:
	return int(gmpy2.f_mod_2exp(integerAbove0, integerAbove0.bit_length() - 1))

def getDictionaryExclusionaryStop(state: EliminationState, dictionaryStart: dict[int, int]) -> dict[int, int]:
	dictionaryExclusionaryStop: dict[int, int] = {}

	indexLeaf0: int = 0
	columnExclusionaryStop = dictionaryStart[indexLeaf0] + 1
	dictionaryExclusionaryStop[indexLeaf0] = columnExclusionaryStop

	# The only indexLeaf in columnN == state.leavesTotal - 1.
	indexLeaf: int = state.leavesTotal // 2
	columnExclusionaryStop = state.leavesTotal
	dictionaryExclusionaryStop[indexLeaf] = columnExclusionaryStop

	exponent: int = 0
	indexLeafPowerOf2: int = state.leavesTotal // 2
	while indexLeafPowerOf2 > 1:
		indexLeafPowerOf2 //= 2
		exponent += 1
		columnExclusionaryStop -= 2**exponent

		for indexLeaf in range(indexLeafPowerOf2, indexLeafPowerOf2 * 2):
			dictionaryExclusionaryStop[indexLeaf] = columnExclusionaryStop - distanceFromPowerOf2(indexLeaf).bit_count()

	for indexLeaf in range(state.leavesTotal // 2, state.leavesTotal):
		dictionaryExclusionaryStop[indexLeaf] = 2**indexLeaf.bit_length() - distanceFromPowerOf2(indexLeaf).bit_count()

	return dictionaryExclusionaryStop

def getDictionaryLeafRanges(state: EliminationState) -> dict[int, range]:
	dictionaryColumnStart: dict[int, int] = {}

	indexLeaf: int = 0
	columnStart: int = 0
	dictionaryColumnStart[indexLeaf] = columnStart

	for indexLeaf in range(1, state.leavesTotal):
		dictionaryColumnStart[indexLeaf] = 1 + distanceFromPowerOf2(indexLeaf).bit_count() + (2**(multiplicityOfPrimeFactor2(indexLeaf) + 1)) - 2

	dictionaryExclusionaryStop: dict[int, int] = getDictionaryExclusionaryStop(state, dictionaryColumnStart)

	dictionaryLeafRanges: dict[int, range] = {}
	for indexLeaf, columnStart in dictionaryColumnStart.items():
		dictionaryLeafRanges[indexLeaf] = range(columnStart, dictionaryExclusionaryStop[indexLeaf], 2)

	indexLeaf = state.leavesTotal // 2 + 1
	dictionaryLeafRanges[indexLeaf] = range(dictionaryColumnStart[indexLeaf], dictionaryExclusionaryStop[indexLeaf], 4)

	return dictionaryLeafRanges

def getDictionaryDifferences(state: EliminationState) -> dict[int, list[int]]:
	dictionaryDifferences: dict[int, list[int]] = {}

	indexLeaf: int = 0
	listOfDifferences: list[int] = [1]
	dictionaryDifferences[indexLeaf] = listOfDifferences

	productsOfDimensions: list[int] = [prod(state.mapShape[0:dimension], start=1) for dimension in range(state.dimensionsTotal)]

	for indexLeaf in range(1, state.leavesTotal):
		products下_indexLeaf: list[int] = productsOfDimensions.copy()

		theMaskOfDirectionality = gmpy2.bit_mask(state.leavesTotal - 1) & indexLeaf
		for index in range(state.dimensionsTotal):
			if gmpy2.bit_test(theMaskOfDirectionality, index):
				products下_indexLeaf[index] *= -1

		slicingIndexStart: int = distanceFromPowerOf2(indexLeaf).bit_count() & 1 ^ 1
		slicingIndexEnd = (indexLeaf.bit_length() - 1) * (slicingIndexStart ^ 1) or None

		if (slicingIndexStart == 1) and (gmpy2.is_even(indexLeaf)):
			slicingIndexStart += multiplicityOfPrimeFactor2(indexLeaf)

		products下_indexLeaf = products下_indexLeaf[slicingIndexStart:None]
		products下_indexLeaf = products下_indexLeaf[0:slicingIndexEnd]
		dictionaryDifferences[indexLeaf] = products下_indexLeaf

	return dictionaryDifferences

def getDictionaryDifferencesReverse(state: EliminationState) -> dict[int, list[int]]:
	dictionaryDifferencesReverse: dict[int, list[int]] = {}

	productsOfDimensions: list[int] = [prod(state.mapShape[0:dimension], start=1) for dimension in range(state.dimensionsTotal)]

	for indexLeaf in range(1, state.leavesTotal):
		products下_indexLeaf: list[int] = productsOfDimensions.copy()

		theMaskOfDirectionality = gmpy2.bit_mask(state.leavesTotal - 1) & indexLeaf
		for index in range(state.dimensionsTotal):
			if gmpy2.bit_test(theMaskOfDirectionality, index):
				products下_indexLeaf[index] *= -1

		slicingIndexStart: int = distanceFromPowerOf2(indexLeaf).bit_count() & 1 ^ 1
		slicingIndexEnd = (indexLeaf.bit_length() - 1) * (slicingIndexStart ^ 1) or None

		originalSlicingIndexStart: int = slicingIndexStart
		if (slicingIndexStart == 1) and (gmpy2.is_even(indexLeaf)):
			slicingIndexStart += multiplicityOfPrimeFactor2(indexLeaf)

		forwardResult: list[int] = products下_indexLeaf[slicingIndexStart:None]
		forwardResult = forwardResult[0:slicingIndexEnd]

		excludedStart: list[int] = products下_indexLeaf[0:slicingIndexStart]
		excludedEnd: list[int] = products下_indexLeaf[slicingIndexEnd:] if slicingIndexEnd is not None else []

		listOfDifferencesReverse: list[int] = []
		if excludedEnd:
			if gmpy2.is_odd(indexLeaf) and (originalSlicingIndexStart == 0):
				listOfDifferencesReverse = [*forwardResult[1:], *excludedEnd]
			elif slicingIndexEnd is not None:
				countExtraElements: int = max(0, slicingIndexEnd - multiplicityOfPrimeFactor2(indexLeaf) - 1)
				listOfDifferencesReverse = [*forwardResult[-countExtraElements:], *excludedEnd] if countExtraElements > 0 else excludedEnd
			else:
				listOfDifferencesReverse = excludedEnd
		elif excludedStart:
			if slicingIndexStart > originalSlicingIndexStart:
				multiplicityFactor2: int = multiplicityOfPrimeFactor2(indexLeaf)
				if multiplicityFactor2 > 1:
					bitCount: int = indexLeaf.bit_count()
					if bitCount == 1:
						listOfDifferencesReverse = excludedStart[:-1]
					else:
						bitLength: int = indexLeaf.bit_length()
						countExtraElements: int = bitLength - multiplicityFactor2 - 2
						listOfDifferencesReverse = [*excludedStart, *forwardResult[0:countExtraElements]]
				else:
					bitLength: int = indexLeaf.bit_length()
					if bitLength == 2:
						listOfDifferencesReverse = [excludedStart[0]]
					else:
						countExtraElements: int = bitLength - 3
						listOfDifferencesReverse = [*excludedStart, *forwardResult[0:countExtraElements]]
			elif gmpy2.is_odd(indexLeaf) and (originalSlicingIndexStart == 1):
				countExtraElements: int = max(0, indexLeaf.bit_length() - 2)
				listOfDifferencesReverse = [*excludedStart, *forwardResult[0:countExtraElements]]
			elif originalSlicingIndexStart == 1:
				listOfDifferencesReverse = [excludedStart[0]]
			else:
				listOfDifferencesReverse = excludedStart

		dictionaryDifferencesReverse[indexLeaf] = listOfDifferencesReverse

	return dictionaryDifferencesReverse

if __name__ == '__main__':
	state= EliminationState((2,) * 6)

	from mapFolding.tests.verify import (
		verifyDictionaryDifferences, verifyDictionaryDifferencesReverse, verifyDictionaryLeafRanges)
	dictionaryLeafRanges = getDictionaryLeafRanges(state)
	dictionaryDifferences = getDictionaryDifferences(state)
	dictionaryDifferencesReverse = getDictionaryDifferencesReverse(state)
	verifyDictionaryLeafRanges(state, dictionaryLeafRanges)
	# verifyDictionaryDifferences(state, dictionaryDifferences)
	# verifyDictionaryDifferencesReverse(state, dictionaryDifferencesReverse)
