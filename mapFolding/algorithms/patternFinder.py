# ruff: noqa: ERA001 T201
from cytoolz.functoolz import curry as syntacticCurry
from hunterMakesPy import raiseIfNone
from itertools import combinations, product as CartesianProduct, repeat
from mapFolding.dataBaskets import EliminationState
from math import prod
from more_itertools import chunked
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING
import gmpy2
import pandas

if TYPE_CHECKING:
	from collections.abc import Callable

def bitBiggest(integerAbove0: int, /) -> int:
	"""Find the 0-indexed position of the biggest set bit in `integerAbove0`."""
	return integerAbove0.bit_length() - 1

def bitSecond(integerAbove0: int, /) -> int | None:
	"""Find the 0-indexed position of the second biggest set bit, if any, in `integerAbove0`."""
	second = bitBiggest(int(gmpy2.bit_flip(integerAbove0, bitBiggest(integerAbove0))))
	return second if second >= 0 else None

def distanceFromPowerOf2(integerAbove0: int, /) -> int:
	return int(gmpy2.f_mod_2exp(integerAbove0, integerAbove0.bit_length() - 1))

def multiplicityOfPrimeFactor2(integerAbove0: int, /) -> int:
	"""Compute the number of times `integerAbove0` is divisible by 2; aka 'CTZ', Count Trailing Zeros in the binary form."""
	return raiseIfNone(gmpy2.bit_scan1(integerAbove0))

@syntacticCurry
def numeralOfLengthInBase(mostSignificantDigits: int | list[int], fillerDigits: str = '0', leastSignificantDigits: int | list[int] = 0, positions: int = 8, base: int = 2) -> int:
	"""Prototype."""
	digitsPrefix: tuple[int, ...] = (mostSignificantDigits,) if isinstance(mostSignificantDigits, int) else tuple(mostSignificantDigits)
	quantityPrefix: int = len(digitsPrefix)
	digitsSuffix: tuple[int, ...] = (leastSignificantDigits,) if isinstance(leastSignificantDigits, int) else tuple(leastSignificantDigits)
	quantitySuffix: int = len(digitsSuffix)
	quantityFiller: int = positions - quantityPrefix - quantitySuffix
	digitsFiller: tuple[int, ...] = tuple(int(digit) for digit in list(repeat(fillerDigits, quantityFiller))[0:quantityFiller])

	tupleDigitsMSBtoLSB: tuple[int, ...] = (*digitsPrefix, *digitsFiller, *digitsSuffix)
	digitsAsString: str = ''.join(str(digit) for digit in tupleDigitsMSBtoLSB)

	numeralAs_int: int = int(digitsAsString, base)

	return numeralAs_int

@syntacticCurry
def makeFillerDigitsNotation(numeral: int, positions: int = 8, base: int = 2) -> tuple[list[int], str, int | list[int]]:
	"""Represent `numeral` as prefix, filler digit, and suffix for reuse with `numeralOfLengthInBase`.

	(AI generated docstring)

	This prototype only supports base 2 and expects a non-negative numeral whose binary expansion
	fits inside the specified number of positions. The returned structure abstracts the repeated
	interior digits so that different position counts share the same notation.
	"""
	if positions <= 0:
		message: str = f'positions must be positive; received {positions}.'
		raise ValueError(message)
	if base != 2:
		message: str = f'makeFillerDigitsNotation currently supports base 2 only; received base {base}.'
		raise ValueError(message)
	if numeral < 0:
		message: str = f'numeral must be non-negative; received {numeral}.'
		raise ValueError(message)

	digitsAsString: str = f'{numeral:b}'
	if len(digitsAsString) > positions:
		message: str = f'numeral {numeral} requires {len(digitsAsString)} positions; received {positions}.'
		raise ValueError(message)
	digitsAsString = digitsAsString.zfill(positions)

	lengthPrefix: int | None = None
	lengthSuffix: int | None = None
	lengthFiller: int = -1
	fillerDigit: str = '0'

	for prefixLength in range(1, positions):
		for suffixLength in range(1, positions - prefixLength + 1):
			fillerLength: int = positions - prefixLength - suffixLength
			if fillerLength < 0:
				continue

			if fillerLength == 0:
				candidateFillerDigit: str = digitsAsString[prefixLength - 1]
			else:
				candidateFillerDigit = digitsAsString[prefixLength]
				segmentFiller: str = digitsAsString[prefixLength:prefixLength + fillerLength]
				if segmentFiller != candidateFillerDigit * fillerLength:
					continue

			if ((lengthPrefix is None)
				or (fillerLength > lengthFiller)
				or ((fillerLength == lengthFiller) and (
												(prefixLength < lengthPrefix)
												or ((prefixLength == lengthPrefix) and (suffixLength < lengthSuffix if lengthSuffix is not None else True))))):
				lengthPrefix = prefixLength
				lengthSuffix = suffixLength
				lengthFiller = fillerLength
				fillerDigit = candidateFillerDigit

	if lengthPrefix is None or lengthSuffix is None:
		lengthPrefix = positions if positions > 0 else 1
		lengthSuffix = max(0, positions - lengthPrefix)
		lengthFiller = 0
		fillerDigit = digitsAsString[lengthPrefix - 1] if positions > 0 else '0'

	mostSignificantDigits: list[int] = [int(digit) for digit in digitsAsString[0:lengthPrefix]]
	leastSignificantDigitsSequence: list[int] = [int(digit) for digit in digitsAsString[positions - lengthSuffix:]] if lengthSuffix > 0 else []
	leastSignificantDigits: int | list[int]
	if lengthSuffix == 1:
		leastSignificantDigits = leastSignificantDigitsSequence[0]
	else:
		leastSignificantDigits = leastSignificantDigitsSequence

	notation: tuple[list[int], str, int | list[int]] = (mostSignificantDigits, fillerDigit, leastSignificantDigits)
	return notation

def getDictionaryAddends4Next(state: EliminationState) -> dict[int, list[int]]:
	dictionaryAddends: dict[int, list[int]] = {}

	indexLeaf: int = 0
	listOfDifferences: list[int] = [1]
	dictionaryAddends[indexLeaf] = listOfDifferences

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
		dictionaryAddends[indexLeaf] = products下_indexLeaf

	return dictionaryAddends

def getDictionaryAddends4Prior(state: EliminationState) -> dict[int, list[int]]:
	dictionaryAddends: dict[int, list[int]] = {}

	indexLeaf: int = 0
	listOfDifferences: list[int] = []
	dictionaryAddends[indexLeaf] = listOfDifferences

	indexLeaf = 1
	listOfDifferences = [-1]
	dictionaryAddends[indexLeaf] = listOfDifferences

	productsOfDimensions: list[int] = [prod(state.mapShape[0:dimension], start=1) for dimension in range(state.dimensionsTotal)]

	for indexLeaf in range(state.leavesTotal - 1, 1, -1):
		products下_indexLeaf: list[int] = productsOfDimensions.copy()

		theMaskOfDirectionality = gmpy2.bit_mask(state.leavesTotal - 1) & indexLeaf
		for index in range(state.dimensionsTotal):
			if gmpy2.bit_test(theMaskOfDirectionality, index):
				products下_indexLeaf[index] *= -1

		slicingIndexStart: int = distanceFromPowerOf2(indexLeaf).bit_count() & 1
		slicingIndexEnd = (indexLeaf.bit_length() - 1) * (slicingIndexStart ^ 1) or None

		if (slicingIndexStart == 1) and (gmpy2.is_even(indexLeaf)):
			slicingIndexStart += multiplicityOfPrimeFactor2(indexLeaf)

		products下_indexLeaf = products下_indexLeaf[slicingIndexStart:None]
		products下_indexLeaf = products下_indexLeaf[0:slicingIndexEnd]
		dictionaryAddends[indexLeaf] = products下_indexLeaf

	return dictionaryAddends

def getDictionaryLeafRanges(state: EliminationState) -> dict[int, range]:
	dictionaryColumnStart: dict[int, int] = {}

	indexLeaf: int = 0
	columnStart: int = 0
	dictionaryColumnStart[indexLeaf] = columnStart

	for indexLeaf in range(1, state.leavesTotal):
		dictionaryColumnStart[indexLeaf] = 1 + distanceFromPowerOf2(indexLeaf).bit_count() + (2**(multiplicityOfPrimeFactor2(indexLeaf) + 1)) - 2

	# dictionaryExclusionaryStop: dict[int, int] = getDictionaryExclusionaryStop(state, dictionaryColumnStart)
	dictionaryExclusionaryStop: dict[int, int] = {}

	indexLeaf0: int = 0
	columnExclusionaryStop = dictionaryColumnStart[indexLeaf0] + 1
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

	dictionaryLeafRanges: dict[int, range] = {}
	for indexLeaf, columnStart in dictionaryColumnStart.items():
		dictionaryLeafRanges[indexLeaf] = range(columnStart, dictionaryExclusionaryStop[indexLeaf], 2)

	indexLeaf = state.leavesTotal // 2 + 1
	dictionaryLeafRanges[indexLeaf] = range(dictionaryColumnStart[indexLeaf], dictionaryExclusionaryStop[indexLeaf], 4)

	return dictionaryLeafRanges

# TODO
def getDictionaryPileToLeavesByFormula(state: EliminationState) -> dict[int, list[int]]:
	dictionaryPileToLeaves: dict[int, list[int]] = {pile: [] for pile in range(state.leavesTotal)}

	return dictionaryPileToLeaves

def getDictionaryPileToLeaves(state: EliminationState) -> dict[int, list[int]]:
	dictionaryPileToLeaves: dict[int, list[int]] = {pile: [] for pile in range(state.leavesTotal)}

	dictionaryLeafRanges: dict[int, range] = getDictionaryLeafRanges(state)

	for indexLeaf, rangePilings in dictionaryLeafRanges.items():
		for pile in rangePilings:
			dictionaryPileToLeaves[pile].append(indexLeaf)

	return dictionaryPileToLeaves

def _getGroupedBy(state: EliminationState, pileTarget: int, groupByIndexLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	pathFilename = Path(f'/apps/mapFolding/Z0Z_notes/arrayFoldingsP2d{state.dimensionsTotal}.pkl')
	arrayFoldings = pandas.read_pickle(pathFilename)  # noqa: S301
	dataframeFoldings = pandas.DataFrame(arrayFoldings)

	groupedBy: dict[int | tuple[int, ...], list[int]] = dataframeFoldings.groupby(list(groupByIndexLeavesAtPiles))[pileTarget].apply(list).to_dict()
	return {indexLeaves: sorted(set(listLeaves)) for indexLeaves, listLeaves in groupedBy.items()}

def getExcludedAddendIndices(state: EliminationState, indexLeafAddend: int, pileTarget: int, groupByIndexLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	groupedBy: dict[int | tuple[int, ...], list[int]] = _getGroupedBy(state, pileTarget, groupByIndexLeavesAtPiles)

	dictionaryExclusion: dict[int | tuple[int, ...], list[int]] = {}
	listAddends: list[int] = getDictionaryAddends4Next(state)[indexLeafAddend]

	for groupByIndexLeaves, listIndexLeavesIncludedAtPile in groupedBy.items():
		listAddendIndicesIncluded: list[int] = [addendIndex for addendIndex, addend in enumerate(listAddends) if indexLeafAddend + addend in listIndexLeavesIncludedAtPile]
		listAddendIndicesExcluded: list[int] = sorted(set(range(len(listAddends))).difference(set(listAddendIndicesIncluded)))
		dictionaryExclusion[groupByIndexLeaves] = listAddendIndicesExcluded

	return dictionaryExclusion

def getExcludedIndexLeaves(state: EliminationState, pileTarget: int, groupByIndexLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	return {indexLeaves: sorted(set(getDictionaryPileToLeaves(state)[pileTarget]).difference(set(listLeaves))) for indexLeaves, listLeaves in _getGroupedBy(state, pileTarget, groupByIndexLeavesAtPiles).items()}

def Z0Z_idk() -> None:
	state5 = EliminationState((2,) * 5)
	state6 = EliminationState((2,) * 6)

	_position5: Callable[[int], tuple[int | list[int], str, int | list[int]]] = makeFillerDigitsNotation(positions=state5.dimensionsTotal, base=state5.mapShape[0])
	ordinal5: Callable[[int | list[int], str, int | list[int]], int] = numeralOfLengthInBase(positions=state5.dimensionsTotal, base=state5.mapShape[0])

	_position6: Callable[[int], tuple[int | list[int], str, int | list[int]]] = makeFillerDigitsNotation(positions=state6.dimensionsTotal, base=state6.mapShape[0])
	ordinal6: Callable[[int | list[int], str, int | list[int]], int] = numeralOfLengthInBase(positions=state6.dimensionsTotal, base=state6.mapShape[0])

	rangePiles = range(5, state5.leavesTotal//2)

	for pile in rangePiles:
		print(f'        if pile == {pile}:')
		print("            listRemoveIndexLeaves = []")
		rangePileExcluders = range(2, pile)
		for pileExcluder in [*rangePileExcluders, 30, 29]:
			exclusions5: dict[int, list[int]] = getExcludedIndexLeaves(state5, pile, (pileExcluder,)) # pyright: ignore[reportAssignmentType]
			exclusions6: dict[int, list[int]] = getExcludedIndexLeaves(state6, pile, (pileExcluder,)) # pyright: ignore[reportAssignmentType]
			if pileExcluder > state5.leavesTotal//2:
				exclusions6: dict[int, list[int]] = getExcludedIndexLeaves(state6, pile, (pileExcluder + state5.leavesTotal,)) # pyright: ignore[reportAssignmentType]
			if pileExcluder < 8:
				print(f"            pileExcluder = {pileExcluder}")
			else:
				print(f"            pileExcluder = ordinal{_position5(pileExcluder)}")

			print("            indexLeafAtPileExcluder = pinnedLeavesWorkbench[pileExcluder]")
			for excluder5, listExcludees5 in exclusions5.items():
				if len(listExcludees5) == 0:
					continue
				list5AsPositions = [_position5(indexLeaf) for indexLeaf in listExcludees5]
				if excluder5 > state5.leavesTotal//2:
					excluder5 = ordinal6(*_position5(excluder5))
				list6AsPositions = [_position6(indexLeaf) for indexLeaf in exclusions6.get(excluder5, [])]
				listPositionsCommon = [str(indexLeafAsPosition) for indexLeafAsPosition in list5AsPositions if indexLeafAsPosition in list6AsPositions]
				if len(listPositionsCommon) == 0:
					continue
				print(f"            if indexLeafAtPileExcluder == ordinal{_position6(excluder5)}:")
				print(f"                listRemoveIndexLeaves.extend([ordinal{', ordinal'.join(listPositionsCommon)}])")



		print("            listIndexLeavesAtPile = addItUp(dictionaryAddends4Next, pinnedLeavesWorkbench[pile - 1], [])")
		print("            listIndexLeavesAtPile = sorted(set(listIndexLeavesAtPile).difference(set(listRemoveIndexLeaves)))")
		print()

if __name__ == '__main__':
	from mapFolding.tests.verify import (
		verifyDictionaryAddends4Next, verifyDictionaryAddends4Prior, verifyDictionaryLeafRanges)

	state = EliminationState((2,) * 6)
	dictionaryLeafRanges = getDictionaryLeafRanges(state)
	# verifyDictionaryLeafRanges(state, dictionaryLeafRanges)
	dictionaryAddends4Next = getDictionaryAddends4Next(state)
	# verifyDictionaryAddends4Next(state, dictionaryAddends4Next)
	dictionaryAddends4Prior = getDictionaryAddends4Prior(state)
	# verifyDictionaryAddends4Prior(state, dictionaryAddends4Prior)

	dictionaryPileToLeaves = getDictionaryPileToLeaves(state)
	colorReset = '\33[0m'
	color = '\33[91m'

	# print(dictionaryPileToLeaves[31])

	# for indexLeaf in [*dictionaryPileToLeaves[31]]:
	# 	print(f"{indexLeaf:2}: {distanceFromPowerOf2(indexLeaf - 0b11).bit_count()} ", end='')
	# 	for d in range(state.dimensionsTotal - 1, -1, -1):
	# 		bit = gmpy2.bit_test(indexLeaf, d)
	# 		print(f'\33[{3+(d&0)}{int(bit+3)}m', "█▊", colorReset, sep='', end='')
	# 	print()


	Z0Z_idk()
