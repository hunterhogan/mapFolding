# ruff: noqa: ERA001 T201 T203  # noqa: RUF100
from cytoolz.functoolz import curry as syntacticCurry
from gmpy2 import bit_mask, is_even, is_odd
from itertools import filterfalse, repeat
from mapFolding._e import (
	coordinatesOf0AtTail, getDictionaryAddends4Next, getDictionaryIndexLeafDomains, getDictionaryPileToIndexLeaves,
	getIndexLeafDomain, indexLeafSubHyperplane, ptount, 一, 三, 二, 四, 零, 首一, 首二, 首零, 首零一)
from mapFolding._e.pinning2DnData import dictionaryExclusions
from mapFolding.dataBaskets import EliminationState
from pathlib import Path
from pprint import pprint
import csv
import pandas

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

def _getDataFrameFoldings(state: EliminationState) -> pandas.DataFrame:
	pathFilename = Path(f'/apps/mapFolding/Z0Z_notes/arrayFoldingsP2d{state.dimensionsTotal}.pkl')
	arrayFoldings = pandas.read_pickle(pathFilename)  # noqa: S301
	return pandas.DataFrame(arrayFoldings)

def _getGroupedBy(state: EliminationState, pileTarget: int, groupByIndexLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	dataframeFoldings: pandas.DataFrame = _getDataFrameFoldings(state)
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
	return {indexLeaves: sorted(set(getDictionaryPileToIndexLeaves(state)[pileTarget]).difference(set(listLeaves))) for indexLeaves, listLeaves in _getGroupedBy(state, pileTarget, groupByIndexLeavesAtPiles).items()}

def getExcludingDictionary(state: EliminationState, indexLeafExcluder: int) -> dict[int, dict[int, list[int]]] | None:
	"""Get.

	dict[pileExcluder, dict[indexLeaf, listIndicesPilesExcluded]]
	If `indexLeafExcluder` is in `pileExcluder`, then `indexLeaf` is excluded from its domainOfPiles at `listIndicesPilesExcluded`.

	Use `state.leavesTotal` and `state.dimensionsTotal` to dynamically generate a dictionary appropriate for the `mapShape`.
	"""
	excludingDictionary: dict[int, dict[int, list[int]]] | None = None
	dictionaryIndexLeafDomains: dict[int, range] = getDictionaryIndexLeafDomains(state)

	if indexLeafExcluder == 一+零:
		domainOfPilesForIndexLeafExcluder: list[int] = list(dictionaryIndexLeafDomains[indexLeafExcluder])
		sizeDomainExcluder: int = len(domainOfPilesForIndexLeafExcluder)
		excludingDictionary = {pileExcluder: {indexLeaf: [] for indexLeaf in range(state.leavesTotal)} for pileExcluder in domainOfPilesForIndexLeafExcluder}

		for indexDomainPileExcluder, pileExcluder in enumerate(excludingDictionary):
			for indexLeaf in range(3, state.leavesTotal):
				domainOfPilesForIndexLeaf: list[int] = list(dictionaryIndexLeafDomains[indexLeaf])
				sizeDomainIndexLeaf: int = len(domainOfPilesForIndexLeaf)

				if indexDomainPileExcluder in list(filterfalse(lambda index: index % 4 == 3, range(sizeDomainExcluder))):  # noqa: SIM102
					if (一+零 < indexLeaf < 首零(state.dimensionsTotal)) and (ptount(indexLeaf) >= state.dimensionsTotal - 3):
						excludingDictionary[pileExcluder][indexLeaf].extend([-1])

				if is_odd(indexLeaf):
					excludingDictionary[pileExcluder][indexLeaf].extend([indexDomainPileExcluder])
					if 1 < coordinatesOf0AtTail(indexLeaf - 1) < state.dimensionsTotal - 1:
						excludingDictionary[pileExcluder][indexLeaf].extend([indexDomainPileExcluder + 1])

				if coordinatesOf0AtTail(indexLeaf) == 1:
					excludingDictionary[pileExcluder][indexLeaf].extend([indexDomainPileExcluder - 1])
					start = 0
					stop = indexDomainPileExcluder - (indexLeafSubHyperplane(indexLeaf) == 2) - (2 * max(0, indexLeaf.bit_count() - 3))
					excludingDictionary[pileExcluder][indexLeaf].extend([*range(start, stop)])

				if coordinatesOf0AtTail(indexLeaf) == 2:
					start = 0
					stop = indexDomainPileExcluder - 2 * (indexLeafSubHyperplane(indexLeaf) == 4) - 2
					excludingDictionary[pileExcluder][indexLeaf].extend([*range(start, stop)])

				if (indexLeaf == 首一(state.dimensionsTotal)):

					if pileExcluder <= 首二(state.dimensionsTotal):
						pass

					elif 首二(state.dimensionsTotal) < pileExcluder < 首一(state.dimensionsTotal):
						excludingDictionary[pileExcluder][indexLeaf].extend([*range(1, sizeDomainIndexLeaf // 2), *range(1 + sizeDomainIndexLeaf // 2, 3 * sizeDomainIndexLeaf // 4)])

					elif pileExcluder == 首一(state.dimensionsTotal):
						excludingDictionary[pileExcluder][indexLeaf].extend([*range(1, sizeDomainIndexLeaf // 2)])

					elif 首一(state.dimensionsTotal) < pileExcluder < 首零(state.dimensionsTotal)-一:
						excludingDictionary[pileExcluder][indexLeaf].extend([*range(3 * sizeDomainIndexLeaf // 4)])

					elif pileExcluder == 首零(state.dimensionsTotal)-一:
						excludingDictionary[pileExcluder][indexLeaf].extend([*range(1, 3 * sizeDomainIndexLeaf // 4)])

					elif pileExcluder == 首零(state.dimensionsTotal):
						excludingDictionary[pileExcluder][indexLeaf].extend([*range(2, sizeDomainIndexLeaf // 2)])

				if indexLeaf == 首零(state.dimensionsTotal) + 零:
					bump: int = 1 - int(pileExcluder.bit_count() == 1)
					howMany: int = state.dimensionsTotal - (pileExcluder.bit_length() + bump)
					onesInBinary = int(bit_mask(howMany))
					ImaPattern: int = sizeDomainIndexLeaf - onesInBinary

					if pileExcluder == 二:
						excludingDictionary[pileExcluder][indexLeaf].extend([零, 一, 二])

					if 二 < pileExcluder <= 首二(state.dimensionsTotal):
						stop: int = sizeDomainIndexLeaf // 2 - 1
						excludingDictionary[pileExcluder][indexLeaf].extend(range(1, stop))

						aDimensionPropertyNotFullyUnderstood = 5
						for _dimension in range(state.dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
							start: int = 1 + stop
							stop += (stop+1) // 2
							excludingDictionary[pileExcluder][indexLeaf].extend([*range(start, stop)])

						excludingDictionary[pileExcluder][indexLeaf].extend([*range(1 + stop, ImaPattern)])

					if 首二(state.dimensionsTotal) < pileExcluder:
						excludingDictionary[pileExcluder][indexLeaf].extend([*range(1, ImaPattern)])

				def normalizeIndex(index: int, lengthIterable: int) -> int:
					if index < 0:
						index = (index + lengthIterable) % lengthIterable
					return index

				excludingDictionary[pileExcluder][indexLeaf] = sorted(set(map(normalizeIndex, excludingDictionary[pileExcluder][indexLeaf], repeat(sizeDomainIndexLeaf))))

	# else:
	# 	excludingDictionary: dict[int, dict[int, list[int]]] | None = dictionaryExclusions.get(indexLeafExcluder)
	return excludingDictionary

def analyzeExclusions() -> None:
	"""Analyze.

	Analyze data, and write formulas to be processed by `getExcludingDictionary`.

	Make formulas to create `listIndicesPilesExcluded` as a function of `pileExcluder` and `indexLeaf`.
	`pileExcluder` can be described by its index in `getIndexLeafRange(state, indexLeafExcluder)`. I wonder if the index is a better input variable.
	"""
	dictionaryExclusionsWIP: dict[int, dict[int, dict[int, list[int]]]] = {}

	for indexLeafExcluder in [二+零, 三+零, 三, 四+零, 四+一, 四+三, 四+二, 一+零]:
	# for indexLeafExcluder in [一+零]:
		for dimensions in [6]:
			state = EliminationState((2,) * dimensions)

			dictionaryByPileExcluderFromIndexLeafToPilesExcluded: dict[int, dict[int, list[int]]] = {}
			dictionaryByPileExcluderFromIndexLeafToIndicesPilesExcluded: dict[int, dict[int, list[int]]] = {}

			domainOfPilesForIndexLeafExcluder: list[int] = list(getIndexLeafDomain(state, indexLeafExcluder))

			for indexDomainPileExcluder, pileExcluder in enumerate(domainOfPilesForIndexLeafExcluder):
				dictionaryByPileExcluderFromIndexLeafToPilesExcluded[indexDomainPileExcluder] = {indexLeaf: [pileExcluder] for indexLeaf in range(2, state.leavesTotal)}
				del dictionaryByPileExcluderFromIndexLeafToPilesExcluded[indexDomainPileExcluder][indexLeafExcluder]
				del dictionaryByPileExcluderFromIndexLeafToPilesExcluded[indexDomainPileExcluder][首零(state.dimensionsTotal)]
				for pileTarget in range(2, state.columnLast):
					if pileTarget == pileExcluder:
						continue
					dictionaryExcludedIndexLeaves: dict[int, list[int]] = getExcludedIndexLeaves(state, pileTarget, (pileExcluder,)) # pyright: ignore[reportAssignmentType]
					for indexLeaf in dictionaryExcludedIndexLeaves[indexLeafExcluder]:
						if indexLeaf == indexLeafExcluder:
							continue
						beans, cornbread = 一+零, 一
						if (beans == indexLeafExcluder and cornbread == indexLeaf) or (cornbread == indexLeafExcluder and beans == indexLeaf):
							continue
						beans, cornbread = 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)
						if (beans == indexLeafExcluder and cornbread == indexLeaf) or (cornbread == indexLeafExcluder and beans == indexLeaf):
							continue
						dictionaryByPileExcluderFromIndexLeafToPilesExcluded[indexDomainPileExcluder].setdefault(indexLeaf, []).append(pileTarget)

			for indexDomainPileExcluder, dictionaryIndexLeafToPilesExcluded in dictionaryByPileExcluderFromIndexLeafToPilesExcluded.items():
				dictionaryByPileExcluderFromIndexLeafToIndicesPilesExcluded[indexDomainPileExcluder] = {}
				for indexLeaf, listPilesExcluded in dictionaryIndexLeafToPilesExcluded.items():
					setPilesExcluded: set[int] = set(listPilesExcluded)
					domainOfPilesForIndexLeaf: list[int] = list(getIndexLeafDomain(state, indexLeaf))
					listIndicesPilesExcluded: list[int] = sorted([domainOfPilesForIndexLeaf.index(pile) for pile in domainOfPilesForIndexLeaf if pile in setPilesExcluded])
					dictionaryByPileExcluderFromIndexLeafToIndicesPilesExcluded[indexDomainPileExcluder][indexLeaf] = listIndicesPilesExcluded

			# pprint(dictionaryByPileExcluderFromIndexLeafToIndicesPilesExcluded, width=240)

			pathFilename = Path("/apps/mapFolding/Z0Z_notes/analyzeExcluders2Dn.csv")
			with pathFilename.open('w', newline='') as writeStream:
				writerCSV = csv.writer(writeStream)
				listColumns: list[int] = list(range(首零(state.dimensionsTotal)))
				writerCSV.writerow(['indexLeafExcluder', 'dimensions', 'pileExcluder', 'indexDomainPileExcluder', 'indexLeaf', 'sizeDomainOfPilesForIndexLeaf', *listColumns])
				for indexDomainPileExcluder, dictionaryIndexLeafToIndicesPilesExcluded in dictionaryByPileExcluderFromIndexLeafToIndicesPilesExcluded.items():
					for indexLeaf, listIndicesPilesExcluded in dictionaryIndexLeafToIndicesPilesExcluded.items():
						Z0Z_list: list[int | str] = [''] * 首零(state.dimensionsTotal)
						for index in listIndicesPilesExcluded:
							Z0Z_list[index] = index
						writerCSV.writerow([indexLeafExcluder, dimensions, domainOfPilesForIndexLeafExcluder[indexDomainPileExcluder], indexDomainPileExcluder, indexLeaf, len(list(getIndexLeafDomain(state, indexLeaf))), *Z0Z_list])

			dictionaryExclusionsWIP[indexLeafExcluder] = dictionaryByPileExcluderFromIndexLeafToIndicesPilesExcluded

	pathFilename = Path('/apps/mapFolding/mapFolding/_e/pinning2DnData.py')
	pathFilename.write_text(f"dictionaryExclusions = {dictionaryExclusionsWIP!r}\n")

if __name__ == '__main__':
	state = EliminationState((2,) * 4)
	# analyzeExclusions()

