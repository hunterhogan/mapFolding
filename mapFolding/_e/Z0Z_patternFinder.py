# ruff: noqa: ERA001 T201 T203  # noqa: RUF100
from gmpy2 import bit_mask, is_even, is_odd
from itertools import filterfalse, repeat
from mapFolding import reverseLookup
from mapFolding._e import (
	getDictionaryAddends4Next, getDictionaryLeafDomains, getDictionaryPileToLeaves, getLeafDomain,
	howMany0coordinatesAtTail, leafInSubHyperplane, ptount, 一, 三, 二, 四, 零, 首一, 首二, 首零, 首零一)
from mapFolding.dataBaskets import EliminationState
from more_itertools import extract
from pathlib import Path
from pprint import pprint
import csv
import pandas

def _getDataFrameFoldings(state: EliminationState) -> pandas.DataFrame:
	pathFilename = Path(f'/apps/mapFolding/Z0Z_notes/arrayFoldingsP2d{state.dimensionsTotal}.pkl')
	arrayFoldings = pandas.read_pickle(pathFilename)  # noqa: S301
	return pandas.DataFrame(arrayFoldings)

def _getGroupedBy(state: EliminationState, pileTarget: int, groupByLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	dataframeFoldings: pandas.DataFrame = _getDataFrameFoldings(state)
	groupedBy: dict[int | tuple[int, ...], list[int]] = dataframeFoldings.groupby(list(groupByLeavesAtPiles))[pileTarget].apply(list).to_dict()
	return {leaves: sorted(set(listLeaves)) for leaves, listLeaves in groupedBy.items()}

def getExcludedAddendIndices(state: EliminationState, leafAddend: int, pileTarget: int, groupByLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	groupedBy: dict[int | tuple[int, ...], list[int]] = _getGroupedBy(state, pileTarget, groupByLeavesAtPiles)

	dictionaryExclusion: dict[int | tuple[int, ...], list[int]] = {}
	listAddends: list[int] = getDictionaryAddends4Next(state)[leafAddend]

	for groupByLeaves, listLeavesIncludedAtPile in groupedBy.items():
		listAddendIndicesIncluded: list[int] = [addendIndex for addendIndex, addend in enumerate(listAddends) if leafAddend + addend in listLeavesIncludedAtPile]
		listAddendIndicesExcluded: list[int] = sorted(set(range(len(listAddends))).difference(set(listAddendIndicesIncluded)))
		dictionaryExclusion[groupByLeaves] = listAddendIndicesExcluded

	return dictionaryExclusion

def getExcludedLeaves(state: EliminationState, pileTarget: int, groupByLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	return {leaves: sorted(set(getDictionaryPileToLeaves(state)[pileTarget]).difference(set(listLeaves))) for leaves, listLeaves in _getGroupedBy(state, pileTarget, groupByLeavesAtPiles).items()}

def getExcludingDictionary(state: EliminationState, leafExcluder: int) -> dict[int, dict[int, list[int]]] | None:
	"""Get.

	dict[pileExcluder, dict[leaf, listIndicesPilesExcluded]]
	If `leafExcluder` is in `pileExcluder`, then `leaf` is excluded from its domainOfPiles at `listIndicesPilesExcluded`.

	Use `state.leavesTotal` and `state.dimensionsTotal` to dynamically generate a dictionary appropriate for the `mapShape`.
	"""
	excludingDictionary: dict[int, dict[int, list[int]]] | None = None
	dictionaryLeafDomains: dict[int, range] = getDictionaryLeafDomains(state)

	if leafExcluder == 一+零:
		domainOfPilesForLeafExcluder: list[int] = list(dictionaryLeafDomains[leafExcluder])
		sizeDomainExcluder: int = len(domainOfPilesForLeafExcluder)
		excludingDictionary = {pileExcluder: {leaf: [] for leaf in range(state.leavesTotal)} for pileExcluder in domainOfPilesForLeafExcluder}

		for indexDomainPileExcluder, pileExcluder in enumerate(excludingDictionary):
			for leaf in range(3, state.leavesTotal):
				domainOfPilesForLeaf: list[int] = list(dictionaryLeafDomains[leaf])
				sizeDomainLeaf: int = len(domainOfPilesForLeaf)

				if indexDomainPileExcluder in list(filterfalse(lambda index: index % 4 == 3, range(sizeDomainExcluder))):  # noqa: SIM102
					if (一+零 < leaf < 首零(state.dimensionsTotal)) and (ptount(leaf) >= state.dimensionsTotal - 3):
						excludingDictionary[pileExcluder][leaf].extend([-1])

				if is_odd(leaf):
					excludingDictionary[pileExcluder][leaf].extend([indexDomainPileExcluder])
					if 1 < howMany0coordinatesAtTail(leaf - 1) < state.dimensionsTotal - 1:
						excludingDictionary[pileExcluder][leaf].extend([indexDomainPileExcluder + 1])

				if howMany0coordinatesAtTail(leaf) == 1:
					excludingDictionary[pileExcluder][leaf].extend([indexDomainPileExcluder - 1])
					start = 0
					stop = indexDomainPileExcluder - (leafInSubHyperplane(leaf) == 2) - (2 * max(0, leaf.bit_count() - 3))
					excludingDictionary[pileExcluder][leaf].extend([*range(start, stop)])

				if howMany0coordinatesAtTail(leaf) == 2:
					start = 0
					stop = indexDomainPileExcluder - 2 * (leafInSubHyperplane(leaf) == 4) - 2
					excludingDictionary[pileExcluder][leaf].extend([*range(start, stop)])

				if (leaf == 首一(state.dimensionsTotal)):

					if pileExcluder <= 首二(state.dimensionsTotal):
						pass

					elif 首二(state.dimensionsTotal) < pileExcluder < 首一(state.dimensionsTotal):
						excludingDictionary[pileExcluder][leaf].extend([*range(1, sizeDomainLeaf // 2), *range(1 + sizeDomainLeaf // 2, 3 * sizeDomainLeaf // 4)])

					elif pileExcluder == 首一(state.dimensionsTotal):
						excludingDictionary[pileExcluder][leaf].extend([*range(1, sizeDomainLeaf // 2)])

					elif 首一(state.dimensionsTotal) < pileExcluder < 首零(state.dimensionsTotal)-一:
						excludingDictionary[pileExcluder][leaf].extend([*range(3 * sizeDomainLeaf // 4)])

					elif pileExcluder == 首零(state.dimensionsTotal)-一:
						excludingDictionary[pileExcluder][leaf].extend([*range(1, 3 * sizeDomainLeaf // 4)])

					elif pileExcluder == 首零(state.dimensionsTotal):
						excludingDictionary[pileExcluder][leaf].extend([*range(2, sizeDomainLeaf // 2)])

				if leaf == 首零(state.dimensionsTotal) + 零:
					bump: int = 1 - int(pileExcluder.bit_count() == 1)
					howMany: int = state.dimensionsTotal - (pileExcluder.bit_length() + bump)
					onesInBinary = int(bit_mask(howMany))
					ImaPattern: int = sizeDomainLeaf - onesInBinary

					if pileExcluder == 二:
						excludingDictionary[pileExcluder][leaf].extend([零, 一, 二])

					if 二 < pileExcluder <= 首二(state.dimensionsTotal):
						stop: int = sizeDomainLeaf // 2 - 1
						excludingDictionary[pileExcluder][leaf].extend(range(1, stop))

						aDimensionPropertyNotFullyUnderstood = 5
						for _dimension in range(state.dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
							start: int = 1 + stop
							stop += (stop+1) // 2
							excludingDictionary[pileExcluder][leaf].extend([*range(start, stop)])

						excludingDictionary[pileExcluder][leaf].extend([*range(1 + stop, ImaPattern)])

					if 首二(state.dimensionsTotal) < pileExcluder:
						excludingDictionary[pileExcluder][leaf].extend([*range(1, ImaPattern)])

				def normalizeIndex(index: int, lengthIterable: int) -> int:
					if index < 0:
						index = (index + lengthIterable) % lengthIterable
					return index

				excludingDictionary[pileExcluder][leaf] = sorted(set(map(normalizeIndex, excludingDictionary[pileExcluder][leaf], repeat(sizeDomainLeaf))))

	# else:
	# 	excludingDictionary: dict[int, dict[int, list[int]]] | None = dictionaryExclusions.get(leafExcluder)
	return excludingDictionary

def Z0Z_TESTexcluding2d5(state: EliminationState, leaf: int) -> bool:

	for leafExcluder in range(state.leavesTotal):
		excludingDictionary: dict[int, dict[int, list[int]]] | None = getExcludingDictionary(state, leafExcluder)
		if excludingDictionary is None:
			continue

		if leaf != leafExcluder and leafExcluder in state.pinnedLeaves.values():
			pileExcluder: int = reverseLookup(state.pinnedLeaves, leafExcluder)
			if pileExcluder in excludingDictionary:
				dictionaryIndicesPilesExcluded: dict[int, list[int]] = excludingDictionary[pileExcluder]
				if leaf in dictionaryIndicesPilesExcluded:
					listIndicesPilesExcluded: list[int] = dictionaryIndicesPilesExcluded[leaf]
					domainOfPilesForLeaf: list[int] = list(getLeafDomain(state, leaf))
					listPilesExcluded: list[int] = list(extract(domainOfPilesForLeaf, listIndicesPilesExcluded))
					if state.pile in listPilesExcluded:
						return True

	return False

def analyzeExclusions() -> None:
	"""Analyze.

	Analyze data, and write formulas to be processed by `getExcludingDictionary`.

	Make formulas to create `listIndicesPilesExcluded` as a function of `pileExcluder` and `leaf`.
	`pileExcluder` can be described by its index in `getLeafRange(state, leafExcluder)`. I wonder if the index is a better input variable.
	"""
	dictionaryExclusionsWIP: dict[int, dict[int, dict[int, list[int]]]] = {}

	for leafExcluder in [二+零, 三+零, 三, 四+零, 四+一, 四+三, 四+二, 一+零]:
	# for leafExcluder in [一+零]:
		for dimensions in [6]:
			state = EliminationState((2,) * dimensions)

			dictionaryByPileExcluderFromLeafToPilesExcluded: dict[int, dict[int, list[int]]] = {}
			dictionaryByPileExcluderFromLeafToIndicesPilesExcluded: dict[int, dict[int, list[int]]] = {}

			domainOfPilesForLeafExcluder: list[int] = list(getLeafDomain(state, leafExcluder))

			for indexDomainPileExcluder, pileExcluder in enumerate(domainOfPilesForLeafExcluder):
				dictionaryByPileExcluderFromLeafToPilesExcluded[indexDomainPileExcluder] = {leaf: [pileExcluder] for leaf in range(2, state.leavesTotal)}
				del dictionaryByPileExcluderFromLeafToPilesExcluded[indexDomainPileExcluder][leafExcluder]
				del dictionaryByPileExcluderFromLeafToPilesExcluded[indexDomainPileExcluder][首零(state.dimensionsTotal)]
				for pileTarget in range(2, state.pileLast):
					if pileTarget == pileExcluder:
						continue
					dictionaryExcludedLeaves: dict[int, list[int]] = getExcludedLeaves(state, pileTarget, (pileExcluder,)) # pyright: ignore[reportAssignmentType]
					for leaf in dictionaryExcludedLeaves[leafExcluder]:
						if leaf == leafExcluder:
							continue
						beans, cornbread = 一+零, 一
						if (beans == leafExcluder and cornbread == leaf) or (cornbread == leafExcluder and beans == leaf):
							continue
						beans, cornbread = 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)
						if (beans == leafExcluder and cornbread == leaf) or (cornbread == leafExcluder and beans == leaf):
							continue
						dictionaryByPileExcluderFromLeafToPilesExcluded[indexDomainPileExcluder].setdefault(leaf, []).append(pileTarget)

			for indexDomainPileExcluder, dictionaryLeafToPilesExcluded in dictionaryByPileExcluderFromLeafToPilesExcluded.items():
				dictionaryByPileExcluderFromLeafToIndicesPilesExcluded[indexDomainPileExcluder] = {}
				for leaf, listPilesExcluded in dictionaryLeafToPilesExcluded.items():
					setPilesExcluded: set[int] = set(listPilesExcluded)
					domainOfPilesForLeaf: list[int] = list(getLeafDomain(state, leaf))
					listIndicesPilesExcluded: list[int] = sorted([domainOfPilesForLeaf.index(pile) for pile in domainOfPilesForLeaf if pile in setPilesExcluded])
					dictionaryByPileExcluderFromLeafToIndicesPilesExcluded[indexDomainPileExcluder][leaf] = listIndicesPilesExcluded

			# pprint(dictionaryByPileExcluderFromLeafToIndicesPilesExcluded, width=240)

			pathFilename = Path("/apps/mapFolding/Z0Z_notes/analyzeExcluders2Dn.csv")
			with pathFilename.open('w', newline='') as writeStream:
				writerCSV = csv.writer(writeStream)
				listPiles: list[int] = list(range(首零(state.dimensionsTotal)))
				writerCSV.writerow(['leafExcluder', 'dimensions', 'pileExcluder', 'indexDomainPileExcluder', 'leaf', 'sizeDomainOfPilesForLeaf', *listPiles])
				for indexDomainPileExcluder, dictionaryLeafToIndicesPilesExcluded in dictionaryByPileExcluderFromLeafToIndicesPilesExcluded.items():
					for leaf, listIndicesPilesExcluded in dictionaryLeafToIndicesPilesExcluded.items():
						Z0Z_list: list[int | str] = [''] * 首零(state.dimensionsTotal)
						for index in listIndicesPilesExcluded:
							Z0Z_list[index] = index
						writerCSV.writerow([leafExcluder, dimensions, domainOfPilesForLeafExcluder[indexDomainPileExcluder], indexDomainPileExcluder, leaf, len(list(getLeafDomain(state, leaf))), *Z0Z_list])

			dictionaryExclusionsWIP[leafExcluder] = dictionaryByPileExcluderFromLeafToIndicesPilesExcluded

	pathFilename = Path('/apps/mapFolding/mapFolding/_e/pinning2DnData.py')
	# pathFilename.write_text(f"dictionaryExclusions = {dictionaryExclusionsWIP!r}\n")

if __name__ == '__main__':
	state = EliminationState((2,) * 4)
	# analyzeExclusions()

