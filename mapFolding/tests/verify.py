# ruff: noqa: T201 T203
from cytoolz.curried import dissoc, filter, map as toolz_map, valmap  # noqa: A004
from cytoolz.functoolz import compose
from gmpy2 import fac
from mapFolding._e import PinnedLeaves, getDictionaryAddends4Next
from mapFolding._e.pinning2DnAnnex import beansWithoutCornbread
from mapFolding.dataBaskets import EliminationState
from math import prod
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING
import csv
import numpy
import pickle

if TYPE_CHECKING:
	from collections.abc import Callable
	from numpy.typing import NDArray

def verifyDictionaryLeafRanges(state: EliminationState, dictionaryLeafRanges: dict[int, range]) -> None:
	dictionaryLeafRangesStartKnown32: dict[int, int] = {0: 0, 1: 1, 2: 3, 3: 2, 4: 7, 5: 2, 6: 4, 7: 3, 8: 15, 9: 2, 10: 4, 11: 3, 12: 8, 13: 3, 14: 5, 15: 4, 16: 31, 17: 2, 18: 4, 19: 3, 20: 8, 21: 3, 22: 5, 23: 4, 24: 16, 25: 3, 26: 5, 27: 4, 28: 9, 29: 4, 30: 6, 31: 5}
	dictionaryLeafRangesStartKnown64Piles: list[int] = [0,1,3,2,7,2,4,3,15,2,4,3,8,3,5,4,31,2,4,3,8,3,5,4,16,3,5,4,9,4,6,5,63,2,4,3,8,3,5,4,16,3,5,4,9,4,6,5,32,3,5,4,9,4,6,5,17,4,6,5,10,5,7,6]
	dictionaryLeafRangesStartKnown64: dict[int, int] = {leaf: dictionaryLeafRangesStartKnown64Piles[leaf] for leaf in range(64)}
	dictionaryLeafRangesStopKnown64Piles: list[int] = [1,2,34,33,50,49,49,48,58,57,57,56,57,56,56,55,62,61,61,60,61,60,60,59,61,60,60,59,60,59,59,58,64,63,63,62,63,62,62,61,63,62,62,61,62,61,61,60,63,62,62,61,62,61,61,60,62,61,61,60,61,60,60,59]
	dictionaryLeafRangesStopKnown64: dict[int, int] = {leaf: dictionaryLeafRangesStopKnown64Piles[leaf] for leaf in range(64)}

	dictionaryLeafRangesStartKnown: dict[int, int] | None = None
	dictionaryLeafRangesStopKnown: dict[int, int] | None = None
	if state.leavesTotal == 32:
		dictionaryLeafRangesStartKnown = dictionaryLeafRangesStartKnown32
	elif state.leavesTotal == 64:
		dictionaryLeafRangesStartKnown = dictionaryLeafRangesStartKnown64
		dictionaryLeafRangesStopKnown = dictionaryLeafRangesStopKnown64

	issues: int = 0

	for leaf, rangeLeaf in sorted(dictionaryLeafRanges.items()):
		print(leaf, repr(rangeLeaf), sep="\t")
		if dictionaryLeafRangesStartKnown is not None and rangeLeaf.start != dictionaryLeafRangesStartKnown[leaf]:
			issues += 1
			print(f"\33[91mKnown range start: {dictionaryLeafRangesStartKnown[leaf]:2d}\33[0m")

		if dictionaryLeafRangesStopKnown is not None:
			if rangeLeaf.stop < dictionaryLeafRangesStopKnown[leaf]:
				issues += 1
				print(f"\33[91mKnown range stop: {dictionaryLeafRangesStopKnown[leaf]:2d}\33[0m")
			if rangeLeaf.stop > dictionaryLeafRangesStopKnown[leaf] + 1:
				issues += 1
				print(f"Known range stop: {dictionaryLeafRangesStopKnown[leaf]:2d}\33[0m")

	if issues:
		print(f"\33[91mFound {issues} issues.\33[0m")
	else:
		print("\33[92mNo issues found.\33[0m")
	print(len(dictionaryLeafRanges), "of", state.leavesTotal)

def verifyDictionaryAddends4Next(state: EliminationState, dictionaryAddends4Next: dict[int, list[int]]) -> None:
	dictionaryAddends4NextKnown64: dict[int, list[int]] = { 0: [1], 1: [2, 4, 8, 16, 32], 2: [4, 8, 16, 32], 3: [-1], 4: [8, 16, 32], 5: [-1, 2], 6: [1, -2], 7: [-2, -4, 8, 16, 32], 8: [16, 32], 9: [-1, 2, 4], 10: [1, -2, 4], 11: [-2, 4, -8, 16, 32], 12: [1, 2, -4], 13: [2, -4, -8, 16, 32], 14: [-4, -8, 16, 32], 15: [-1, -2, -4], 16: [32], 17: [-1, 2, 4, 8], 18: [1, -2, 4, 8], 19: [-2, 4, 8, -16, 32], 20: [1, 2, -4, 8], 21: [2, -4, 8, -16, 32], 22: [-4, 8, -16, 32], 23: [-1, -2, -4, 8], 24: [1, 2, 4, -8], 25: [2, 4, -8, -16, 32], 26: [4, -8, -16, 32], 27: [-1, -2, 4, -8], 28: [-8, -16, 32], 29: [-1, 2, -4, -8], 30: [1, -2, -4, -8], 31: [-2, -4, -8, -16, 32], 32: [], 33: [-1, 2, 4, 8, 16], 34: [1, -2, 4, 8, 16], 35: [-2, 4, 8, 16, -32], 36: [1, 2, -4, 8, 16], 37: [2, -4, 8, 16, -32], 38: [-4, 8, 16, -32], 39: [-1, -2, -4, 8, 16], 40: [1, 2, 4, -8, 16], 41: [2, 4, -8, 16, -32], 42: [4, -8, 16, -32], 43: [-1, -2, 4, -8, 16], 44: [-8, 16, -32], 45: [-1, 2, -4, -8, 16], 46: [1, -2, -4, -8, 16], 47: [-2, -4, -8, 16, -32], 48: [1, 2, 4, 8, -16], 49: [2, 4, 8, -16, -32], 50: [4, 8, -16, -32], 51: [-1, -2, 4, 8, -16], 52: [8, -16, -32], 53: [-1, 2, -4, 8, -16], 54: [1, -2, -4, 8, -16], 55: [-2, -4, 8, -16, -32], 56: [-16, -32], 57: [-1, 2, 4, -8, -16], 58: [1, -2, 4, -8, -16], 59: [-2, 4, -8, -16, -32], 60: [1, 2, -4, -8, -16], 61: [2, -4, -8, -16, -32], 62: [-4, -8, -16, -32], 63: [-1, -2, -4, -8, -16] }  # noqa: E501

	dictionaryAddends4NextKnown: dict[int, list[int]] | None = None
	if state.leavesTotal == 64:
		dictionaryAddends4NextKnown = dictionaryAddends4NextKnown64

	pprint(dictionaryAddends4Next)
	print(len(dictionaryAddends4Next), "of", state.leavesTotal)

	if dictionaryAddends4NextKnown is not None:
		for leaf, listAddends in dictionaryAddends4Next.items():
			if listAddends != dictionaryAddends4NextKnown[leaf]:
				print(f"\33[91m{leaf = :2d}\t{listAddends = } != {dictionaryAddends4NextKnown[leaf]}, the known value.\33[0m")
		print("\33[92mChecked known values.\33[0m")

def verifyDictionaryAddends4Prior(state: EliminationState, dictionaryAddends4Prior: dict[int, list[int]]) -> None:
	dictionaryAddends4PriorKnown16: dict[int, list[int]] = { 0: [], 1: [-1], 2: [1], 3: [-2, 4, 8], 4: [1, 2], 5: [2, -4, 8], 6: [-4, 8], 7: [-1, -2], 8: [1, 2, 4], 9: [2, 4, -8], 10: [4, -8], 11: [-1, -2, 4], 12: [-8], 13: [-1, 2, -4], 14: [1, -2, -4], 15: [-2, -4, -8]}
	dictionaryAddends4PriorKnown32: dict[int, list[int]] = { 0: [], 1: [-1], 2: [1], 3: [-2, 4, 8, 16], 4: [1, 2], 5: [2, -4, 8, 16], 6: [-4, 8, 16], 7: [-1, -2], 8: [1, 2, 4], 9: [2, 4, -8, 16], 10: [4, -8, 16], 11: [-1, -2, 4], 12: [-8, 16], 13: [-1, 2, -4], 14: [1, -2, -4], 15: [-2, -4, -8, 16], 16: [1, 2, 4, 8], 17: [2, 4, 8, -16], 18: [4, 8, -16], 19: [-1, -2, 4, 8], 20: [8, -16], 21: [-1, 2, -4, 8], 22: [1, -2, -4, 8], 23: [-2, -4, 8, -16], 24: [-16], 25: [-1, 2, 4, -8], 26: [1, -2, 4, -8], 27: [-2, 4, -8, -16], 28: [1, 2, -4, -8], 29: [2, -4, -8, -16], 30: [-4, -8, -16], 31: [-1, -2, -4, -8]}  # noqa: E501
	dictionaryAddends4PriorKnown64: dict[int, list[int]] = { 0: [], 1: [-1], 2: [1], 3: [-2, 4, 8, 16, 32], 4: [1, 2], 5: [2, -4, 8, 16, 32], 6: [-4, 8, 16, 32], 7: [-1, -2], 8: [1, 2, 4], 9: [2, 4, -8, 16, 32], 10: [4, -8, 16, 32], 11: [-1, -2, 4], 12: [-8, 16, 32], 13: [-1, 2, -4], 14: [1, -2, -4], 15: [-2, -4, -8, 16, 32], 16: [1, 2, 4, 8], 17: [2, 4, 8, -16, 32], 18: [4, 8, -16, 32], 19: [-1, -2, 4, 8], 20: [8, -16, 32], 21: [-1, 2, -4, 8], 22: [1, -2, -4, 8], 23: [-2, -4, 8, -16, 32], 24: [-16, 32], 25: [-1, 2, 4, -8], 26: [1, -2, 4, -8], 27: [-2, 4, -8, -16, 32], 28: [1, 2, -4, -8], 29: [2, -4, -8, -16, 32], 30: [-4, -8, -16, 32], 31: [-1, -2, -4, -8], 32: [1, 2, 4, 8, 16], 33: [2, 4, 8, 16, -32], 34: [4, 8, 16, -32], 35: [-1, -2, 4, 8, 16], 36: [8, 16, -32], 37: [-1, 2, -4, 8, 16], 38: [1, -2, -4, 8, 16], 39: [-2, -4, 8, 16, -32], 40: [16, -32], 41: [-1, 2, 4, -8, 16], 42: [1, -2, 4, -8, 16], 43: [-2, 4, -8, 16, -32], 44: [1, 2, -4, -8, 16], 45: [2, -4, -8, 16, -32], 46: [-4, -8, 16, -32], 47: [-1, -2, -4, -8, 16], 48: [-32], 49: [-1, 2, 4, 8, -16], 50: [1, -2, 4, 8, -16], 51: [-2, 4, 8, -16, -32], 52: [1, 2, -4, 8, -16], 53: [2, -4, 8, -16, -32], 54: [-4, 8, -16, -32], 55: [-1, -2, -4, 8, -16], 56: [1, 2, 4, -8, -16], 57: [2, 4, -8, -16, -32], 58: [4, -8, -16, -32], 59: [-1, -2, 4, -8, -16], 60: [-8, -16, -32], 61: [-1, 2, -4, -8, -16], 62: [1, -2, -4, -8, -16], 63: [-2, -4, -8, -16, -32] }  # noqa: E501

	dictionaryAddends4PriorKnown: dict[int, list[int]] | None = None
	if state.leavesTotal == 16:
		dictionaryAddends4PriorKnown = dictionaryAddends4PriorKnown16
	elif state.leavesTotal == 32:
		dictionaryAddends4PriorKnown = dictionaryAddends4PriorKnown32
	elif state.leavesTotal == 64:
		dictionaryAddends4PriorKnown = dictionaryAddends4PriorKnown64

	pprint(dictionaryAddends4Prior)
	print(len(dictionaryAddends4Prior), "of", state.leavesTotal)

	if dictionaryAddends4PriorKnown is not None:
		for leaf, listAddends in dictionaryAddends4Prior.items():
			if listAddends != dictionaryAddends4PriorKnown[leaf]:
				print(f"\33[91m{leaf = :2d}\t{listAddends = } != {dictionaryAddends4PriorKnown[leaf]}, the known value.\33[0m")
		print("\33[92mChecked known values.\33[0m")

def verifyPinning2Dn(state: EliminationState) -> None:
	colorReset = '\33[0m'
	pathFilename: Path = Path(f"/apps/mapFolding/Z0Z_notes/arrayFoldingsP2d{state.dimensionsTotal}.pkl")
	arrayFoldings: NDArray[numpy.uint8] = pickle.loads(pathFilename.read_bytes())  # noqa: S301

	rowsTotal: int = int(arrayFoldings.shape[0])
	listMasks: list[numpy.ndarray] = []
	listDictionaryPinned: list[dict[int, int]] = []
	for pinnedLeaves in state.listPinnedLeaves:
		maskMatches: numpy.ndarray = numpy.ones(rowsTotal, dtype=bool)
		for indexPile, leaf in pinnedLeaves.items():
			maskMatches = maskMatches & (arrayFoldings[:, indexPile] == leaf)
		if not bool(maskMatches.any()):
			listDictionaryPinned.append(pinnedLeaves)
		listMasks.append(maskMatches)

	print("\33[93m", end='')
	pprint(listDictionaryPinned[0:2], width=140)
	print(colorReset, end='')
	print(len(listDictionaryPinned), "surplus dictionaries.")

	pathFilename = Path(f"/apps/mapFolding/Z0Z_notes/p2d{state.dimensionsTotal}SurplusDictionaries.csv")

	if listDictionaryPinned:
		with pathFilename.open('w', newline='') as writeStream:
			writerCSV = csv.writer(writeStream)
			listPiles: list[int] = list(range(state.leavesTotal))
			writerCSV.writerow(listPiles)
			for pinnedLeaves in listDictionaryPinned:
				writerCSV.writerow([pinnedLeaves.get(pile, '') for pile in listPiles])

	masksStacked: numpy.ndarray = numpy.column_stack(listMasks)
	coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
	indicesOverlappedRows: numpy.ndarray = numpy.nonzero(coverageCountPerRow >= 2)[0]
	if indicesOverlappedRows.size > 0:
		overlappingIndices: set[int] = set()
		for indexMask, mask in enumerate(listMasks):
			if bool(mask[indicesOverlappedRows].any()):
				overlappingIndices.add(indexMask)
		color = '\33[91m'
		print(f"{color}{len(overlappingIndices)} overlapping dictionaries", colorReset)
		for indexDictionary in sorted(overlappingIndices)[0:2]:
			pprint(state.listPinnedLeaves[indexDictionary], width=140)

	beansOrCornbread: Callable[[PinnedLeaves], bool] = beansWithoutCornbread(state)
	listBeans: list[PinnedLeaves] = list(filter(beansOrCornbread, state.listPinnedLeaves))
	if listBeans:
		color = '\33[95m'
		print(f"{color}{len(listBeans)} dictionaries with beans but no cornbread.", colorReset)
		pprint(listBeans[0], width=140)

	maskUnion = numpy.logical_or.reduce(listMasks)
	rowsCovered: int = int(maskUnion.sum())
	color = colorReset
	if rowsCovered < rowsTotal:
		color = '\33[91m'
		indicesMissingRows: numpy.ndarray = numpy.flatnonzero(~maskUnion)
		for indexRow in indicesMissingRows[0:2]:
			pprint(arrayFoldings[indexRow, :], width=140)
	print(f"{color}Covered rows: {rowsCovered}/{rowsTotal}{colorReset}")

def printStatisticsPermutations(state: EliminationState) -> None:
	dictionaryAddends4Next: dict[int, list[int]] = getDictionaryAddends4Next(state)
	permutationsTotal: Callable[[dict[int, list[int]]], int] = compose(prod, filter(None), dict[int, list[int]].values, valmap(len))
	permutationsPinnedLeaves: Callable[[dict[int, int]], int] = compose(permutationsTotal, lambda pinnedLeaves: dissoc(dictionaryAddends4Next, *pinnedLeaves.keys())) # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType, reportUnknownMemberType]
	permutationsPinnedLeavesTotal: Callable[[list[dict[int, int]]], int] = compose(sum, toolz_map(permutationsPinnedLeaves))

	print(fac(state.leavesTotal))
	print(permutationsTotal(dictionaryAddends4Next))
	print(permutationsPinnedLeavesTotal(state.listPinnedLeaves))

def verifyDictionaryPileToLeaves(state: EliminationState, dictionaryPileToLeaves: dict[int, list[int]]) -> None:
	dictionaryPileToLeavesKnown64 = {0: [0],
		1: [1],
		2: [3, 5, 9, 17, 33],
		3: [2, 7, 11, 13, 19, 21, 25, 35, 37, 41, 49],
		4: [3, 5, 6, 9, 10, 15, 17, 18, 23, 27, 29, 34, 39, 43, 45, 51, 53, 57],
		5: [2, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31, 35, 37, 38, 41, 42, 47, 49, 50, 55, 59, 61],
		6: [3, 5, 6, 9, 10, 15, 17, 18, 23, 27, 29, 30, 33, 34, 39, 43, 45, 46, 51, 53, 54, 57, 58, 63],
		7: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31, 35, 37, 38, 41, 42, 47, 49, 50, 55, 59, 61, 62],
		8: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30, 34, 36, 39, 43, 45, 46, 51, 53, 54, 57, 58, 63],
		9: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 59, 61, 62],
		10: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30, 33, 34, 36, 39, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63],
		11: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 59, 61, 62],
		12: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30, 34, 36, 39, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63],
		13: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 59, 61, 62],
		14: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30, 33, 34, 36, 39, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63],
		15: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 59, 61, 62],
		16: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 34, 36, 39, 40, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63],
		17: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		18: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63],
		19: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		20: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 34, 36, 39, 40, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63],
		21: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		22: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63],
		23: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		24: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 34, 36, 39, 40, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63],
		25: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		26: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63],
		27: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		28: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 34, 36, 39, 40, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63],
		29: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		30: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63],
		31: [2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		32: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		33: [2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		34: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		35: [4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		36: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		37: [4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		38: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		39: [4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		40: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		41: [4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		42: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		43: [4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		44: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		45: [4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		46: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		47: [4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		48: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		49: [4, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		50: [9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		51: [8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		52: [9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		53: [8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		54: [9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		55: [8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		56: [9, 10, 12, 17, 18, 20, 23, 24, 27, 29, 30, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		57: [8, 16, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		58: [17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60, 63],
		59: [16, 19, 21, 22, 25, 26, 28, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62],
		60: [17, 18, 20, 24, 34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 54, 57, 58, 60],
		61: [16, 35, 37, 38, 41, 42, 44, 49, 50, 52, 56],
		62: [33, 34, 36, 40, 48],
		63: [32]}

	dictionaryPileToLeavesKnown32 = {0: [0],
		1: [1],
		2: [3, 5, 9, 17],
		3: [2, 7, 11, 13, 19, 21, 25],
		4: [3, 5, 6, 9, 10, 15, 18, 23, 27, 29],
		5: [2, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31],
		6: [3, 5, 6, 9, 10, 15, 17, 18, 23, 27, 29, 30],
		7: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31],
		8: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 27, 29, 30],
		9: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
		10: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30],
		11: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
		12: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 27, 29, 30],
		13: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
		14: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30],
		15: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
		16: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
		17: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
		18: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
		19: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
		20: [5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
		21: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
		22: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
		23: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
		24: [5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
		25: [4, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
		26: [9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
		27: [8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
		28: [9, 10, 12, 18, 20, 23, 24, 27, 29, 30],
		29: [8, 19, 21, 22, 25, 26, 28],
		30: [17, 18, 20, 24],
		31: [16]}

	dictionaryPileToLeavesKnown16 = {0: [0],
		1: [1],
		2: [3, 5, 9],
		3: [2, 7, 11, 13],
		4: [3, 5, 6, 10, 15],
		5: [2, 7, 11, 13, 14],
		6: [3, 5, 6, 9, 10, 15],
		7: [2, 4, 7, 11, 13, 14],
		8: [3, 5, 6, 10, 12, 15],
		9: [2, 4, 7, 11, 13, 14],
		10: [5, 6, 9, 10, 12, 15],
		11: [4, 7, 11, 13, 14],
		12: [5, 6, 10, 12, 15],
		13: [4, 11, 13, 14],
		14: [9, 10, 12],
		15: [8]}

	dictionaryPileToLeavesKnown: dict[int, list[int]] | None = None
	if state.leavesTotal == 16:
		dictionaryPileToLeavesKnown = dictionaryPileToLeavesKnown16
	elif state.leavesTotal == 32:
		dictionaryPileToLeavesKnown = dictionaryPileToLeavesKnown32
	elif state.leavesTotal == 64:
		dictionaryPileToLeavesKnown = dictionaryPileToLeavesKnown64

	pprint(dictionaryPileToLeaves, width=140, compact=True)
	print(len(dictionaryPileToLeaves), "of", state.leavesTotal)

	if dictionaryPileToLeavesKnown is not None:
		for pile, listLeaves in dictionaryPileToLeaves.items():
			if listLeaves != dictionaryPileToLeavesKnown[pile]:
				print(f"\33[91m{pile = :2d}\t{listLeaves = } != {dictionaryPileToLeavesKnown[pile]}, the known value.\33[0m")
		print("\33[92mChecked known values.\33[0m")
