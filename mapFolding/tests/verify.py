# ruff: noqa: T201
from cytoolz.curried import dissoc, filter, map as toolz_map, valmap  # noqa: A004
from cytoolz.functoolz import compose
from gmpy2 import fac
from mapFolding.algorithms.patternFinder import getDictionaryDifferences
from mapFolding.dataBaskets import EliminationState
from math import prod
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING
import numpy
import pickle

if TYPE_CHECKING:
	from collections.abc import Callable
	from numpy.typing import NDArray

def verifyDictionaryLeafRanges(state: EliminationState, dictionaryLeafRanges: dict[int, range]) -> None:
	known32: dict[int, int] = {0: 0, 1: 1, 2: 3, 3: 2, 4: 7, 5: 2, 6: 4, 7: 3, 8: 15, 9: 2, 10: 4, 11: 3, 12: 8, 13: 3, 14: 5, 15: 4, 16: 31, 17: 2, 18: 4, 19: 3, 20: 8, 21: 3, 22: 5, 23: 4, 24: 16, 25: 3, 26: 5, 27: 4, 28: 9, 29: 4, 30: 6, 31: 5}
	known64: dict[int, int] = {}
	known64columns: list[int] = [0,1,3,2,7,2,4,3,15,2,4,3,8,3,5,4,31,2,4,3,8,3,5,4,16,3,5,4,9,4,6,5,63,2,4,3,8,3,5,4,16,3,5,4,9,4,6,5,32,3,5,4,9,4,6,5,17,4,6,5,10,5,7,6]
	known64 = {indexLeaf: known64columns[indexLeaf] for indexLeaf in range(64)}
	knownStop64columns: list[int] = [1,2,34,33,50,49,49,48,58,57,57,56,57,56,56,55,62,61,61,60,61,60,60,59,61,60,60,59,60,59,59,58,64,63,63,62,63,62,62,61,63,62,62,61,62,61,61,60,63,62,62,61,62,61,61,60,62,61,61,60,61,60,60,59]
	knownStop64 = {indexLeaf: knownStop64columns[indexLeaf] for indexLeaf in range(64)}

	knownColumnStart = None
	knownColumnStop = None
	if state.leavesTotal == 32:
		knownColumnStart = known32
	elif state.leavesTotal == 64:
		knownColumnStart = known64
		knownColumnStop = knownStop64

	issues: int = 0

	for indexLeaf, rangeLeaf in sorted(dictionaryLeafRanges.items()):
		print(indexLeaf, repr(rangeLeaf), sep="\t")
		if knownColumnStart is not None and rangeLeaf.start != knownColumnStart[indexLeaf]:
			issues += 1
			print(f"\33[91mKnown column start: {knownColumnStart[indexLeaf]:2d}\33[0m")

		if knownColumnStop is not None:
			if rangeLeaf.stop < knownColumnStop[indexLeaf]:
				issues += 1
				print(f"\33[91mKnown column stop: {knownColumnStop[indexLeaf]:2d}\33[0m")
			if rangeLeaf.stop > knownColumnStop[indexLeaf] + 1:
				issues += 1
				print(f"Known column stop: {knownColumnStop[indexLeaf]:2d}\33[0m")

	if issues:
		print(f"\33[91mFound {issues} issues.\33[0m")
	else:
		print("\33[92mNo issues found.\33[0m")
	print(len(dictionaryLeafRanges), "of", state.leavesTotal)

def verifyDictionaryDifferences(state: EliminationState, dictionaryDifferences: dict[int, list[int]]) -> None:
	dictionaryDifferencesKnown64: dict[int, list[int]] = {
	0: [1],
	1: [2, 4, 8, 16, 32],
	2: [4, 8, 16, 32],
	3: [-1],
	4: [8, 16, 32],
	5: [-1, 2],
	6: [1, -2],
	7: [-2, -4, 8, 16, 32],
	8: [16, 32],
	9: [-1, 2, 4],
	10: [1, -2, 4],
	11: [-2, 4, -8, 16, 32],
	12: [1, 2, -4],
	13: [2, -4, -8, 16, 32],
	14: [-4, -8, 16, 32],
	15: [-1, -2, -4],
	16: [32],
	17: [-1, 2, 4, 8],
	18: [1, -2, 4, 8],
	19: [-2, 4, 8, -16, 32],
	20: [1, 2, -4, 8],
	21: [2, -4, 8, -16, 32],
	22: [-4, 8, -16, 32],
	23: [-1, -2, -4, 8],
	24: [1, 2, 4, -8],
	25: [2, 4, -8, -16, 32],
	26: [4, -8, -16, 32],
	27: [-1, -2, 4, -8],
	28: [-8, -16, 32],
	29: [-1, 2, -4, -8],
	30: [1, -2, -4, -8],
	31: [-2, -4, -8, -16, 32],
	32: [],
	33: [-1, 2, 4, 8, 16],
	34: [1, -2, 4, 8, 16],
	35: [-2, 4, 8, 16, -32],
	36: [1, 2, -4, 8, 16],
	37: [2, -4, 8, 16, -32],
	38: [-4, 8, 16, -32],
	39: [-1, -2, -4, 8, 16],
	40: [1, 2, 4, -8, 16],
	41: [2, 4, -8, 16, -32],
	42: [4, -8, 16, -32],
	43: [-1, -2, 4, -8, 16],
	44: [-8, 16, -32],
	45: [-1, 2, -4, -8, 16],
	46: [1, -2, -4, -8, 16],
	47: [-2, -4, -8, 16, -32],
	48: [1, 2, 4, 8, -16],
	49: [2, 4, 8, -16, -32],
	50: [4, 8, -16, -32],
	51: [-1, -2, 4, 8, -16],
	52: [8, -16, -32],
	53: [-1, 2, -4, 8, -16],
	54: [1, -2, -4, 8, -16],
	55: [-2, -4, 8, -16, -32],
	56: [-16, -32],
	57: [-1, 2, 4, -8, -16],
	58: [1, -2, 4, -8, -16],
	59: [-2, 4, -8, -16, -32],
	60: [1, 2, -4, -8, -16],
	61: [2, -4, -8, -16, -32],
	62: [-4, -8, -16, -32],
	63: [-1, -2, -4, -8, -16]
}

	pprint(dictionaryDifferences)  # noqa: T203
	print(len(dictionaryDifferences), "of", state.leavesTotal)

	for indexLeaf, listDifferences in dictionaryDifferences.items():
		if listDifferences != dictionaryDifferencesKnown64[indexLeaf]:
			print(f"\33[91m{indexLeaf = :2d}\t{listDifferences = } != {dictionaryDifferencesKnown64[indexLeaf]}, the known value.\33[0m")
	print("\33[92mChecked known values.\33[0m")

def verifyDictionaryDifferencesReverse(state: EliminationState, dictionaryDifferencesReverse: dict[int, list[int]]) -> None:
	dictionaryDifferencesReverseKnown64: dict[int, list[int]] = {
	1: [-1],
	2: [1],
	3: [-2, 4, 8, 16, 32],
	4: [1, 2],
	5: [2, -4, 8, 16, 32],
	6: [-4, 8, 16, 32],
	7: [-1, -2],
	8: [1, 2, 4],
	9: [2, 4, -8, 16, 32],
	10: [4, -8, 16, 32],
	11: [-1, -2, 4],
	12: [-8, 16, 32],
	13: [-1, 2, -4],
	14: [1, -2, -4],
	15: [-2, -4, -8, 16, 32],
	16: [1, 2, 4, 8],
	17: [2, 4, 8, -16, 32],
	18: [4, 8, -16, 32],
	19: [-1, -2, 4, 8],
	20: [8, -16, 32],
	21: [-1, 2, -4, 8],
	22: [1, -2, -4, 8],
	23: [-2, -4, 8, -16, 32],
	24: [-16, 32],
	25: [-1, 2, 4, -8],
	26: [1, -2, 4, -8],
	27: [-2, 4, -8, -16, 32],
	28: [1, 2, -4, -8],
	29: [2, -4, -8, -16, 32],
	30: [-4, -8, -16, 32],
	31: [-1, -2, -4, -8],
	32: [1, 2, 4, 8, 16],
	33: [2, 4, 8, 16, -32],
	34: [4, 8, 16, -32],
	35: [-1, -2, 4, 8, 16],
	36: [8, 16, -32],
	37: [-1, 2, -4, 8, 16],
	38: [1, -2, -4, 8, 16],
	39: [-2, -4, 8, 16, -32],
	40: [16, -32],
	41: [-1, 2, 4, -8, 16],
	42: [1, -2, 4, -8, 16],
	43: [-2, 4, -8, 16, -32],
	44: [1, 2, -4, -8, 16],
	45: [2, -4, -8, 16, -32],
	46: [-4, -8, 16, -32],
	47: [-1, -2, -4, -8, 16],
	48: [-32],
	49: [-1, 2, 4, 8, -16],
	50: [1, -2, 4, 8, -16],
	51: [-2, 4, 8, -16, -32],
	52: [1, 2, -4, 8, -16],
	53: [2, -4, 8, -16, -32],
	54: [-4, 8, -16, -32],
	55: [-1, -2, -4, 8, -16],
	56: [1, 2, 4, -8, -16],
	57: [2, 4, -8, -16, -32],
	58: [4, -8, -16, -32],
	59: [-1, -2, 4, -8, -16],
	60: [-8, -16, -32],
	61: [-1, 2, -4, -8, -16],
	62: [1, -2, -4, -8, -16],
	63: [-2, -4, -8, -16, -32]
}

	pprint(dictionaryDifferencesReverse)  # noqa: T203
	print(len(dictionaryDifferencesReverse), "of", state.leavesTotal)

	for indexLeaf, listDifferences in dictionaryDifferencesReverse.items():
		if listDifferences != dictionaryDifferencesReverseKnown64[indexLeaf]:
			print(f"\33[91m{indexLeaf = :2d}\t{listDifferences = } != {dictionaryDifferencesReverseKnown64[indexLeaf]}, the known value.\33[0m")
	print("\33[92mChecked known values.\33[0m")

def verifyPinning2Dn(state: EliminationState) -> None:
	colorReset = '\33[0m'
	pathFilename: Path = Path("/apps/mapFolding/Z0Z_notes/arrayFoldingsP2d6.pkl")
	arrayFoldingsP2d6: NDArray[numpy.uint8] = pickle.loads(pathFilename.read_bytes())  # noqa: S301

	rowsTotal: int = int(arrayFoldingsP2d6.shape[0])
	listMasks: list[numpy.ndarray] = []
	listDictionaryPinned: list[dict[int, int]] = []
	for dictionaryPinned in state.listPinnedLeaves:
		maskMatches: numpy.ndarray = numpy.ones(rowsTotal, dtype=bool)
		for indexPile, indexLeaf in dictionaryPinned.items():
			maskMatches = maskMatches & (arrayFoldingsP2d6[:, indexPile] == indexLeaf)
		if not bool(maskMatches.any()):
			print(f"\33[93m{(dictionaryPinned)}\33[0m")
			listDictionaryPinned.append(dictionaryPinned)
		listMasks.append(maskMatches)

	maskUnion = numpy.logical_or.reduce(listMasks)
	rowsCovered: int = int(maskUnion.sum())
	color = colorReset
	if rowsCovered < rowsTotal:
		color = '\33[91m'
		indicesMissingRows: numpy.ndarray = numpy.flatnonzero(~maskUnion)
		for indexRow in indicesMissingRows[0:1]:
			print(arrayFoldingsP2d6[indexRow, :])
	print(f"{color}Covered rows: {rowsCovered}/{rowsTotal}{colorReset}")

	masksStacked: numpy.ndarray = numpy.column_stack(listMasks)
	coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
	indicesOverlappedRows: numpy.ndarray = numpy.nonzero(coverageCountPerRow >= 2)[0]
	if indicesOverlappedRows.size > 0:
		overlappingIndices: set[int] = set()
		for indexMask, mask in enumerate(listMasks):
			if bool(mask[indicesOverlappedRows].any()):
				overlappingIndices.add(indexMask)
		for indexDictionary in sorted(overlappingIndices):
			print("Overlapping", state.listPinnedLeaves[indexDictionary])

def printStatisticsPermutations(state: EliminationState) -> None:
	dictionaryDifferences: dict[int, list[int]] = getDictionaryDifferences(state)
	permutationsTotal: Callable[[dict[int, list[int]]], int] = compose(prod, filter(None), dict[int, list[int]].values, valmap(len))
	permutationsPinnedLeaves: Callable[[dict[int, int]], int] = compose(permutationsTotal, lambda pinnedLeaves: dissoc(dictionaryDifferences, *pinnedLeaves.keys())) # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType, reportUnknownMemberType]
	permutationsPinnedLeavesTotal: Callable[[list[dict[int, int]]], int] = compose(sum, toolz_map(permutationsPinnedLeaves))

	print(fac(64))
	print(permutationsTotal(dictionaryDifferences))
	print(permutationsPinnedLeavesTotal(state.listPinnedLeaves))
