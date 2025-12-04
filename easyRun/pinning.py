# ruff: noqa: T201, T203, D100, D103
# pyright: reportUnusedImport=false
from cytoolz.curried import map as toolz_map, valfilter, valmap
from cytoolz.dicttoolz import dissoc
from cytoolz.functoolz import compose
from gmpy2 import fac
from itertools import accumulate
from mapFolding import packageSettings
from mapFolding._e import (
	getDictionaryLeafDomains, getDictionaryPileRanges, getDomainDimension二, getDomain二一零and二一, getDomain二零and二,
	getLeafDomain, getListLeavesIncrease, getPileRange, PinnedLeaves, 一, 二, 零)
from mapFolding._e.pinning2Dn import pinLeaf首零_零, secondOrderLeaves, secondOrderPilings, thirdOrderPilings, Z0Z_k_r
from mapFolding._e.pinning2DnAnnex import beansWithoutCornbread
from mapFolding.dataBaskets import EliminationState
from math import prod
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING
import csv
import numpy
import pickle
import time

if TYPE_CHECKING:
	from collections.abc import Callable
	from numpy.typing import NDArray

def verifyPinning2Dn(state: EliminationState) -> None:
	colorReset = '\33[0m'
	pathFilename = Path(f'{packageSettings.pathPackage}/tests/dataSamples/arrayFoldingsP2d{state.dimensionsTotal}.pkl')
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
	pprint(listDictionaryPinned[0:5], width=140)
	print(colorReset, end='')
	print(len(listDictionaryPinned), "surplus dictionaries.")

	pathFilename = Path(f"{packageSettings.pathPackage}/_e/analysisExcel/p2d{state.dimensionsTotal}SurplusDictionaries.csv")

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
			print(color, arrayFoldings[indexRow, :])
	print(f"{color}Covered rows: {rowsCovered}/{rowsTotal}{colorReset}")

def printStatisticsPermutations(state: EliminationState) -> None:
	dictionaryPileRanges: dict[int, list[int]] = getDictionaryPileRanges(state)
	permutationsTotal: Callable[[dict[int, list[int]]], int] = compose(prod, dict[int, list[int]].values, valmap(len), valfilter(bool))
	def stripped(pinnedLeaves: PinnedLeaves) -> dict[int, list[int]]:
		return dissoc(dictionaryPileRanges, *pinnedLeaves.keys())
	permutationsPinnedLeaves: Callable[[dict[int, int]], int] = compose(permutationsTotal, stripped)
	permutationsPinnedLeavesTotal: Callable[[list[dict[int, int]]], int] = compose(sum, toolz_map(permutationsPinnedLeaves))

	print(fac(state.leavesTotal))
	print(permutationsTotal(dictionaryPileRanges))
	print(permutationsPinnedLeavesTotal(state.listPinnedLeaves))

if __name__ == '__main__':
	state = EliminationState((2,) * 5)

	printThis = True

	if printThis:
		timeStart = time.perf_counter()
		state: EliminationState = secondOrderLeaves(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tsecondOrderLeaves")
		verifyPinning2Dn(state)
		print(f"{time.perf_counter() - timeStart:.2f}\tverifyPinning2Dn")
		printStatisticsPermutations(state)
		print(f"{len(state.listPinnedLeaves)=}")
		dictionaryLeafDomains = getDictionaryLeafDomains(state)
		pprint(dictionaryLeafDomains)

	elif printThis:
		state: EliminationState = secondOrderPilings(state)
		state: EliminationState = thirdOrderPilings(state)
		# print(f"{time.perf_counter() - timeStart:.2f}\tthirdOrderPilings")  # noqa: ERA001
		state: EliminationState = pinLeaf首零_零(state)
		dictionaryPileRanges = getDictionaryPileRanges(state)
		domainsOfDimensionOrigins = tuple(getLeafDomain(state, leaf) for leaf in state.productsOfDimensions)[0:-1]
		pprint(state.listPinnedLeaves)
		pprint(state.listPinnedLeaves[0:5])
		print(*domainsOfDimensionOrigins, sep='\n')
		state: EliminationState = Z0Z_k_r(state)


		sumsOfDimensionOrigins = tuple(accumulate(state.productsOfDimensions))[0:-1]
		sumsOfDimensionOriginsReversed = tuple(accumulate(state.productsOfDimensions[::-1], initial=-state.leavesTotal))[1:None]

		for dimensionOrigin, domain, sumOrigins, sumReversed in zip(state.productsOfDimensions, domainsOfDimensionOrigins, sumsOfDimensionOrigins, sumsOfDimensionOriginsReversed, strict=False):
			print(f"{dimensionOrigin:<2}\t{domain.start == sumOrigins = }\t{sumOrigins}\t{sumReversed+2}\t{domain.stop == sumReversed+2 = }")

		print(*sumsOfDimensionOrigins, sep='\n')
		print(*sumsOfDimensionOriginsReversed, sep='\n')
