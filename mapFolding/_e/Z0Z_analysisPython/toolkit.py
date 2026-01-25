from collections.abc import Callable, Sequence
from cytoolz.dicttoolz import valfilter as leafFilter
from dataclasses import dataclass
from mapFolding import ansiColorReset, ansiColors, packageSettings
from mapFolding._e import PermutationSpace, thisIsALeaf
from mapFolding._e._dataDynamic import getDataFrameFoldings
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pin2上nDimensionsAnnex import beansWithoutCornbread
from pathlib import Path
from pprint import pformat
import csv
import numpy
import sys

@dataclass
class PermutationSpaceStatus:
	listSurplusDictionaries: list[PermutationSpace]
	maskUnion: numpy.ndarray
	indicesOverlappingRows: numpy.ndarray
	indicesOverlappingPermutationSpace: set[int]
	rowsRequired: int
	rowsTotal: int

def detectPermutationSpaceErrors(arrayFoldings: numpy.ndarray, listPermutationSpace: Sequence[PermutationSpace]) -> PermutationSpaceStatus:
	rowsTotal: int = int(arrayFoldings.shape[0])
	listMasks: list[numpy.ndarray] = []
	listSurplusDictionaries: list[PermutationSpace] = []
	for permutationSpace in listPermutationSpace:
		maskMatches: numpy.ndarray = numpy.ones(rowsTotal, dtype=bool)
		for pile, leaf in leafFilter(thisIsALeaf, permutationSpace).items():
			maskMatches = maskMatches & (arrayFoldings[:, pile] == leaf)
		if not bool(maskMatches.any()):
			listSurplusDictionaries.append(permutationSpace)
		listMasks.append(maskMatches)

	if listMasks:
		masksStacked: numpy.ndarray = numpy.column_stack(listMasks)
	else:
		masksStacked = numpy.zeros((rowsTotal, 0), dtype=bool)

	coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
	maskUnion: numpy.ndarray = coverageCountPerRow > 0
	rowsRequired: int = int(maskUnion.sum())
	indicesOverlappingRows: numpy.ndarray = numpy.flatnonzero(coverageCountPerRow >= 2)
	indicesOverlappingPermutationSpace: set[int] = set()
	if indicesOverlappingRows.size > 0:
		for indexMask, mask in enumerate(listMasks):
			if bool(mask[indicesOverlappingRows].any()):
				indicesOverlappingPermutationSpace.add(indexMask)

	return PermutationSpaceStatus(listSurplusDictionaries, maskUnion, indicesOverlappingRows, indicesOverlappingPermutationSpace, rowsRequired, rowsTotal)

def verifyPinning2Dn(state: EliminationState) -> None:
	def getPermutationSpaceWithLeafValuesOnly(permutationSpace: PermutationSpace) -> PermutationSpace:
		return leafFilter(thisIsALeaf, permutationSpace)
	arrayFoldings = getDataFrameFoldings(state)
	if arrayFoldings is not None:
		arrayFoldings = arrayFoldings.to_numpy(dtype=numpy.uint8, copy=False)
		pinningCoverage: PermutationSpaceStatus = detectPermutationSpaceErrors(arrayFoldings, state.listPermutationSpace)

		listSurplusDictionariesOriginal: list[PermutationSpace] = pinningCoverage.listSurplusDictionaries
		listDictionaryPinned: list[PermutationSpace] = [
			getPermutationSpaceWithLeafValuesOnly(permutationSpace)
			for permutationSpace in listSurplusDictionariesOriginal
		]
		if listDictionaryPinned:
			sys.stdout.write(ansiColors.YellowOnBlack)
			sys.stdout.write(pformat(listDictionaryPinned[0:5], width=140) + '\n')
		else:
			sys.stdout.write(ansiColors.GreenOnBlack)
		sys.stdout.write(f"{len(listDictionaryPinned)} surplus dictionaries.\n")
		sys.stdout.write(ansiColorReset)

		pathFilename = Path(f"{packageSettings.pathPackage}/_e/analysisExcel/p2d{state.dimensionsTotal}SurplusDictionaries.csv")

		if listDictionaryPinned:
			with pathFilename.open('w', newline='') as writeStream:
				writerCSV = csv.writer(writeStream)
				listPiles: list[int] = list(range(state.leavesTotal))
				writerCSV.writerow(listPiles)
				for permutationSpace in listDictionaryPinned:
					writerCSV.writerow([permutationSpace.get(pile, '') for pile in listPiles])

		if pinningCoverage.indicesOverlappingPermutationSpace:
			sys.stdout.write(f"{ansiColors.RedOnWhite}{len(pinningCoverage.indicesOverlappingPermutationSpace)} overlapping dictionaries{ansiColorReset}\n")
			for indexDictionary in sorted(pinningCoverage.indicesOverlappingPermutationSpace)[0:2]:
				sys.stdout.write(pformat(leafFilter(thisIsALeaf, state.listPermutationSpace[indexDictionary]), width=140) + '\n')

		beansOrCornbread: Callable[[PermutationSpace], bool] = beansWithoutCornbread(state)
		listBeans: list[PermutationSpace] = list(filter(beansOrCornbread, state.listPermutationSpace))
		if listBeans:
			sys.stdout.write(f"{ansiColors.MagentaOnBlack}{len(listBeans)} dictionaries with beans but no cornbread.{ansiColorReset}\n")
			sys.stdout.write(pformat(getPermutationSpaceWithLeafValuesOnly(listBeans[0]), width=140) + '\n')

		maskUnion: numpy.ndarray = pinningCoverage.maskUnion
		rowsRequired: int = pinningCoverage.rowsRequired
		rowsTotal: int = pinningCoverage.rowsTotal
		color = ansiColorReset
		if rowsRequired < rowsTotal:
			color = ansiColors.RedOnWhite
			indicesMissingRows: numpy.ndarray = numpy.flatnonzero(~maskUnion)
			for indexRow in indicesMissingRows[0:2]:
				sys.stdout.write(f"{color}{arrayFoldings[indexRow, :]}\n")
		sys.stdout.write(f"{color}Required rows: {rowsRequired}/{rowsTotal}{ansiColorReset}\n")

def verifyDomainAgainstKnown(domainComputed: Sequence[tuple[int, ...]], domainKnown: Sequence[tuple[int, ...]], *, printResults: bool = True) -> dict[str, list[tuple[int, ...]]]:
	"""Compare a computed domain against known verification data.

	Parameters
	----------
	domainComputed : Sequence[tuple[int, ...]]
		The domain generated by the function under development.
	domainKnown : Sequence[tuple[int, ...]]
		The empirically extracted domain from verification data (e.g., from `makeVerificationDataLeavesDomain`).
	printResults : bool = True
		Whether to print the comparison results using pprint.

	Returns
	-------
	comparisonResults : dict[str, list[tuple[int, ...]]]
		Dictionary with keys:
		- 'missing': tuples in domainKnown but not in domainComputed (the function fails to generate these)
		- 'surplus': tuples in domainComputed but not in domainKnown (the function generates extra invalid tuples)
		- 'matched': tuples present in both domains

	"""
	setComputed: set[tuple[int, ...]] = set(domainComputed)
	setKnown: set[tuple[int, ...]] = set(domainKnown)

	listMissing: list[tuple[int, ...]] = sorted(setKnown - setComputed)
	listSurplus: list[tuple[int, ...]] = sorted(setComputed - setKnown)
	listMatched: list[tuple[int, ...]] = sorted(setComputed & setKnown)

	comparisonResults: dict[str, list[tuple[int, ...]]] = {
		'missing': listMissing,
		'surplus': listSurplus,
		'matched': listMatched,
	}

	if printResults:
		countComputed: int = len(setComputed)
		countKnown: int = len(setKnown)
		countMissing: int = len(listMissing)
		countSurplus: int = len(listSurplus)
		countMatched: int = len(listMatched)

		sys.stdout.write(f"Domain comparison: {countComputed} computed vs {countKnown} known\n")
		sys.stdout.write(f"  Matched: {countMatched} ({100 * countMatched / countKnown:.1f}% of known)\n")

		if listMissing:
			sys.stdout.write(f"  Missing ({countMissing} tuples in known but not in computed):\n")
			sys.stdout.write(pformat(listMissing, width=140, compact=True) + '\n')

		if listSurplus:
			sys.stdout.write(f"  Surplus ({countSurplus} tuples in computed but not in known):\n")
			sys.stdout.write(pformat(listSurplus, width=140, compact=True) + '\n')

		if not listMissing and not listSurplus:
			sys.stdout.write("  Perfect match!\n")

	return comparisonResults
