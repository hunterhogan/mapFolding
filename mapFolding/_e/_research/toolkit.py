from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from humpy_cytoolz import valfilter as filterLeaf
from mapFolding import ansiColorReset, ansiColors
from mapFolding._e.filters import leaf吗
from mapFolding._e.tests.test_pinning import beansWithoutCornbread
from mapFolding.kitFilesystem import getDataFrameFoldings
from mapFolding.theSSOT import settingsPackage
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING
import csv
import numpy
import sys

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence
	from mapFolding._e.dataBaskets import PermutationSpace, StateElimination
	from mapFolding._e.theTypes import PinnedLeaves

@dataclass
class PermutationSpaceStatus:
	boxOfSurplusDictionaries: list[PermutationSpace]
	maskUnion: numpy.ndarray
	indicesOverlappingRows: numpy.ndarray
	indicesOverlappingPermutationSpace: set[int]
	rowsRequired: int
	rowsTotal: int

def detectPermutationSpaceErrors(arrayFoldings: numpy.ndarray, boxOfPermutationSpace: Sequence[PermutationSpace]) -> PermutationSpaceStatus:
	rowsTotal: int = int(arrayFoldings.shape[0])
	boxOfMasks: list[numpy.ndarray] = []
	boxOfSurplusDictionaries: list[PermutationSpace] = []
	for permutationSpace in boxOfPermutationSpace:
		maskMatches: numpy.ndarray = numpy.ones(rowsTotal, dtype=bool)
		for pile, leaf in filterLeaf(leaf吗, permutationSpace).items():
			maskMatches &= (arrayFoldings[:, pile] == leaf)
		if not bool(maskMatches.any()):
			boxOfSurplusDictionaries.append(permutationSpace)
		boxOfMasks.append(maskMatches)

	if boxOfMasks:
		masksStacked: numpy.ndarray = numpy.column_stack(boxOfMasks)
	else:
		masksStacked = numpy.zeros((rowsTotal, 0), dtype=bool)

	coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
	maskUnion: numpy.ndarray = coverageCountPerRow > 0
	rowsRequired: int = int(maskUnion.sum())
	indicesOverlappingRows: numpy.ndarray = numpy.flatnonzero(coverageCountPerRow >= 2)
	indicesOverlappingPermutationSpace: set[int] = set()
	if indicesOverlappingRows.size > 0:
		for 次Mask, mask in enumerate(boxOfMasks):
			if bool(mask[indicesOverlappingRows].any()):
				indicesOverlappingPermutationSpace.add(次Mask)

	return PermutationSpaceStatus(boxOfSurplusDictionaries, maskUnion, indicesOverlappingRows, indicesOverlappingPermutationSpace, rowsRequired, rowsTotal)

#======== Specialized tools ===============================

def verifyPinning2Dn(state: StateElimination) -> None:
	def getPermutationSpaceWithLeafValuesOnly(permutationSpace: PermutationSpace) -> PinnedLeaves:
		return permutationSpace.pinnedLeaves()
	arrayFoldings = getDataFrameFoldings(state)
	if arrayFoldings is not None:
		arrayFoldings = arrayFoldings.to_numpy(dtype=numpy.uint8, copy=False)
		pinningCoverage: PermutationSpaceStatus = detectPermutationSpaceErrors(arrayFoldings, state.boxOfPermutationSpace)

		boxOfSurplusDictionariesOriginal: list[PermutationSpace] = pinningCoverage.boxOfSurplusDictionaries
		boxOfDictionaryPinned: list[PinnedLeaves] = [
			getPermutationSpaceWithLeafValuesOnly(permutationSpace)
			for permutationSpace in boxOfSurplusDictionariesOriginal
		]
		if boxOfDictionaryPinned:
			sys.stdout.write(ansiColors.YellowOnBlack)
			sys.stdout.write(pformat(boxOfDictionaryPinned[0:5], width=200) + '\n')
		else:
			sys.stdout.write(ansiColors.GreenOnBlack)
		sys.stdout.write(f"{len(boxOfDictionaryPinned)} surplus dictionaries.\n")
		sys.stdout.write(ansiColorReset)

		pathFilename = Path(f"{settingsPackage.pathPackage}/_e/_development/excel/p2d{state.totalDimensions}SurplusDictionaries.csv")

		if boxOfDictionaryPinned:
			with pathFilename.open('w', encoding='utf-8', newline='') as writeStream:
				writerCSV = csv.writer(writeStream)
				boxOfPiles: list[int] = list(range(state.totalLeaves))
				writerCSV.writerow(boxOfPiles)
				for permutationSpace in boxOfDictionaryPinned:
					writerCSV.writerow([permutationSpace.get(pile, '') for pile in boxOfPiles])

		if pinningCoverage.indicesOverlappingPermutationSpace:
			sys.stdout.write(f"{ansiColors.RedOnWhite}{len(pinningCoverage.indicesOverlappingPermutationSpace)} overlapping dictionaries{ansiColorReset}\n")
			for 次Dictionary in sorted(pinningCoverage.indicesOverlappingPermutationSpace)[0:2]:
				sys.stdout.write(pformat(filterLeaf(leaf吗, state.boxOfPermutationSpace[次Dictionary]), width=140) + '\n')

		beansOrCornbread: Callable[[PermutationSpace], bool] = partial(beansWithoutCornbread, state)
		boxOfBeans: list[PermutationSpace] = list(filter(beansOrCornbread, state.boxOfPermutationSpace))
		if boxOfBeans:
			sys.stdout.write(f"{ansiColors.MagentaOnBlack}{len(boxOfBeans)} dictionaries with beans but no cornbread.{ansiColorReset}\n")
			sys.stdout.write(pformat(getPermutationSpaceWithLeafValuesOnly(boxOfBeans[0]), width=140) + '\n')

		maskUnion: numpy.ndarray = pinningCoverage.maskUnion
		rowsRequired: int = pinningCoverage.rowsRequired
		rowsTotal: int = pinningCoverage.rowsTotal
		color = ansiColorReset
		if rowsRequired < rowsTotal:
			color = ansiColors.RedOnWhite
			indicesMissingRows: numpy.ndarray = numpy.flatnonzero(~maskUnion)
			for 次Row in indicesMissingRows[0:2]:
				sys.stdout.write(f"{color}{arrayFoldings[次Row, :]}\n")
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
	boxOfComputed: set[tuple[int, ...]] = set(domainComputed)
	boxOfKnown: set[tuple[int, ...]] = set(domainKnown)

	boxOfMissing: list[tuple[int, ...]] = sorted(boxOfKnown - boxOfComputed)
	boxOfSurplus: list[tuple[int, ...]] = sorted(boxOfComputed - boxOfKnown)
	boxOfMatched: list[tuple[int, ...]] = sorted(boxOfComputed & boxOfKnown)

	comparisonResults: dict[str, list[tuple[int, ...]]] = {
		'missing': boxOfMissing,
		'surplus': boxOfSurplus,
		'matched': boxOfMatched,
	}

	if printResults:
		countComputed: int = len(boxOfComputed)
		countKnown: int = len(boxOfKnown)
		countMissing: int = len(boxOfMissing)
		countSurplus: int = len(boxOfSurplus)
		countMatched: int = len(boxOfMatched)

		sys.stdout.write(f"Domain comparison: {countComputed} computed vs {countKnown} known\n")
		sys.stdout.write(f"  Matched: {countMatched} ({100 * countMatched / countKnown:.1f}% of known)\n")

		if boxOfMissing:
			sys.stdout.write(f"  Missing ({countMissing} tuples in known but not in computed):\n")
			sys.stdout.write(pformat(boxOfMissing, width=140, compact=True) + '\n')

		if boxOfSurplus:
			sys.stdout.write(f"  Surplus ({countSurplus} tuples in computed but not in known):\n")
			sys.stdout.write(pformat(boxOfSurplus, width=140, compact=True) + '\n')

		if not boxOfMissing and not boxOfSurplus:
			sys.stdout.write("  Perfect match!\n")

	return comparisonResults
