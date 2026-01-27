# NOTE to AI assistants: this module is not representative of my coding style. Most of it is AI generated, but because it's temporary code, I didn't strictly enforce my usual standards. Do not emulate it.
from collections.abc import Sequence
from fractions import Fraction
from functools import cache, reduce
from gmpy2 import bit_flip
from hunterMakesPy import CallableFunction, raiseIfNone
from hunterMakesPy.dataStructures import updateExtendPolishDictionaryLists
from hunterMakesPy.filesystemToolkit import importPathFilename2Identifier, writePython
from itertools import product as CartesianProduct, repeat
from mapFolding import ansiColorReset, ansiColors, inclusive, packageSettings
from mapFolding._e import (
	getDictionaryLeafDomains, getLeafDomain, getPileRange, PermutationSpace, 首一, 首一三, 首一二, 首一二三, 首三, 首二, 首二三, 首零, 首零一,
	首零一三, 首零一二, 首零一二三, 首零三, 首零二, 首零二三)
from mapFolding._e._dataDynamic import getDataFrameFoldings
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import between, exclude
from mapFolding._e.pin2上nDimensions import pinPilesAtEnds
from mapFolding._e.pinIt import deconstructPermutationSpaceAtPile, deconstructPermutationSpaceByDomainOfLeaf
from mapFolding._e.Z0Z_analysisPython.toolkit import detectPermutationSpaceErrors, PermutationSpaceStatus
from more_itertools import consecutive_groups
from operator import indexOf, neg, pos
from pathlib import Path, PurePath
from pprint import pformat
from typing import TYPE_CHECKING
import numpy
import sys

if TYPE_CHECKING:
	import pandas

def 首一1(dd: int, /) -> int: return 首一(dd) + 1
def 首一三1(dd: int, /) -> int: return 首一三(dd) + 1
def 首一二1(dd: int, /) -> int: return 首一二(dd) + 1
def 首一二三1(dd: int, /) -> int: return 首一二三(dd) + 1
def 首三1(dd: int, /) -> int: return 首三(dd) + 1
def 首二1(dd: int, /) -> int: return 首二(dd) + 1
def 首二三1(dd: int, /) -> int: return 首二三(dd) + 1
def 首零1(dd: int, /) -> int: return 首零(dd) + 1
def 首零一1(dd: int, /) -> int: return 首零一(dd) + 1
def 首零一三1(dd: int, /) -> int: return 首零一三(dd) + 1
def 首零一二1(dd: int, /) -> int: return 首零一二(dd) + 1
def 首零一二三1(dd: int, /) -> int: return 首零一二三(dd) + 1
def 首零三1(dd: int, /) -> int: return 首零三(dd) + 1
def 首零二1(dd: int, /) -> int: return 首零二(dd) + 1
def 首零二三1(dd: int, /) -> int: return 首零二三(dd) + 1

type Addend = int
type FractionAddend = tuple[Fraction, Addend]
type IndexPilesTotal = int
type Leaf = int
type MapKind = str
type Pile = int
type strLeafExcluded = str
type strLeafExcluder = str
type strPileExcluded = str
type strPileExcluder = str
type ExclusionData = dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]]

pathExclusionData: Path = Path(f"{packageSettings.pathPackage}/_e/Z0Z_analysisPython/exclusionData")
pathExclusionData.mkdir(parents=True, exist_ok=True)

functionsHeadDimensions: list[CallableFunction[[int], int]] = [
	首一, 首一三, 首一二, 首一二三, 首三, 首二, 首二三, 首零, 首零一, 首零一三, 首零一二, 首零一二三, 首零三, 首零二, 首零二三,
	首一1, 首一三1, 首一二1, 首一二三1, 首三1, 首二1, 首二三1, 首零1, 首零一1, 首零一三1, 首零一二1, 首零一二三1, 首零三1, 首零二1, 首零二三1]
dictionaryFunctionsByName: dict[str, CallableFunction[[int], int]] = {function.__name__: function for function in functionsHeadDimensions}

#======== Collate exclusion data =======

def writeExclusionDataCollated(listDimensions: Sequence[int] = (5, 6)) -> list[PurePath]:
	"""{mapShape: {leafExcluder: {pileExcluder: {leafExcluded: listIndicesExcluded}}}}."""
	dictionaryIndices: dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]] = {}
	dictionaryIndicesNegative: dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]] = {}

# TODO Use the docstring to explain this computation, and change the computation to a simpler statement.
	listsAreAlwaysLessThanHalfLeavesTotal = 1
	integerDivisionIsSillyIfTheNumeratorIsLessThanTwiceTheDenominator = 1
	qq = min(listDimensions) - listsAreAlwaysLessThanHalfLeavesTotal - integerDivisionIsSillyIfTheNumeratorIsLessThanTwiceTheDenominator
	denominatorsValid: tuple[int, ...] = tuple(int(bit_flip(0, ww)) for ww in range(1, qq))

	for dimensionsTotal in listDimensions:
		state: EliminationState = EliminationState((2,) * dimensionsTotal)
		dataframeFoldings: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))

		dictionaryLeafDomains: dict[Leaf, range] = getDictionaryLeafDomains(state)

		mapKind: MapKind = f"p2d{dimensionsTotal}"
		dictionaryIndices[mapKind] = {}
		dictionaryIndicesNegative[mapKind] = {}

		for leafExcluder, pileExcluder, leafExcluded in CartesianProduct(functionsHeadDimensions, functionsHeadDimensions, functionsHeadDimensions):
			if pileExcluder(dimensionsTotal) not in dictionaryLeafDomains.get(leafExcluder(dimensionsTotal), []):
				continue
			pilesInTheDataframe: pandas.Series = dataframeFoldings.loc[dataframeFoldings[pileExcluder(dimensionsTotal)] == leafExcluder(dimensionsTotal)].eq(leafExcluded(dimensionsTotal)).any()
			leafExcludedValue = leafExcluded(dimensionsTotal)
			if leafExcludedValue not in dictionaryLeafDomains:
				continue
			listOfPiles: list[Pile] = list(dictionaryLeafDomains[leafExcludedValue])
			listPilesExcluded: set[Pile] = set(listOfPiles).difference(pilesInTheDataframe[pilesInTheDataframe].index.tolist())
			listIndicesExcluded: list[IndexPilesTotal] = sorted(map(indexOf, repeat(listOfPiles), listPilesExcluded))

			pilesTotal = len(listOfPiles)
			denominators: list[int] = list(filter(between(0, pilesTotal), denominatorsValid))
			dictionaryIndices[mapKind].setdefault(leafExcluder.__name__, {}).setdefault(pileExcluder.__name__, {})[leafExcluded.__name__] = [
				expressIndexAsFractionAddend(index, pilesTotal, tuple(denominators)) for index in listIndicesExcluded]
			dictionaryIndicesNegative[mapKind].setdefault(leafExcluder.__name__, {}).setdefault(pileExcluder.__name__, {})[leafExcluded.__name__] = [
				expressIndexAsFractionAddend(index - pilesTotal, pilesTotal, tuple(denominators)) for index in listIndicesExcluded]

	listPathFilenames: list[PurePath] = []

	for mapKind in dictionaryIndices:
		for leafExcluderName, dictionary in dictionaryIndices[mapKind].items():
			pythonSource: str = f"from fractions import Fraction\n\nleafExcluderData: dict[str, dict[str, list[tuple[Fraction, int]]]] = {pformat(dictionary, indent=0, width=160)}"
			pathFilename: Path = pathExclusionData / f"collated{mapKind}{leafExcluderName}.py"
			writePython(pythonSource, pathFilename)
			listPathFilenames.append(PurePath(pathFilename))

	for mapKind in dictionaryIndicesNegative:
		for leafExcluderName, dictionary in dictionaryIndicesNegative[mapKind].items():
			pythonSource: str = f"from fractions import Fraction\n\nleafExcluderData: dict[str, dict[str, list[tuple[Fraction, int]]]] = {pformat(dictionary, indent=0, width=160)}"
			pathFilename: Path = pathExclusionData / f"collated{mapKind}{leafExcluderName}Negative.py"
			writePython(pythonSource, pathFilename)
			listPathFilenames.append(PurePath(pathFilename))

	return listPathFilenames

@cache
def expressIndexAsFractionAddend(index: IndexPilesTotal, pilesTotal: int, denominators: tuple[int, ...]) -> FractionAddend:
	indexAsFractionAndAddend: FractionAddend = (Fraction(0, 1), index)
	direction = pos if index >= 0 else neg

	if denominators:
		addendsMagnitude: int = pilesTotal // max(denominators)
		distanceBest: float = 9000.1

		for denominator, addend in CartesianProduct(denominators, range(-(addendsMagnitude), addendsMagnitude + inclusive)):
			for numerator in range(1, denominator):
				if ((numerator / denominator).is_integer()) and (numerator // denominator in denominators):
					continue
				numerator = direction(numerator)
				if index == (((numerator * pilesTotal) // denominator) + addend):
					distance: float = abs(index - (((numerator * pilesTotal) / denominator) + addend))
					if distance < distanceBest:
						indexAsFractionAndAddend = (Fraction(numerator, denominator), addend)
						distanceBest = distance

	return indexAsFractionAndAddend

#======== Analyze exclusion data =======

def loadCollatedIndices(*, negative: bool = False) -> ExclusionData:
	collatedIndices: ExclusionData = {}
	stringGlob: str = 'collated*'
	stringGlob += 'Negative' if negative else ''
	for pathFilename in pathExclusionData.glob(stringGlob + ".py"):
		stem: str = pathFilename.stem.removeprefix("collated").removesuffix("Negative")
		mapKind, leafExcluderName = stem[0:4], stem[4:]
		collatedIndices.setdefault(mapKind, {})[leafExcluderName] = importPathFilename2Identifier(pathFilename, "leafExcluderData")
	return collatedIndices

@cache
def _dictionaryLeafDomainsByMapKind(mapKind: MapKind) -> dict[Leaf, range]:
	return getDictionaryLeafDomains(EliminationState((2,) * _dimensionsTotalFromMapKind(mapKind)))

def _dimensionsTotalFromMapKind(mapKind: MapKind) -> int:
	return int(mapKind.removeprefix("p2d"))

@cache
def _fractionAddendToIndex(fractionAddend: FractionAddend, pilesTotal: int) -> int:
	fraction, addend = fractionAddend
	index: int = ((fraction.numerator * pilesTotal) // fraction.denominator) + addend
	if index < 0:
		return pilesTotal + index
	return index

def _listFractionAddendsToIndices(mapKind: MapKind, leafExcludedName: strLeafExcluded, listFractionAddends: list[FractionAddend]) -> list[int]:
	pilesTotal: int = _pilesTotalOfLeafExcluded(mapKind, leafExcludedName)
	return [_fractionAddendToIndex(fractionAddend, pilesTotal) for fractionAddend in listFractionAddends]

@cache
def _pilesTotalOfLeafExcluded(mapKind: MapKind, leafExcludedName: strLeafExcluded) -> int:
	leaf: Leaf = dictionaryFunctionsByName[leafExcludedName](_dimensionsTotalFromMapKind(mapKind))
	return len(_dictionaryLeafDomainsByMapKind(mapKind)[leaf])

def _fractionAddendsForIndexSubset(listFractionAddends: list[FractionAddend], listResolvedIndices: list[int], indicesSubset: list[int]) -> set[FractionAddend]:
	if not indicesSubset:
		return set()
	indicesSet: set[int] = set(indicesSubset)
	return {fractionAddend for fractionAddend, resolvedIndex in zip(listFractionAddends, listResolvedIndices, strict=True) if resolvedIndex in indicesSet}

def _fractionAddendFromIndex(index: IndexPilesTotal) -> FractionAddend:
	return (Fraction(0, 1), index)

def _getContiguousFromStart(listIndices: list[IndexPilesTotal]) -> list[IndexPilesTotal]:
	"""Return the first contiguous group starting at index 0, if it has at least 2 elements."""
	listContiguous: list[IndexPilesTotal] = []
	if listIndices and listIndices[0] == 0:
		listContiguous = list(next(consecutive_groups(listIndices)))
		if len(listContiguous) < 2:
			listContiguous = []
	return listContiguous

def _getContiguousEndingAtNegativeOne(listRelativeIndices: list[int]) -> list[int]:
	"""Return the last contiguous group ending at -1, if it has at least 2 elements."""
	listContiguous: list[int] = []
	listRelativeIndicesSorted: list[int] = sorted(listRelativeIndices)
	if listRelativeIndicesSorted and listRelativeIndicesSorted[-1] == -1:
		for group in consecutive_groups(listRelativeIndicesSorted):
			listContiguous = list(group)
		if (len(listContiguous) < 2) or (listContiguous[-1] != -1):
			listContiguous = []
	return listContiguous

def analyzeContiguousStartAbsolute(dataset: ExclusionData) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	"""Find common contiguous indices starting from 0 across all map shapes, expressed as absolute indices."""
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	listMapKinds: list[MapKind] = list(dataset.keys())
	mapKind0: MapKind = listMapKinds[0]

	for leafExcluderName in dataset[mapKind0]:
		if any(leafExcluderName not in dataset[mapKind] for mapKind in listMapKinds):
			continue
		aggregatedExclusions[leafExcluderName] = {}

		for pileExcluderName in dataset[mapKind0][leafExcluderName]:
			if any(pileExcluderName not in dataset[mapKind][leafExcluderName] for mapKind in listMapKinds):
				continue
			aggregatedExclusions[leafExcluderName][pileExcluderName] = {}

			for leafExcludedName in dataset[mapKind0][leafExcluderName][pileExcluderName]:
				if any(leafExcludedName not in dataset[mapKind][leafExcluderName][pileExcluderName] for mapKind in listMapKinds):
					continue

				listContiguousLengths: list[int] = []

				for mapKind in listMapKinds:
					listFractionAddends: list[FractionAddend] = dataset[mapKind][leafExcluderName][pileExcluderName][leafExcludedName]
					resolvedIndices: list[int] = _listFractionAddendsToIndices(mapKind, leafExcludedName, listFractionAddends)
					listContiguousLengths.append(len(_getContiguousFromStart(resolvedIndices)))

				commonLength: int = min(listContiguousLengths) if listContiguousLengths else 0
				listFractionAddendsCommon: list[FractionAddend] = [_fractionAddendFromIndex(indexValue) for indexValue in range(commonLength)] if commonLength >= 2 else []
				aggregatedExclusions[leafExcluderName][pileExcluderName][leafExcludedName] = listFractionAddendsCommon

	return aggregatedExclusions

def analyzeContiguousEndAbsolute(dataset: ExclusionData) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	"""Find common contiguous indices ending at pilesTotal-1 across all map shapes, expressed as negative absolute indices."""
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	listMapKinds: list[MapKind] = list(dataset.keys())
	mapKind0: MapKind = listMapKinds[0]

	for leafExcluderName in dataset[mapKind0]:
		if any(leafExcluderName not in dataset[mapKind] for mapKind in listMapKinds):
			continue
		aggregatedExclusions[leafExcluderName] = {}

		for pileExcluderName in dataset[mapKind0][leafExcluderName]:
			if any(pileExcluderName not in dataset[mapKind][leafExcluderName] for mapKind in listMapKinds):
				continue
			aggregatedExclusions[leafExcluderName][pileExcluderName] = {}

			for leafExcludedName in dataset[mapKind0][leafExcluderName][pileExcluderName]:
				if any(leafExcludedName not in dataset[mapKind][leafExcluderName][pileExcluderName] for mapKind in listMapKinds):
					continue

				listContiguousLengths: list[int] = []

				for mapKind in listMapKinds:
					listFractionAddends: list[FractionAddend] = dataset[mapKind][leafExcluderName][pileExcluderName][leafExcludedName]
					listRelativeIndices: list[int] = _listFractionAddendsToIndices(mapKind, leafExcludedName, listFractionAddends)
					listContiguousLengths.append(len(_getContiguousEndingAtNegativeOne(listRelativeIndices)))

				commonLength: int = min(listContiguousLengths) if listContiguousLengths else 0
				rangeIndices: range = range(-commonLength, 0) if commonLength >= 2 else range(0)
				aggregatedExclusions[leafExcluderName][pileExcluderName][leafExcludedName] = [_fractionAddendFromIndex(indexValue) for indexValue in rangeIndices]

	return aggregatedExclusions

def analyzeContiguousStartRelative(dataset: ExclusionData) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	"""Find common contiguous indices starting from 0 across all map shapes, expressed as fractions of pilesTotal."""
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	listMapKinds: list[MapKind] = list(dataset.keys())
	mapKind0: MapKind = listMapKinds[0]

	for leafExcluderName in dataset[mapKind0]:
		if any(leafExcluderName not in dataset[mapKind] for mapKind in listMapKinds):
			continue
		aggregatedExclusions[leafExcluderName] = {}

		for pileExcluderName in dataset[mapKind0][leafExcluderName]:
			if any(pileExcluderName not in dataset[mapKind][leafExcluderName] for mapKind in listMapKinds):
				continue
			aggregatedExclusions[leafExcluderName][pileExcluderName] = {}

			for leafExcludedName in dataset[mapKind0][leafExcluderName][pileExcluderName]:
				if any(leafExcludedName not in dataset[mapKind][leafExcluderName][pileExcluderName] for mapKind in listMapKinds):
					continue

				listSetsFractionAddends: list[set[FractionAddend]] = []

				for mapKind in listMapKinds:
					listFractionAddends: list[FractionAddend] = dataset[mapKind][leafExcluderName][pileExcluderName][leafExcludedName]
					resolvedIndices: list[int] = _listFractionAddendsToIndices(mapKind, leafExcludedName, listFractionAddends)
					contiguousIndices: list[int] = _getContiguousFromStart(resolvedIndices)
					setFractionAddends: set[FractionAddend] = _fractionAddendsForIndexSubset(listFractionAddends, resolvedIndices, contiguousIndices)
					listSetsFractionAddends.append(setFractionAddends)

				commonFractionAddends: set[FractionAddend] = reduce(set[FractionAddend].intersection, listSetsFractionAddends)
				aggregatedExclusions[leafExcluderName][pileExcluderName][leafExcludedName] = list(commonFractionAddends)

	return aggregatedExclusions

def analyzeContiguousEndRelative(dataset: ExclusionData) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	"""Find common contiguous indices ending at pilesTotal-1 across all map shapes, expressed as fractions of pilesTotal."""
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	listMapKinds: list[MapKind] = list(dataset.keys())
	mapKind0: MapKind = listMapKinds[0]

	for leafExcluderName in dataset[mapKind0]:
		if any(leafExcluderName not in dataset[mapKind] for mapKind in listMapKinds):
			continue
		aggregatedExclusions[leafExcluderName] = {}

		for pileExcluderName in dataset[mapKind0][leafExcluderName]:
			if any(pileExcluderName not in dataset[mapKind][leafExcluderName] for mapKind in listMapKinds):
				continue
			aggregatedExclusions[leafExcluderName][pileExcluderName] = {}

			for leafExcludedName in dataset[mapKind0][leafExcluderName][pileExcluderName]:
				if any(leafExcludedName not in dataset[mapKind][leafExcluderName][pileExcluderName] for mapKind in listMapKinds):
					continue

				listSetsFractionAddends: list[set[FractionAddend]] = []

				for mapKind in listMapKinds:
					listFractionAddends: list[FractionAddend] = dataset[mapKind][leafExcluderName][pileExcluderName][leafExcludedName]
					listRelativeIndices: list[int] = _listFractionAddendsToIndices(mapKind, leafExcludedName, listFractionAddends)
					contiguousRelativeIndices: list[int] = _getContiguousEndingAtNegativeOne(listRelativeIndices)
					setFractionAddends: set[FractionAddend] = _fractionAddendsForIndexSubset(listFractionAddends, listRelativeIndices, contiguousRelativeIndices)
					listSetsFractionAddends.append(setFractionAddends)

				commonFractionAddends: set[FractionAddend] = reduce(set[FractionAddend].intersection, listSetsFractionAddends)
				aggregatedExclusions[leafExcluderName][pileExcluderName][leafExcludedName] = list(commonFractionAddends)

	return aggregatedExclusions

def analyzeNonContiguousIndicesRelative(dataset: ExclusionData) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	"""Find common indices across all map shapes (contiguous or not), expressed as fractions of pilesTotal."""
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	listMapKinds: list[MapKind] = list(dataset.keys())
	mapKind0: MapKind = listMapKinds[0]

	for leafExcluderName in dataset[mapKind0]:
		if any(leafExcluderName not in dataset[mapKind] for mapKind in listMapKinds):
			continue

		for pileExcluderName in dataset[mapKind0][leafExcluderName]:
			if any(pileExcluderName not in dataset[mapKind][leafExcluderName] for mapKind in listMapKinds):
				continue

			for leafExcludedName in dataset[mapKind0][leafExcluderName][pileExcluderName]:
				if any(leafExcludedName not in dataset[mapKind][leafExcluderName][pileExcluderName] for mapKind in listMapKinds):
					continue

				listSetsFractionAddends: list[set[FractionAddend]] = []

				for mapKind in listMapKinds:
					listFractionAddends: list[FractionAddend] = dataset[mapKind][leafExcluderName][pileExcluderName][leafExcludedName]
					setFractionAddends: set[FractionAddend] = set(listFractionAddends)
					listSetsFractionAddends.append(setFractionAddends)

				aggregatedExclusions.setdefault(leafExcluderName, {}).setdefault(pileExcluderName, {})[leafExcludedName] = list(reduce(set[FractionAddend].intersection, listSetsFractionAddends))

	return aggregatedExclusions

#======== Aggregate exclusion data =======

def writeAggregatedExclusions(pathWrite: Path | None = None) -> list[PurePath]:
	"""{leafExcluder: {pileExcluder: {leafExcluded: listIndicesAsFractionAddends}}}."""
	if pathWrite is None:
		pathWrite = pathExclusionData
	listPathFilenames: list[PurePath] = []
	collatedIndices: ExclusionData = loadCollatedIndices()
	collatedIndicesNegative: ExclusionData = loadCollatedIndices(negative=True)

	listExclusions: list[dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]] = [
		analyzeContiguousEndAbsolute(collatedIndicesNegative),
		analyzeContiguousEndRelative(collatedIndicesNegative),
		analyzeContiguousStartAbsolute(collatedIndices),
		analyzeContiguousStartRelative(collatedIndices),
		analyzeNonContiguousIndicesRelative(collatedIndices),
		analyzeNonContiguousIndicesRelative(collatedIndicesNegative),
	]

	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}

	for leafExcluder, pileExcluder in CartesianProduct(functionsHeadDimensions, functionsHeadDimensions):
		dictionaryMerged: dict[strLeafExcluded, list[FractionAddend]] = updateExtendPolishDictionaryLists(
			*[dictionaryExclusions.get(leafExcluder.__name__, {}).get(pileExcluder.__name__, {}) for dictionaryExclusions in listExclusions]
			, destroyDuplicates=True, reorderLists=True)
		aggregatedExclusions.setdefault(leafExcluder.__name__, {})[pileExcluder.__name__] = dictionaryMerged

	for leafExcluderName, pileExcluderData in aggregatedExclusions.items():
		pythonSource: str = "from fractions import Fraction\n\n"
		pythonSource += "type FractionAddend = tuple[Fraction, int]\n\n"
		dataFormatted: str = pformat(pileExcluderData, indent=0, width=160, compact=True)
		pythonSource += f"dictionaryExclusions: dict[str, dict[str, list[FractionAddend]]] = {dataFormatted}\n"
		pathFilename: Path = pathWrite / f"aggregated{leafExcluderName}.py"
		writePython(pythonSource, pathFilename)
		listPathFilenames.append(PurePath(pathFilename))

	return listPathFilenames

#======== Create exclusion dictionaries for elimination tools =======

def loadAggregatedExclusions() -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	for pathFilename in pathExclusionData.glob("aggregated*.py"):
		leafExcluderName: str = pathFilename.stem.removeprefix("aggregated")
		aggregatedExclusions[leafExcluderName] = importPathFilename2Identifier(pathFilename, "dictionaryExclusions")
	return aggregatedExclusions

def restructureAggregatedExclusionsForMapShape(dimensionsTotal: int, aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]) -> tuple[dict[Leaf, dict[Pile, dict[Pile, list[Leaf]]]], dict[Pile, dict[Leaf, dict[Pile, list[Leaf]]]]]:
	leafDomains: dict[Leaf, range] = getDictionaryLeafDomains(EliminationState(mapShape=(2,) * dimensionsTotal))

	dictionaryLeafExcludedAtPileByPile: dict[Leaf, dict[Pile, dict[Pile, list[Leaf]]]] = {}
	dictionaryAtPileLeafExcludedByPile: dict[Pile, dict[Leaf, dict[Pile, list[Leaf]]]] = {}

	for leafExcluderName, pileExcluderData in aggregatedExclusions.items():
		for pileExcluderName, leafExcludedData in pileExcluderData.items():
			for leafExcludedName, listFractionAddends in leafExcludedData.items():
				leafExcluded: Leaf = dictionaryFunctionsByName[leafExcludedName](dimensionsTotal)
				leafExcluderValue: Leaf = dictionaryFunctionsByName[leafExcluderName](dimensionsTotal)
				pileExcluderValue: Pile = dictionaryFunctionsByName[pileExcluderName](dimensionsTotal)

				domainOfLeafExcluded: list[Pile] = list(leafDomains[leafExcluded])
				pilesTotal: int = len(domainOfLeafExcluded)

				for fractionAddend in listFractionAddends:
					indexResolved: int = _fractionAddendToIndex(fractionAddend, pilesTotal)
					pileExcluded: Pile = domainOfLeafExcluded[indexResolved]

					dictionaryLeafExcludedAtPileByPile.setdefault(leafExcluded, {}
						).setdefault(pileExcluded, {}
							).setdefault(pileExcluderValue, []
								).append(leafExcluderValue)
					dictionaryAtPileLeafExcludedByPile.setdefault(pileExcluded, {}
						).setdefault(leafExcluded, {}
							).setdefault(pileExcluderValue, []
								).append(leafExcluderValue)

	for leafExcluded in dictionaryLeafExcludedAtPileByPile:
		for pileExcluded in dictionaryLeafExcludedAtPileByPile[leafExcluded]:
			for pileExcluder in dictionaryLeafExcludedAtPileByPile[leafExcluded][pileExcluded]:
				dictionaryLeafExcludedAtPileByPile[leafExcluded][pileExcluded][pileExcluder] = sorted(set(dictionaryLeafExcludedAtPileByPile[leafExcluded][pileExcluded][pileExcluder]))

	for pileExcluded in dictionaryAtPileLeafExcludedByPile:
		for leafExcluded in dictionaryAtPileLeafExcludedByPile[pileExcluded]:
			for pileExcluder in dictionaryAtPileLeafExcludedByPile[pileExcluded][leafExcluded]:
				dictionaryAtPileLeafExcludedByPile[pileExcluded][leafExcluded][pileExcluder] = sorted(set(dictionaryAtPileLeafExcludedByPile[pileExcluded][leafExcluded][pileExcluder]))

	return (dictionaryLeafExcludedAtPileByPile, dictionaryAtPileLeafExcludedByPile)

def writeExclusionDictionaries(pathExclusionsFile: PurePath | None = None) -> PurePath:
	listDimensions: list[int] = [5, 6]
	listLines: list[str] = ['"""Static exclusion dictionaries for elimination tools."""', ""]

	for dimensionsTotal in listDimensions:
		dictionaryLeafExcludedAtPileByPile, dictionaryAtPileLeafExcludedByPile = restructureAggregatedExclusionsForMapShape(dimensionsTotal, loadAggregatedExclusions())

		mapKind: MapKind = f"2d{dimensionsTotal}"

		listLines.append(f"dictionary{mapKind}LeafExcludedAtPileByPile: dict[int, dict[int, dict[int, list[int]]]] = {pformat(dictionaryLeafExcludedAtPileByPile, indent=4, width=160, compact=True)}")
		listLines.append("")
		listLines.append(f"dictionary{mapKind}AtPileLeafExcludedByPile: dict[int, dict[int, dict[int, list[int]]]] = {pformat(dictionaryAtPileLeafExcludedByPile, indent=4, width=160, compact=True)}")
		listLines.append("")

	pathFilename: Path = Path(pathExclusionsFile) if pathExclusionsFile is not None else Path(f"{packageSettings.pathPackage}/_e/_exclusions.py")
	pythonSource: str = "\n".join(listLines)
	writePython(pythonSource, pathFilename)

	return PurePath(pathFilename)

#======== Validation functions =======

@cache
def _getArrayFoldingsByDimensions(dimensionsTotal: int) -> numpy.ndarray:
	return raiseIfNone(getDataFrameFoldings(EliminationState((2,) * dimensionsTotal))).to_numpy(dtype=numpy.uint8, copy=False)

def validateExclusionDictionaries(dictionaryLeafExcludedAtPileByPile: dict[Leaf, dict[Pile, dict[Pile, list[Leaf]]]], dictionaryAtPileLeafExcludedByPile: dict[Pile, dict[Leaf, dict[Pile, list[Leaf]]]], mapKind: MapKind) -> tuple[bool, list[str]]:
	dimensions: int = int(mapKind[3:])
	listValidationErrors: list[str] = []
	arrayFoldings = _getArrayFoldingsByDimensions(dimensions)

	for pileExcluded, leafExcludedData in dictionaryAtPileLeafExcludedByPile.items():
		for leafExcluded, pileExcluderData in leafExcludedData.items():
			for pileExcluder, listLeafExcluders in pileExcluderData.items():
				for leafExcluder in listLeafExcluders:
					mask = (arrayFoldings[:, pileExcluder] == leafExcluder)
					if (arrayFoldings[mask, pileExcluded] == leafExcluded).any():
						listValidationErrors.append(
							f"Invalid exclusion in dictionaryAtPile...: If pile {pileExcluder} has leaf {leafExcluder}, "
							f"pile {pileExcluded} cannot have leaf {leafExcluded}. Found counter-example."
						)

	for leafExcluded, pileExcludedData in dictionaryLeafExcludedAtPileByPile.items():
		for pileExcluded, pileExcluderData in pileExcludedData.items():
			for pileExcluder, listLeafExcluders in pileExcluderData.items():
				for leafExcluder in listLeafExcluders:
					mask = (arrayFoldings[:, pileExcluder] == leafExcluder)
					if (arrayFoldings[mask, pileExcluded] == leafExcluded).any():
						listValidationErrors.append(
							f"Invalid exclusion in dictionaryLeaf...: If pile {pileExcluder} has leaf {leafExcluder}, "
							f"pile {pileExcluded} cannot have leaf {leafExcluded}. Found counter-example."
						)

	return len(listValidationErrors) == 0, listValidationErrors

def validateAnalysisMethodForMapShape(exclusionsFromAnalysisMethod: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]], mapKind: MapKind) -> tuple[bool, list[str]]:
	dimensions: int = int(mapKind[3:])
	listValidationErrors: list[str] = []
	arrayFoldings = _getArrayFoldingsByDimensions(dimensions)
	rowsTotal: int = int(arrayFoldings.shape[0])

	for leafExcluderName in exclusionsFromAnalysisMethod:
		leafExcluderFunction: CallableFunction[[int], int] = dictionaryFunctionsByName[leafExcluderName]
		leafExcluder: int = leafExcluderFunction(dimensions)

		for pileExcluderName in exclusionsFromAnalysisMethod[leafExcluderName]:
			pileExcluderFunction: CallableFunction[[int], int] = dictionaryFunctionsByName[pileExcluderName]
			pileExcluder: int = pileExcluderFunction(dimensions)

			for leafExcludedName in exclusionsFromAnalysisMethod[leafExcluderName][pileExcluderName]:
				leafExcludedFunction: CallableFunction[[int], int] = dictionaryFunctionsByName[leafExcludedName]
				leafExcluded: int = leafExcludedFunction(dimensions)

				listFractionAddends: list[FractionAddend] = exclusionsFromAnalysisMethod[leafExcluderName][pileExcluderName][leafExcludedName]

				if not listFractionAddends:
					continue

				stateValidation: EliminationState = EliminationState(mapShape=(2,) * dimensions)
				pilesTotalCurrent: int = len(list(getLeafDomain(stateValidation, dictionaryFunctionsByName[leafExcludedName](dimensions))))

				listIndicesExcluded: list[int] = []
				for fractionAddend in listFractionAddends:
					indexComputed: int = _fractionAddendToIndex(fractionAddend, pilesTotalCurrent)
					listIndicesExcluded.append(indexComputed)

				if not listIndicesExcluded:
					continue

				stateValidation = pinPilesAtEnds(stateValidation, 1)

				pileRange: list[int] = list(getPileRange(stateValidation, pileExcluder))
				dictionaryDeconstructed: dict[int, PermutationSpace] = deconstructPermutationSpaceAtPile(stateValidation.listPermutationSpace[0], pileExcluder, pileRange)

				permutationSpaceWithExcluder: PermutationSpace | None = dictionaryDeconstructed.get(leafExcluder)
				if permutationSpaceWithExcluder is None:
					continue

				listPermutationSpaceOther: list[PermutationSpace] = [permutationSpace for leaf, permutationSpace in dictionaryDeconstructed.items() if leaf != leafExcluder]

				domainOfLeafExcluded: list[int] = list(getLeafDomain(stateValidation, leafExcluded))
				domainReduced: list[int] = list(exclude(domainOfLeafExcluded, listIndicesExcluded))

				listPermutationSpaceFromExcluder: list[PermutationSpace] = deconstructPermutationSpaceByDomainOfLeaf(permutationSpaceWithExcluder, leafExcluded, domainReduced)

				stateValidation.listPermutationSpace = listPermutationSpaceOther + listPermutationSpaceFromExcluder

				pinningCoverage: PermutationSpaceStatus = detectPermutationSpaceErrors(arrayFoldings, stateValidation.listPermutationSpace)

				if pinningCoverage.rowsRequired < rowsTotal:
					listValidationErrors.append(
						f"{mapKind} {leafExcluderName}->{pileExcluderName}->{leafExcludedName}: {pinningCoverage.rowsRequired = }/{rowsTotal}")

				overlappingRowCount: int = int(pinningCoverage.indicesOverlappingRows.size)
				if overlappingRowCount > 0:
					listValidationErrors.append(
						f"{mapKind} {leafExcluderName}->{pileExcluderName}->{leafExcludedName}: {overlappingRowCount} overlapping rows")

	isValid: bool = len(listValidationErrors) == 0
	return (isValid, listValidationErrors)

def validateAnalysisMethod(analysisMethodCallable: CallableFunction[[ExclusionData], dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]]) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	collatedIndices: ExclusionData = loadCollatedIndices()
	listMapShapeNames: list[str] = list(collatedIndices.keys())
	exclusionsFromMethod: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = analysisMethodCallable(collatedIndices)
	errorsByMapShape: dict[str, list[str]] = {}

	for mapShapeName in listMapShapeNames:
		isValid, listValidationErrors = validateAnalysisMethodForMapShape(exclusionsFromMethod, mapShapeName)
		if not isValid:
			errorsByMapShape.setdefault(mapShapeName, []).extend(listValidationErrors)

		dimensions: int = int(mapShapeName[3:])
		dictionaryLeafExcludedAtPileByPile, dictionaryAtPileLeafExcludedByPile = restructureAggregatedExclusionsForMapShape(dimensions, exclusionsFromMethod)
		isValidDictionaries, listDictionaryErrors = validateExclusionDictionaries(dictionaryLeafExcludedAtPileByPile, dictionaryAtPileLeafExcludedByPile, mapShapeName)
		if not isValidDictionaries:
			errorsByMapShape.setdefault(mapShapeName, []).extend(listDictionaryErrors)

	if not errorsByMapShape:
		colorSuccess = ansiColors.BlackOnCyan
		sys.stdout.write(f"{colorSuccess}{analysisMethodCallable.__name__} validated across {len(listMapShapeNames)} mapShapes{ansiColorReset}\n")
	else:
		colorFailure = ansiColors.WhiteOnMagenta
		sys.stdout.write(f"{colorFailure}{analysisMethodCallable.__name__} validation failed for {len(errorsByMapShape)} mapShapes{ansiColorReset}\n")
		for mapShapeName, listErrors in errorsByMapShape.items():
			sys.stdout.write(f"{colorFailure}{mapShapeName}: {len(listErrors)} issues{ansiColorReset}\n")
			for error in listErrors[0:3]:
				sys.stdout.write(f"{colorFailure}  {error}{ansiColorReset}\n")

	return exclusionsFromMethod

def runGenerators() -> None:
	sys.stdout.write(f"{writeExclusionDataCollated() = }\n")
	sys.stdout.write(f"{writeAggregatedExclusions() = }\n")
	sys.stdout.write(f"{writeExclusionDictionaries() = }\n")

def runValidators() -> None:
	collatedIndices: ExclusionData = loadCollatedIndices()
	if not collatedIndices:
		sys.stdout.write("No collated indices found. Run 'generate' mode first to create exclusion data.\\n")
		return

	listAnalysisMethods = [
		analyzeNonContiguousIndicesRelative,
		analyzeContiguousStartAbsolute,
		analyzeContiguousEndAbsolute,
		analyzeContiguousStartRelative,
		analyzeContiguousEndRelative,
	]

	for analysisMethod in listAnalysisMethods:
		validateAnalysisMethod(analysisMethod)

if __name__ == '__main__':
	runGenerators()
	runValidators()

