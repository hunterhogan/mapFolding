from collections.abc import Callable, Iterable, Sequence
from fractions import Fraction
from functools import cache, reduce
from gmpy2 import bit_flip
from hunterMakesPy import importPathFilename2Identifier, updateExtendPolishDictionaryLists, writePython
from itertools import product as CartesianProduct, repeat
from mapFolding import (
	between, consecutive, decreasing, DOTvalues, exclude, inclusive, mappingHasKey, noDuplicates, packageSettings,
	reverseLookup)
from mapFolding._e import (
	getDictionaryLeafDomains, 首一, 首一三, 首一二, 首一二三, 首三, 首二, 首二三, 首零, 首零一, 首零一三, 首零一二, 首零一二三, 首零三, 首零二, 首零二三)
from mapFolding._e._data import getDataFrameFoldings
from mapFolding._e.analysisPython.exclusionData.collated import dictionaryExclusionData as exclusionData
from mapFolding._e.pinIt import notLeafOriginOrLeaf零
from mapFolding.dataBaskets import EliminationState
from more_itertools import consecutive_groups
from operator import indexOf, neg, pos
from pathlib import Path, PurePath
from pprint import pformat
from typing import TYPE_CHECKING
import cytoolz.dicttoolz
import cytoolz.functoolz
import cytoolz.itertoolz
import functools
import hunterMakesPy as humpy
import itertools
import more_itertools
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
type OperatorSign = Callable[[int], int]
type FractionAddend = tuple[OperatorSign, Fraction, Addend]
type IndexPilesTotal = int
type Leaf = int
type MapKind = str
type Pile = int
type strLeafExcluded = str
type strLeafExcluder = str
type strPileExcluded = str
type strPileExcluder = str

functionsHeadDimensions: list[Callable[[int], int]] = [
	首一, 首一三, 首一二, 首一二三, 首三, 首二, 首二三, 首零, 首零一, 首零一三, 首零一二, 首零一二三, 首零三, 首零二, 首零二三,
	首一1, 首一三1, 首一二1, 首一二三1, 首三1, 首二1, 首二三1, 首零1, 首零一1, 首零一三1, 首零一二1, 首零一二三1, 首零三1, 首零二1, 首零二三1]
dictionaryFunctionsByName: dict[str, Callable[[int], int]] = {function.__name__: function for function in functionsHeadDimensions}

pathExclusionData: Path = Path(f"{packageSettings.pathPackage}/_e/analysisPython/exclusionData")
pathExclusionData.mkdir(parents=True, exist_ok=True)

# ======= Collate exclusion data =======

def writeExclusionDataCollated(listDimensions: Sequence[int] = (5, 6)) -> PurePath:
	"""{mapShape: {leafExcluder: {pileExcluder: {leafExcluded: listIndicesExcluded}}}}."""
	dictionaryExclusionData: dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]] = {}

# TODO Use the docstring to explain this computation, and change the computation to a simpler statement.
	listsAreAlwaysLessThanHalfLeavesTotal = 1
	integerDivisionIsSillyIfTheNumeratorIsLessThanTwiceTheDenominator = 1
	qq = min(listDimensions) - listsAreAlwaysLessThanHalfLeavesTotal - integerDivisionIsSillyIfTheNumeratorIsLessThanTwiceTheDenominator
	ww = int(bit_flip(0, qq))
	denominatorsValid: tuple[int, ...] = tuple(range(2, ww + inclusive, 2))

	for dimensionsTotal in listDimensions:
		state: EliminationState = EliminationState((2,) * dimensionsTotal)
		dataframeFoldings: pandas.DataFrame = getDataFrameFoldings(state)

		dictionaryLeafDomains: dict[Leaf, range] = getDictionaryLeafDomains(state)

		mapKind: MapKind = f"p2d{dimensionsTotal}"
		dictionaryExclusionData[mapKind] = {}

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
			listFractionAddends: list[FractionAddend] = [expressIndexAsFractionAddend(index, pilesTotal, tuple(denominators)) for index in listIndicesExcluded]

			dictionaryExclusionData[mapKind].setdefault(leafExcluder.__name__, {}).setdefault(pileExcluder.__name__, {})[leafExcluded.__name__] = listFractionAddends

	pathFilename: Path = pathExclusionData / "collated.py"
	pythonSource: str = f"dictionaryExclusionData: dict[str, dict[str, dict[str, dict[str, list[int]]]]] = {pformat(dictionaryExclusionData, indent=2, width=160)}"
	writePython(pythonSource, pathFilename)

	return PurePath(pathFilename)

@cache
def expressIndexAsFractionAddend(index: IndexPilesTotal, pilesTotal: int, denominators: tuple[int, ...]) -> FractionAddend:
	indexSign: OperatorSign = pos if index >= 0 else neg
	indexMagnitude: IndexPilesTotal = indexSign(index)
	indexAsFractionAndAddend: FractionAddend = (indexSign, Fraction(0, 1), indexMagnitude)

	if denominators:
		addendsMagnitude: int = pilesTotal // max(denominators)
		distanceBest: float = 9000.1

# TODO This code block is a half-implemented idea. I got stuck: I don't think this is "right" way to do it.
# The block needs to iterate computations, while skipping some combinations.
# Skip equivalent, unreduced fractions: compute 1/2, but not 2/4.
# If `indexSign` is positive, a negative addend must not accidentally convert the computed index to a negative index--and vice versa.
		for sliceOfDenominators, Z0Z_addendFloor in [(denominators[0:1], 0), (denominators[1:None], -(addendsMagnitude))]:
			for denominator, addend in CartesianProduct(sliceOfDenominators, range(Z0Z_addendFloor, addendsMagnitude + inclusive)):
				for numerator in range(denominator):
					if ((numerator / denominator).is_integer()) and (numerator // denominator in denominators):
						continue
					candidateMagnitude: int = (((numerator * pilesTotal) // denominator) + addend)
					if indexMagnitude == candidateMagnitude:
						distance: float = abs(indexMagnitude - (((numerator * pilesTotal) / denominator) + addend))
						if distance < distanceBest:
							indexAsFractionAndAddend = (indexSign, Fraction(numerator, denominator), addend)
							distanceBest = distance

	return indexAsFractionAndAddend

# ======= Analyze exclusion data =======

def _fractionAddendFromIndex(indexValue: IndexPilesTotal) -> FractionAddend:
	signOperator: OperatorSign = pos if indexValue >= 0 else neg
	return (signOperator, Fraction(0, 1), signOperator(indexValue))

def _sortedFractionAddends(iterable: Iterable[FractionAddend]) -> list[FractionAddend]:
	def _fractionAddendSortKey(fractionAddend: FractionAddend) -> tuple[str, Fraction, Addend]:
		signOperator, indexFraction, addend = fractionAddend
		return (signOperator.__name__, indexFraction, addend)
	return sorted(iterable, key=_fractionAddendSortKey)

def _getContiguousFromStart(listIndices: list[IndexPilesTotal]) -> list[IndexPilesTotal]:
	"""Return the first contiguous group starting at index 0, if it has at least 2 elements."""
	listContiguous: list[IndexPilesTotal] = []
	if listIndices and listIndices[0] == 0:
		listContiguous = list(next(consecutive_groups(listIndices)))
		if len(listContiguous) < 2:
			listContiguous = []
	return listContiguous

def _getContiguousFromEnd(listIndices: list[IndexPilesTotal], pilesTotal: int) -> list[IndexPilesTotal]:
	"""Return the last contiguous group ending at pilesTotal-1, if it has at least 2 elements."""
	listContiguous: list[IndexPilesTotal] = []
	if listIndices and listIndices[-1] == pilesTotal - 1:
		for group in consecutive_groups(listIndices):
			listContiguous = list(group)
		if len(listContiguous) < 2:
			listContiguous = []
	return listContiguous

def analyzeContiguousStartAbsolute(exclusionDataSource: dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[IndexPilesTotal]]]]]) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	"""Find common contiguous indices starting from 0 across all map shapes, expressed as absolute indices."""
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	listMapKinds: list[MapKind] = list(exclusionDataSource.keys())
	mapShapeFirst: MapKind = listMapKinds[0]

	for leafExcluderName in exclusionDataSource[mapShapeFirst]:
		if any(leafExcluderName not in exclusionDataSource[mapShapeName] for mapShapeName in listMapKinds):
			continue
		aggregatedExclusions[leafExcluderName] = {}

		for pileExcluderName in exclusionDataSource[mapShapeFirst][leafExcluderName]:
			if any(pileExcluderName not in exclusionDataSource[mapShapeName][leafExcluderName] for mapShapeName in listMapKinds):
				continue
			aggregatedExclusions[leafExcluderName][pileExcluderName] = {}

			for leafExcludedName in exclusionDataSource[mapShapeFirst][leafExcluderName][pileExcluderName]:
				if any(leafExcludedName not in exclusionDataSource[mapShapeName][leafExcluderName][pileExcluderName] for mapShapeName in listMapKinds):
					continue

				listContiguousLengths: list[int] = []
				maximumPilesTotal: int = 0

				for mapShapeName in listMapKinds:
					listIndicesWithPilesTotal: list[IndexPilesTotal] = exclusionDataSource[mapShapeName][leafExcluderName][pileExcluderName][leafExcludedName]
					pilesTotal: int = listIndicesWithPilesTotal[-1]
					indicesOnly: list[IndexPilesTotal] = listIndicesWithPilesTotal[:-1]
					listContiguousLengths.append(len(_getContiguousFromStart(indicesOnly)))
					maximumPilesTotal = max(maximumPilesTotal, pilesTotal)

				commonLength: int = min(listContiguousLengths)
				listFractionAddends: list[FractionAddend] = [_fractionAddendFromIndex(index) for index in range(commonLength)] if commonLength >= 2 else []
				listFractionAddends.append((pos, Fraction(*maximumPilesTotal.as_integer_ratio()), 0))
				aggregatedExclusions[leafExcluderName][pileExcluderName][leafExcludedName] = listFractionAddends

	return aggregatedExclusions

def analyzeContiguousEndAbsolute(exclusionDataSource: dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[IndexPilesTotal]]]]]) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	"""Find common contiguous indices ending at pilesTotal-1 across all map shapes, expressed as negative absolute indices."""
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	listMapKinds: list[MapKind] = list(exclusionDataSource.keys())
	mapShapeFirst: MapKind = listMapKinds[0]

	for leafExcluderName in exclusionDataSource[mapShapeFirst]:
		if any(leafExcluderName not in exclusionDataSource[mapShapeName] for mapShapeName in listMapKinds):
			continue
		aggregatedExclusions[leafExcluderName] = {}

		for pileExcluderName in exclusionDataSource[mapShapeFirst][leafExcluderName]:
			if any(pileExcluderName not in exclusionDataSource[mapShapeName][leafExcluderName] for mapShapeName in listMapKinds):
				continue
			aggregatedExclusions[leafExcluderName][pileExcluderName] = {}

			for leafExcludedName in exclusionDataSource[mapShapeFirst][leafExcluderName][pileExcluderName]:
				if any(leafExcludedName not in exclusionDataSource[mapShapeName][leafExcluderName][pileExcluderName] for mapShapeName in listMapKinds):
					continue

				listContiguousLengths: list[int] = []
				maximumPilesTotal: int = 0

				for mapShapeName in listMapKinds:
					listIndicesWithPilesTotal: list[IndexPilesTotal] = exclusionDataSource[mapShapeName][leafExcluderName][pileExcluderName][leafExcludedName]
					pilesTotal: int = listIndicesWithPilesTotal[-1]
					indicesOnly: list[IndexPilesTotal] = listIndicesWithPilesTotal[:-1]
					listContiguousLengths.append(len(_getContiguousFromEnd(indicesOnly, pilesTotal)))
					maximumPilesTotal = max(maximumPilesTotal, pilesTotal)

				commonLength: int = min(listContiguousLengths)
				listFractionAddends: list[FractionAddend] = [_fractionAddendFromIndex(index) for index in range(0 + decreasing, (commonLength * decreasing) + inclusive, decreasing)] if commonLength >= 2 else []
				listFractionAddends.append((pos, Fraction(*maximumPilesTotal.as_integer_ratio()), 0))
				aggregatedExclusions[leafExcluderName][pileExcluderName][leafExcludedName] = listFractionAddends

	return aggregatedExclusions

def analyzeContiguousStartRelative(exclusionDataSource: dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[IndexPilesTotal]]]]]) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	"""Find common contiguous indices starting from 0 across all map shapes, expressed as fractions of pilesTotal."""
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	listMapKinds: list[MapKind] = list(exclusionDataSource.keys())
	mapShapeFirst: MapKind = listMapKinds[0]

	for leafExcluderName in exclusionDataSource[mapShapeFirst]:
		if any(leafExcluderName not in exclusionDataSource[mapShapeName] for mapShapeName in listMapKinds):
			continue
		aggregatedExclusions[leafExcluderName] = {}

		for pileExcluderName in exclusionDataSource[mapShapeFirst][leafExcluderName]:
			if any(pileExcluderName not in exclusionDataSource[mapShapeName][leafExcluderName] for mapShapeName in listMapKinds):
				continue
			aggregatedExclusions[leafExcluderName][pileExcluderName] = {}

			for leafExcludedName in exclusionDataSource[mapShapeFirst][leafExcluderName][pileExcluderName]:
				if any(leafExcludedName not in exclusionDataSource[mapShapeName][leafExcluderName][pileExcluderName] for mapShapeName in listMapKinds):
					continue

				listSetsFractionAddends: list[set[FractionAddend]] = []
				maximumPilesTotal: int = 0

				for mapShapeName in listMapKinds:
					listIndicesWithPilesTotal: list[IndexPilesTotal] = exclusionDataSource[mapShapeName][leafExcluderName][pileExcluderName][leafExcludedName]
					pilesTotal: int = listIndicesWithPilesTotal[-1]
					indicesOnly: list[IndexPilesTotal] = listIndicesWithPilesTotal[:-1]
					contiguousIndices: list[IndexPilesTotal] = _getContiguousFromStart(indicesOnly)
					setFractionAddends: set[FractionAddend] = {expressIndexAsFractionAddend(index, pilesTotal) for index in contiguousIndices} if len(contiguousIndices) >= 2 else set()
					listSetsFractionAddends.append(setFractionAddends)
					maximumPilesTotal = max(maximumPilesTotal, pilesTotal)

				commonFractionAddends: set[FractionAddend] = reduce(set[FractionAddend].intersection, listSetsFractionAddends) if listSetsFractionAddends and all(listSetsFractionAddends) else set()
				listFractionAddends: list[FractionAddend] = _sortedFractionAddends(commonFractionAddends)
				listFractionAddends.append((pos, Fraction(*maximumPilesTotal.as_integer_ratio()), 0))
				aggregatedExclusions[leafExcluderName][pileExcluderName][leafExcludedName] = listFractionAddends

	return aggregatedExclusions

def analyzeContiguousEndRelative(exclusionDataSource: dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[IndexPilesTotal]]]]]) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	"""Find common contiguous indices ending at pilesTotal-1 across all map shapes, expressed as fractions of pilesTotal."""
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	listMapKinds: list[MapKind] = list(exclusionDataSource.keys())
	mapShapeFirst: MapKind = listMapKinds[0]

	for leafExcluderName in exclusionDataSource[mapShapeFirst]:
		if any(leafExcluderName not in exclusionDataSource[mapShapeName] for mapShapeName in listMapKinds):
			continue
		aggregatedExclusions[leafExcluderName] = {}

		for pileExcluderName in exclusionDataSource[mapShapeFirst][leafExcluderName]:
			if any(pileExcluderName not in exclusionDataSource[mapShapeName][leafExcluderName] for mapShapeName in listMapKinds):
				continue
			aggregatedExclusions[leafExcluderName][pileExcluderName] = {}

			for leafExcludedName in exclusionDataSource[mapShapeFirst][leafExcluderName][pileExcluderName]:
				if any(leafExcludedName not in exclusionDataSource[mapShapeName][leafExcluderName][pileExcluderName] for mapShapeName in listMapKinds):
					continue

				listSetsFractionAddends: list[set[FractionAddend]] = []
				maximumPilesTotal: int = 0

				for mapShapeName in listMapKinds:
					listIndicesWithPilesTotal: list[IndexPilesTotal] = exclusionDataSource[mapShapeName][leafExcluderName][pileExcluderName][leafExcludedName]
					pilesTotal: int = listIndicesWithPilesTotal[-1]
					indicesOnly: list[IndexPilesTotal] = listIndicesWithPilesTotal[:-1]
					contiguousIndices: list[IndexPilesTotal] = _getContiguousFromEnd(indicesOnly, pilesTotal)
					setFractionAddends: set[FractionAddend] = {expressIndexAsFractionAddend(index, pilesTotal) for index in contiguousIndices} if len(contiguousIndices) >= 2 else set()
					listSetsFractionAddends.append(setFractionAddends)
					maximumPilesTotal = max(maximumPilesTotal, pilesTotal)

				commonFractionAddends: set[FractionAddend] = reduce(set[FractionAddend].intersection, listSetsFractionAddends) if listSetsFractionAddends and all(listSetsFractionAddends) else set()
				listFractionAddends: list[FractionAddend] = _sortedFractionAddends(commonFractionAddends)
				listFractionAddends.append((pos, Fraction(*maximumPilesTotal.as_integer_ratio()), 0))
				aggregatedExclusions[leafExcluderName][pileExcluderName][leafExcludedName] = listFractionAddends

	return aggregatedExclusions

def analyzeNonContiguousIndicesRelative(exclusionDataSource: dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[IndexPilesTotal]]]]]) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	"""Find common indices across all map shapes (contiguous or not), expressed as fractions of pilesTotal."""
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	listMapKinds: list[MapKind] = list(exclusionDataSource.keys())
	mapShapeFirst: MapKind = listMapKinds[0]

	for leafExcluderName in exclusionDataSource[mapShapeFirst]:
		if any(leafExcluderName not in exclusionDataSource[mapShapeName] for mapShapeName in listMapKinds):
			continue
		aggregatedExclusions[leafExcluderName] = {}

		for pileExcluderName in exclusionDataSource[mapShapeFirst][leafExcluderName]:
			if any(pileExcluderName not in exclusionDataSource[mapShapeName][leafExcluderName] for mapShapeName in listMapKinds):
				continue
			aggregatedExclusions[leafExcluderName][pileExcluderName] = {}

			for leafExcludedName in exclusionDataSource[mapShapeFirst][leafExcluderName][pileExcluderName]:
				if any(leafExcludedName not in exclusionDataSource[mapShapeName][leafExcluderName][pileExcluderName] for mapShapeName in listMapKinds):
					continue

				listSetsFractionAddends: list[set[FractionAddend]] = []
				maximumPilesTotal: int = 0

				for mapShapeName in listMapKinds:
					listIndicesWithPilesTotal: list[IndexPilesTotal] = exclusionDataSource[mapShapeName][leafExcluderName][pileExcluderName][leafExcludedName]
					pilesTotal: int = listIndicesWithPilesTotal[-1]
					indicesOnly: list[IndexPilesTotal] = listIndicesWithPilesTotal[:-1]
					setFractionAddends: set[FractionAddend] = {expressIndexAsFractionAddend(index, pilesTotal) for index in indicesOnly} if indicesOnly else set()
					listSetsFractionAddends.append(setFractionAddends)
					maximumPilesTotal = max(maximumPilesTotal, pilesTotal)

				commonFractionAddends: set[FractionAddend] = reduce(set[FractionAddend].intersection, listSetsFractionAddends) if listSetsFractionAddends and all(listSetsFractionAddends) else set()
				listFractionAddends: list[FractionAddend] = _sortedFractionAddends(commonFractionAddends)
				listFractionAddends.append((pos, Fraction(*maximumPilesTotal.as_integer_ratio()), 0))
				aggregatedExclusions[leafExcluderName][pileExcluderName][leafExcludedName] = listFractionAddends

	return aggregatedExclusions

# ======= Aggregate exclusion data =======

def aggregateExclusions(listAnalysisMethods: list[Callable[[dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[IndexPilesTotal]]]]]], dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]]]) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:  # noqa: E501
	"""{leafExcluder: {pileExcluder: {leafExcluded: listIndicesAsFractionAddends}}}."""
	listExclusions: list[dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]] = [analysisMethod(exclusionData) for analysisMethod in listAnalysisMethods]

	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}

	for leafExcluder, pileExcluder in CartesianProduct(functionsHeadDimensions, functionsHeadDimensions):
		dictionaryMerged: dict[strLeafExcluded, list[FractionAddend]] = updateExtendPolishDictionaryLists(
			*[dictionaryExclusions.get(leafExcluder.__name__, {}).get(pileExcluder.__name__, {}) for dictionaryExclusions in listExclusions]
			, destroyDuplicates=True, reorderLists=False)
		for leafExcludedName, listFractionAddends in dictionaryMerged.items():
			dictionaryMerged[leafExcludedName] = _sortedFractionAddends(listFractionAddends)
		aggregatedExclusions.setdefault(leafExcluder.__name__, {})[pileExcluder.__name__] = dictionaryMerged

	return aggregatedExclusions

def writeAggregatedExclusions(listAnalysisMethods: list[Callable[[dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[IndexPilesTotal]]]]]], dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]]]) -> list[PurePath]:
	listPathFilenames: list[PurePath] = []

	for leafExcluderName, pileExcluderData in aggregateExclusions(listAnalysisMethods).items():
		pythonSource: str = "from collections.abc import Callable\nfrom fractions import Fraction\nfrom operator import neg, pos\n\n"
		pythonSource += "type FractionAddend = tuple[Callable[[int], int], Fraction, int]\n\n"
		dataFormatted: str = pformat(pileExcluderData, indent=0, width=160, compact=True)
		dataFormatted = dataFormatted.replace('<built-in function neg>', 'neg').replace('<built-in function pos>', 'pos')
		pythonSource += f"dictionaryExclusions: dict[str, dict[str, list[FractionAddend]]] = {dataFormatted}\n"
		pathFilename: Path = pathExclusionData / f"aggregated{leafExcluderName}.py"
		writePython(pythonSource, pathFilename)
		listPathFilenames.append(PurePath(pathFilename))

	return listPathFilenames

# ======= Create exclusion dictionaries for elimination tools =======

def loadAggregatedExclusions() -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = {}
	for pathFilename in pathExclusionData.glob("aggregated*.py"):
		leafExcluderName: str = pathFilename.stem.removeprefix("aggregated")
		aggregatedExclusions[leafExcluderName] = importPathFilename2Identifier(pathFilename, "dictionaryExclusions")
	return aggregatedExclusions

def resolveIndexFromFractionAddend(fractionAddend: FractionAddend, pilesTotal: int) -> int:
	signOperator, indexFraction, addend = fractionAddend
	productNumerator: int = indexFraction.numerator * pilesTotal
	indexMagnitude: int = (productNumerator // indexFraction.denominator) + addend
	if signOperator is pos:
		return indexMagnitude
	return pilesTotal - indexMagnitude

def restructureAggregatedExclusionsForMapShape(dimensionsTotal: int, aggregatedExclusions: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]) -> tuple[dict[Leaf, dict[Pile, dict[Pile, list[Leaf]]]], dict[Pile, dict[Leaf, dict[Pile, list[Leaf]]]]]:
	leafDomains: dict[Leaf, range] = getDictionaryLeafDomains(EliminationState(mapShape=(2,) * dimensionsTotal))

	dictionaryLeafExcludedAtPileByPile: dict[Leaf, dict[Pile, dict[Pile, list[Leaf]]]] = {}
	dictionaryAtPileLeafExcludedByPile: dict[Pile, dict[Leaf, dict[Pile, list[Leaf]]]] = {}

	for leafExcluderName, pileExcluderData in aggregatedExclusions.items():
		for pileExcluderName, leafExcludedData in pileExcluderData.items():
				for leafExcludedName, listFractionAddends in leafExcludedData.items():
					leafExcluded: Leaf = dictionaryFunctionsByName[leafExcludedName](dimensionsTotal)

					domainOfLeafExcluded: list[Pile] = list(leafDomains[leafExcluded])
					pilesTotal: int = len(domainOfLeafExcluded)

					for fractionAddend in listFractionAddends[0:-1]:
						indexResolved: int = resolveIndexFromFractionAddend(fractionAddend, pilesTotal)
						pileExcluded: Pile = domainOfLeafExcluded[indexResolved]

						dictionaryLeafExcludedAtPileByPile.setdefault(leafExcluded, {}
							).setdefault(pileExcluded, {}
								).setdefault(dictionaryFunctionsByName[pileExcluderName](dimensionsTotal), []
									).append(dictionaryFunctionsByName[leafExcluderName](dimensionsTotal))
						dictionaryAtPileLeafExcludedByPile.setdefault(pileExcluded, {}
							).setdefault(leafExcluded, {}
								).setdefault(dictionaryFunctionsByName[pileExcluderName](dimensionsTotal), []
									).append(dictionaryFunctionsByName[leafExcluderName](dimensionsTotal))

	for leafExcluded in dictionaryLeafExcludedAtPileByPile:  # noqa: PLC0206
		for pileExcluded in dictionaryLeafExcludedAtPileByPile[leafExcluded]:
			for pileExcluder in dictionaryLeafExcludedAtPileByPile[leafExcluded][pileExcluded]:
				dictionaryLeafExcludedAtPileByPile[leafExcluded][pileExcluded][pileExcluder] = sorted(set(dictionaryLeafExcludedAtPileByPile[leafExcluded][pileExcluded][pileExcluder]))

	for pileExcluded in dictionaryAtPileLeafExcludedByPile:  # noqa: PLC0206
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

if __name__ == '__main__':
	listAnalysisMethods: list[Callable[[dict[MapKind, dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[IndexPilesTotal]]]]]], dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]]] = [
		analyzeNonContiguousIndicesRelative,
		analyzeContiguousStartAbsolute,
		analyzeContiguousEndAbsolute,
		analyzeContiguousStartRelative,
		analyzeContiguousEndRelative,
	]

	sys.stdout.write(f"{writeExclusionDataCollated() = }\n")
	sys.stdout.write(f"{writeAggregatedExclusions(listAnalysisMethods) = }\n")
	sys.stdout.write(f"{writeExclusionDictionaries() = }\n")

