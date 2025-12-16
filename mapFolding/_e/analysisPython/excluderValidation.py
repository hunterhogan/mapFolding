from collections.abc import Callable
from functools import cache
from hunterMakesPy import raiseIfNone
from mapFolding import exclude
from mapFolding._e import getLeafDomain, getPileRange, PinnedLeaves
from mapFolding._e._data import getDataFrameFoldings
from mapFolding._e.analysisPython.theExcluderBeast import (
	_fractionAddendToIndex, analyzeContiguousEndAbsolute, analyzeContiguousEndRelative, analyzeContiguousStartAbsolute,
	analyzeContiguousStartRelative, analyzeNonContiguousIndicesRelative, dictionaryFunctionsByName, FractionAddend,
	loadCollatedIndices, MapKind, restructureAggregatedExclusionsForMapShape, strLeafExcluded, strLeafExcluder,
	strPileExcluder)
from mapFolding._e.analysisPython.Z0Z_patternFinder import detectPermutationSpaceErrors, PermutationSpaceStatus
from mapFolding._e.pinIt import deconstructPinnedLeavesAtPile, deconstructPinnedLeavesByDomainOfLeaf
from mapFolding._e.pinning2Dn import pinPiles
from mapFolding.dataBaskets import EliminationState
import numpy
import sys

type Leaf = int
type Pile = int
type strPileExcluded = str

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

	for leafExcluderName in exclusionsFromAnalysisMethod:  # noqa: PLC0206
		leafExcluderFunction: Callable[[int], int] = dictionaryFunctionsByName[leafExcluderName]
		leafExcluder: int = leafExcluderFunction(dimensions)

		for pileExcluderName in exclusionsFromAnalysisMethod[leafExcluderName]:
			pileExcluderFunction: Callable[[int], int] = dictionaryFunctionsByName[pileExcluderName]
			pileExcluder: int = pileExcluderFunction(dimensions)

			for leafExcludedName in exclusionsFromAnalysisMethod[leafExcluderName][pileExcluderName]:
				leafExcludedFunction: Callable[[int], int] = dictionaryFunctionsByName[leafExcludedName]
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

				stateValidation = pinPiles(stateValidation, 1)

				pileRange: list[int] = list(getPileRange(stateValidation, pileExcluder))
				dictionaryDeconstructed: dict[int, PinnedLeaves] = deconstructPinnedLeavesAtPile(stateValidation.listPinnedLeaves[0], pileExcluder, pileRange)

				leavesPinnedWithExcluder: PinnedLeaves | None = dictionaryDeconstructed.get(leafExcluder)
				if leavesPinnedWithExcluder is None:
					continue

				listPinnedLeavesOther: list[PinnedLeaves] = [leavesPinned for leaf, leavesPinned in dictionaryDeconstructed.items() if leaf != leafExcluder]

				domainOfLeafExcluded: list[int] = list(getLeafDomain(stateValidation, leafExcluded))
				domainReduced: list[int] = list(exclude(domainOfLeafExcluded, listIndicesExcluded))

				listPinnedLeavesFromExcluder: list[PinnedLeaves] = deconstructPinnedLeavesByDomainOfLeaf(leavesPinnedWithExcluder, leafExcluded, domainReduced)

				stateValidation.listPinnedLeaves = listPinnedLeavesOther + listPinnedLeavesFromExcluder

				pinningCoverage: PermutationSpaceStatus = detectPermutationSpaceErrors(arrayFoldings, stateValidation.listPinnedLeaves)

				if pinningCoverage.rowsRequired < rowsTotal:
					listValidationErrors.append(
						f"{mapKind} {leafExcluderName}->{pileExcluderName}->{leafExcludedName}: {pinningCoverage.rowsRequired = }/{rowsTotal}")

				overlappingRowCount: int = int(pinningCoverage.indicesOverlappingRows.size)
				if overlappingRowCount > 0:
					listValidationErrors.append(
						f"{mapKind} {leafExcluderName}->{pileExcluderName}->{leafExcludedName}: {overlappingRowCount} overlapping rows")

	isValid: bool = len(listValidationErrors) == 0
	return (isValid, listValidationErrors)

def validateAnalysisMethod(analysisMethodCallable: Callable[[], dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]]) -> dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]]:
	listMapShapeNames: list[str] = list(loadCollatedIndices().keys())
	exclusionsFromMethod: dict[strLeafExcluder, dict[strPileExcluder, dict[strLeafExcluded, list[FractionAddend]]]] = analysisMethodCallable()
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

	colorReset = '\33[0m'
	if not errorsByMapShape:
		colorSuccess = '\33[92m'
		sys.stdout.write(f"{colorSuccess}{analysisMethodCallable.__name__} validated across {len(listMapShapeNames)} mapShapes{colorReset}\n")
	else:
		colorFailure = '\33[91m'
		sys.stdout.write(f"{colorFailure}{analysisMethodCallable.__name__} validation failed for {len(errorsByMapShape)} mapShapes{colorReset}\n")
		for mapShapeName, listErrors in errorsByMapShape.items():
			sys.stdout.write(f"{colorFailure}{mapShapeName}: {len(listErrors)} issues{colorReset}\n")
			for error in listErrors[0:3]:
				sys.stdout.write(f"{colorFailure}  {error}{colorReset}\n")

	return exclusionsFromMethod

if __name__ == '__main__':
	listAnalysisMethods = [
		analyzeNonContiguousIndicesRelative,
		analyzeContiguousStartAbsolute,
		analyzeContiguousEndAbsolute,
		analyzeContiguousStartRelative,
		analyzeContiguousEndRelative,
	]

	for analysisMethod in listAnalysisMethods:
		validateAnalysisMethod(analysisMethod)
