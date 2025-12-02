"""Tests for elimination algorithm data functions.

These tests verify the correctness of functions in `mapFolding._e._data` that compute
leaf domains, pile ranges, and addend dictionaries for map folding elimination algorithms.

The test data is stored in `tests/dataSamples/A001417.py` and supports multiple `mapShape`
configurations. Currently, data exists for:
- (2,)*4 → 16 leaves (2d4)
- (2,)*5 → 32 leaves (2d5)
- (2,)*6 → 64 leaves (2d6)

When adding new test data for additional `mapShape` values, add the data to `A001417.py`
and the tests will automatically pick them up via parametrization.
"""

from collections.abc import Callable, Iterable, Sequence
from mapFolding._e import (
	getDictionaryLeafDomains, getDictionaryPileRanges, getDomainDimension一, getDomainDimension二, getDomainDimension首二,
	getDomain二一零and二一, getDomain二零and二, getLeafDomain, getPileRange)
from mapFolding._e.pinning2Dn import secondOrderLeaves, secondOrderPilings, thirdOrderPilings
from mapFolding.dataBaskets import EliminationState
from mapFolding.tests.dataSamples import (
	A001417, p2DnDomain3_2_首一_首零一, p2DnDomain5_4, p2DnDomain6_7_5_4, p2DnDomain7_6, p2DnDomain首二_首零二_首零一二_首一二)
from numpy.typing import NDArray
from types import ModuleType
import numpy
import pytest

dictionaryPinningFunctionDimensionsTotals: dict[Callable[[EliminationState], EliminationState], list[int]] = {
	thirdOrderPilings: [5, 6],
	secondOrderLeaves: [4, 5, 6],
	secondOrderPilings: [4, 5, 6],
}

listPinningFunctionDimensionsTotals: list[tuple[Callable[[EliminationState], EliminationState], int]] = [
	(pinningFunction, dimensionsTotal)
	for pinningFunction, listDimensionsTotals in dictionaryPinningFunctionDimensionsTotals.items()
	for dimensionsTotal in listDimensionsTotals
]

listCombinedDomainCases: list[tuple[str, Callable[[EliminationState], Sequence[tuple[int, ...]]], ModuleType, str]] = [
	( 'getDomainDimension一', getDomainDimension一, p2DnDomain3_2_首一_首零一, 'Domain3_2_首一_首零一' ),
	( 'getDomainDimension二', getDomainDimension二, p2DnDomain6_7_5_4, 'Domain6_7_5_4' ),
	( 'getDomainDimension首二', getDomainDimension首二, p2DnDomain首二_首零二_首零一二_首一二, 'Domain首二_首零二_首零一二_首一二' ),
	( 'getDomain二一零and二一', getDomain二一零and二一, p2DnDomain7_6, 'Domain7_6' ),
	( 'getDomain二零and二', getDomain二零and二, p2DnDomain5_4, 'Domain5_4' ),
]

listCombinedDomainDimensions: list[int] = [5, 6]

def verifyPinnedLeavesAgainstFoldings(state: EliminationState, arrayFoldings: NDArray[numpy.uint8]) -> tuple[int, int, int]:
	"""Verify pinned leaves cover all foldings without overlap.

	Parameters
	----------
	state : EliminationState
		State containing listPinnedLeaves to verify.
	arrayFoldings : NDArray[numpy.uint8]
		Known valid foldings to compare against.

	Returns
	-------
	verificationResults : tuple[int, int, int]
		Tuple of (rowsCovered, rowsTotal, countOverlappingDictionaries).
	"""
	rowsTotal: int = int(arrayFoldings.shape[0])
	listMasks: list[numpy.ndarray] = []

	for pinnedLeaves in state.listPinnedLeaves:
		maskMatches: numpy.ndarray = numpy.ones(rowsTotal, dtype=bool)
		for indexPile, leaf in pinnedLeaves.items():
			maskMatches = maskMatches & (arrayFoldings[:, indexPile] == leaf)
		listMasks.append(maskMatches)

	masksStacked: numpy.ndarray = numpy.column_stack(listMasks)
	coverageCountPerRow: numpy.ndarray = masksStacked.sum(axis=1)
	indicesOverlappedRows: numpy.ndarray = numpy.nonzero(coverageCountPerRow >= 2)[0]

	countOverlappingDictionaries: int = 0
	if indicesOverlappedRows.size > 0:
		for _indexMask, mask in enumerate(listMasks):
			if bool(mask[indicesOverlappedRows].any()):
				countOverlappingDictionaries += 1

	maskUnion = numpy.logical_or.reduce(listMasks)
	rowsCovered: int = int(maskUnion.sum())

	return rowsCovered, rowsTotal, countOverlappingDictionaries

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafDomainStartKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainStartKnown])
def test_getDictionaryLeafDomains_startValuesMatch(mapShape: tuple[int, ...]) -> None:
	state = EliminationState(mapShape=mapShape)
	dictionaryStartExpected = A001417.dictionaryLeafDomainStartKnown[mapShape]

	dictionaryActual = getDictionaryLeafDomains(state)

	assert len(dictionaryActual) == state.leavesTotal, (
		f"getDictionaryLeafDomains: dictionary length mismatch for {mapShape=}. "
		f"Expected {state.leavesTotal}, got {len(dictionaryActual)}."
	)

	for leaf in range(state.leavesTotal):
		rangeActual = dictionaryActual[leaf]
		startExpected = dictionaryStartExpected[leaf]
		assert rangeActual.start == startExpected, (
			f"getDictionaryLeafDomains: range.start mismatch at {leaf=} for {mapShape=}. "
			f"Expected start={startExpected}, got {rangeActual.start}."
		)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryLeafDomainStopKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryLeafDomainStopKnown])
def test_getDictionaryLeafDomains_stopValuesMatch(mapShape: tuple[int, ...]) -> None:
	state = EliminationState(mapShape=mapShape)
	dictionaryStopExpected = A001417.dictionaryLeafDomainStopKnown[mapShape]

	dictionaryActual = getDictionaryLeafDomains(state)

	for leaf in range(state.leavesTotal):
		rangeActual = dictionaryActual[leaf]
		stopExpected = dictionaryStopExpected[leaf]
		assert rangeActual.stop >= stopExpected, (
			f"getDictionaryLeafDomains: range.stop too small at {leaf=} for {mapShape=}. "
			f"Expected stop>={stopExpected}, got {rangeActual.stop}."
		)
		assert rangeActual.stop <= stopExpected + 1, (
			f"getDictionaryLeafDomains: range.stop too large at {leaf=} for {mapShape=}. "
			f"Expected stop<={stopExpected + 1}, got {rangeActual.stop}."
		)

@pytest.mark.parametrize("mapShape,leaf", [ (mapShape, leaf) for mapShape in A001417.dictionaryLeafDomainStartKnown for leaf in A001417.leafIndicesSample2d6[:5] if leaf < 2**len(mapShape) ], ids=lambda parameterValue: str(parameterValue))
def test_getLeafDomain_spotCheckStartValues(mapShape: tuple[int, ...], leaf: int) -> None:
	state = EliminationState(mapShape=mapShape)
	startExpected = A001417.dictionaryLeafDomainStartKnown[mapShape][leaf]

	rangeActual = getLeafDomain(state, leaf)

	assert rangeActual.start == startExpected, (
		f"getLeafDomain: start mismatch for {leaf=} with {mapShape=}. "
		f"Expected {startExpected}, got {rangeActual.start}."
	)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryPileRangesKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryPileRangesKnown])
def test_getDictionaryPileRanges_completeComparisonKnown(mapShape: tuple[int, ...]) -> None:
	state = EliminationState(mapShape=mapShape)
	dictionaryExpected = A001417.dictionaryPileRangesKnown[mapShape]

	dictionaryActual = getDictionaryPileRanges(state)

	assert len(dictionaryActual) == state.leavesTotal, (
		f"getDictionaryPileRanges: dictionary length mismatch for {mapShape=}. "
		f"Expected {state.leavesTotal}, got {len(dictionaryActual)}."
	)

	for pile in range(state.leavesTotal):
		assert dictionaryActual[pile] == dictionaryExpected[pile], (
			f"getDictionaryPileRanges: mismatch at {pile=} for {mapShape=}. "
			f"Expected {dictionaryExpected[pile]}, got {dictionaryActual[pile]}."
		)

@pytest.mark.parametrize("mapShape,pile", [ (mapShape, pile) for mapShape in A001417.dictionaryPileRangesKnown for pile in A001417.pileIndicesSample2d4 if pile < 2**len(mapShape) ], ids=lambda parameterValue: str(parameterValue))
def test_getPileRange_spotCheckValues(mapShape: tuple[int, ...], pile: int) -> None:
	state = EliminationState(mapShape=mapShape)
	listExpected = A001417.dictionaryPileRangesKnown[mapShape][pile]

	iterableActual: Iterable[int] = getPileRange(state, pile)
	listActual = list(iterableActual)

	assert listActual == listExpected, (
		f"getPileRange: mismatch for {pile=} with {mapShape=}. "
		f"Expected {listExpected}, got {listActual}."
	)

@pytest.mark.parametrize("mapShape", list(A001417.dictionaryPileRangesKnown), ids=[f"mapShape={shape}" for shape in A001417.dictionaryPileRangesKnown])
def test_getPileRange_consistencyWithDictionary(mapShape: tuple[int, ...]) -> None:
	"""Verify `getPileRange` results match `getDictionaryPileRanges` for all piles."""
	state = EliminationState(mapShape=mapShape)

	dictionaryFromFunction = getDictionaryPileRanges(state)

	for pile in range(state.leavesTotal):
		listFromGetPileRange = list(getPileRange(state, pile))
		listFromDictionary = dictionaryFromFunction[pile]

		assert listFromGetPileRange == listFromDictionary, (
			f"getPileRange inconsistent with getDictionaryPileRanges at {pile=} for {mapShape=}. "
			f"getPileRange returned {listFromGetPileRange}, dictionary has {listFromDictionary}."
		)

@pytest.mark.parametrize( "dimensionsTotal", listCombinedDomainDimensions, ids=[f"2d{dimensionsTotal}" for dimensionsTotal in listCombinedDomainDimensions] )
@pytest.mark.parametrize( "caseName,domainFunction,moduleExpected,attributeSuffix", listCombinedDomainCases, ids=[caseName for caseName, _domainFunction, _moduleExpected, _attributeSuffix in listCombinedDomainCases] )
def test_combinedDomainFunctions_matchVerificationData(caseName: str, domainFunction: Callable[[EliminationState], Sequence[tuple[int, ...]]], moduleExpected: ModuleType, attributeSuffix: str, dimensionsTotal: int) -> None:
	"""Compare combined domain functions against empirical verification datasets."""
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state = EliminationState(mapShape=mapShape)
	domainActual: tuple[tuple[int, ...], ...] = tuple(domainFunction(state))
	attributeName: str = f"list2D{dimensionsTotal}{attributeSuffix}"
	domainExpected: Sequence[tuple[int, ...]] = getattr(moduleExpected, attributeName)
	setActual: set[tuple[int, ...]] = set(domainActual)
	setExpected: set[tuple[int, ...]] = set(domainExpected)
	missingTuplesFull: list[tuple[int, ...]] = sorted(setExpected.difference(setActual))
	if missingTuplesFull:
		pytest.fail(
			f"{caseName}: combined domain missing {len(missingTuplesFull)} tuples for mapShape={mapShape}. "
			f"Missing samples: {missingTuplesFull[:3]}"
		)

@pytest.mark.parametrize("pinningFunction,dimensionsTotal", listPinningFunctionDimensionsTotals, ids=[f"{pinningFunction.__name__}_2d{dimensionsTotal}" for pinningFunction, dimensionsTotal in listPinningFunctionDimensionsTotals])
def test_pinningFunction_fullCoverage(loadArrayFoldings: Callable[[int], NDArray[numpy.uint8]], pinningFunction: Callable[[EliminationState], EliminationState], dimensionsTotal: int) -> None:
	mapShape = (2,) * dimensionsTotal
	state = EliminationState(mapShape=mapShape)
	arrayFoldings = loadArrayFoldings(dimensionsTotal)

	state = pinningFunction(state)

	rowsCovered, rowsTotal, _countOverlapping = verifyPinnedLeavesAgainstFoldings(state, arrayFoldings)

	assert rowsCovered == rowsTotal, (
		f"{pinningFunction.__name__}: incomplete coverage for {mapShape=}. "
		f"Covered {rowsCovered}/{rowsTotal} rows ({100 * rowsCovered / rowsTotal:.2f}%)."
	)

@pytest.mark.parametrize("pinningFunction,dimensionsTotal", listPinningFunctionDimensionsTotals, ids=[f"{pinningFunction.__name__}_2d{dimensionsTotal}" for pinningFunction, dimensionsTotal in listPinningFunctionDimensionsTotals])
def test_pinningFunction_noOverlap(loadArrayFoldings: Callable[[int], NDArray[numpy.uint8]], pinningFunction: Callable[[EliminationState], EliminationState], dimensionsTotal: int) -> None:
	mapShape = (2,) * dimensionsTotal
	state = EliminationState(mapShape=mapShape)
	arrayFoldings = loadArrayFoldings(dimensionsTotal)

	state = pinningFunction(state)

	_rowsCovered, _rowsTotal, countOverlappingDictionaries = verifyPinnedLeavesAgainstFoldings(state, arrayFoldings)

	assert countOverlappingDictionaries == 0, (
		f"{pinningFunction.__name__}: found {countOverlappingDictionaries} overlapping dictionaries for {mapShape=}. "
		f"Expected 0 overlaps."
	)

